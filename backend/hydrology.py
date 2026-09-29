from __future__ import annotations

import logging
import math
from typing import Any, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# NumPy ≥ 2.0 compatibility — pysheds still calls np.in1d which was removed.
# np.isin is a strict superset: for 1-D arrays the return shape is identical.
# This patch must happen before pysheds is imported (module-load time).
# ---------------------------------------------------------------------------
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]

from pysheds.grid import Grid
from rasterio.features import geometry_mask, shapes
from rasterio.io import MemoryFile
from rasterio.transform import xy as raster_xy
from rasterio.warp import transform, transform_geom
from scipy.ndimage import label

from .opentopography import MAX_GRID_CELLS, validate_area_polygon

logger = logging.getLogger(__name__)
MIN_DEPRESSION_DEPTH_M = 1.0
MIN_DEPRESSION_CELLS = 3
MAX_DEM_CANDIDATES = 12
MAX_CATCHMENT_EVALUATIONS = 100
NODATA_VALUE = -9999.0


def _cell_areas_m2(dataset: Any) -> np.ndarray:
    transform = dataset.transform
    determinant = abs(transform.a * transform.e - transform.b * transform.d)
    if dataset.crs.is_geographic:
        latitudes = transform.f + (np.arange(dataset.height) + 0.5) * transform.e
        row_areas = determinant * 111_320.0 * 110_574.0 * np.cos(np.radians(latitudes))
        return np.broadcast_to(row_areas[:, np.newaxis], (dataset.height, dataset.width))

    unit_factor = dataset.crs.linear_units_factor[1]
    return np.full((dataset.height, dataset.width), determinant * unit_factor**2)


def _mask_geometry(mask: np.ndarray, affine: Any) -> dict[str, Any] | None:
    polygons = [
        geometry["coordinates"]
        for geometry, value in shapes(
            mask.astype("uint8"), mask=mask, transform=affine, connectivity=8
        )
        if value == 1
    ]
    if not polygons:
        return None
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def _to_wgs84(longitude: float, latitude: float, crs: Any) -> tuple[float, float]:
    if crs.is_geographic and crs.to_epsg() == 4326:
        return longitude, latitude
    longitudes, latitudes = transform(crs, "EPSG:4326", [longitude], [latitude])
    return longitudes[0], latitudes[0]


def _geometry_to_wgs84(geometry: dict[str, Any] | None, crs: Any) -> dict[str, Any] | None:
    if geometry is None or (crs.is_geographic and crs.to_epsg() == 4326):
        return geometry
    return transform_geom(crs, "EPSG:4326", geometry)


def analyze_dem_hydrology(
    dem_bytes: bytes,
    area_polygon: Sequence[Tuple[float, float]],
    dataset_name: str = "DEM",
) -> dict[str, Any]:
    validate_area_polygon(area_polygon)
    if not dem_bytes:
        raise ValueError("The elevation grid is empty.")

    with MemoryFile(dem_bytes) as source_memory:
        with source_memory.open() as source:
            if source.crs is None:
                raise ValueError("The DEM must have a coordinate reference system.")
            if source.width * source.height > MAX_GRID_CELLS:
                raise ValueError(f"The DEM exceeds the {MAX_GRID_CELLS:,}-cell processing limit.")
            if not math.isclose(source.transform.b, 0.0) or not math.isclose(source.transform.d, 0.0):
                raise ValueError("Rotated DEM grids are not supported.")

            dem = source.read(1, masked=True).astype("float64")
            valid = ~np.ma.getmaskarray(dem) & np.isfinite(dem.data)
            if valid.sum() < 16:
                raise ValueError("The selected area contains too few valid elevation cells.")

            polygon_geojson = {"type": "Polygon", "coordinates": [[list(point) for point in area_polygon]]}
            if source.crs.to_epsg() != 4326:
                from rasterio.warp import transform_geom

                polygon_geojson = transform_geom("EPSG:4326", source.crs, polygon_geojson)
            selected = geometry_mask(
                [polygon_geojson],
                out_shape=(source.height, source.width),
                transform=source.transform,
                invert=True,
            ) & valid
            if not selected.any():
                raise ValueError("The selected polygon does not overlap this DEM.")

            dem_values = np.where(valid, dem.data, NODATA_VALUE)
            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                dtype="float64",
                nodata=NODATA_VALUE,
                compress="deflate",
            )
            grid_memory = MemoryFile()
            try:
                with grid_memory.open(**profile) as grid_dataset:
                    grid_dataset.write(dem_values, 1)

                grid = Grid.from_raster(grid_memory.name)
                elevation = grid.read_raster(grid_memory.name)
                logger.info(
                    "DEM hydrology started: dataset=%s width=%d height=%d selected_cells=%d resolution=(%.3f, %.3f)",
                    dataset_name,
                    source.width,
                    source.height,
                    int(selected.sum()),
                    abs(source.transform.a),
                    abs(source.transform.e),
                )

                pit_filled = grid.fill_pits(elevation)
                filled = grid.fill_depressions(pit_filled)
                filled_values = np.asarray(filled, dtype="float64")
                depression_depth = np.where(valid, filled_values - dem_values, 0.0)
                depression_mask = (
                    valid
                    & (dem_values != NODATA_VALUE)
                    & (depression_depth >= MIN_DEPRESSION_DEPTH_M)
                )
                components, component_count = label(depression_mask, structure=np.ones((3, 3), dtype="uint8"))

                flow_direction = grid.flowdir(elevation, routing="d8")
                accumulated_cells = np.asarray(grid.accumulation(flow_direction, routing="d8"), dtype="float64")
                cell_areas = _cell_areas_m2(source)
                candidates = []
                for component_id in range(1, component_count + 1):
                    component = components == component_id
                    rows, columns = np.nonzero(component)
                    if len(rows) < MIN_DEPRESSION_CELLS:
                        continue
                    local_lowest = int(np.argmin(dem_values[component]))
                    row = int(rows[local_lowest])
                    column = int(columns[local_lowest])
                    if not selected[row, column]:
                        continue

                    basin_area = float(cell_areas[component].sum())
                    capacity = float((depression_depth[component] * cell_areas[component]).sum())
                    max_depth = float(depression_depth[component].max())
                    candidates.append(
                        {
                            "row": row,
                            "column": column,
                            "basinMask": component,
                            "basinAreaSqM": basin_area,
                            "estimatedVolumeM3": capacity,
                            "basinDepthM": max_depth,
                            "accumulatedCells": float(accumulated_cells[row, column]),
                        }
                    )

                candidates.sort(key=lambda candidate: candidate["estimatedVolumeM3"], reverse=True)
                candidates = candidates[:MAX_CATCHMENT_EVALUATIONS]
                fallback_used = not candidates
                if fallback_used:
                    selected_values = np.where(selected, dem_values, np.inf)
                    row, column = np.unravel_index(int(np.argmin(selected_values)), selected_values.shape)
                    candidates = [
                        {
                            "row": int(row),
                            "column": int(column),
                            "basinMask": np.zeros(selected.shape, dtype=bool),
                            "basinAreaSqM": 0.0,
                            "estimatedVolumeM3": 0.0,
                            "basinDepthM": 0.0,
                            "accumulatedCells": float(accumulated_cells[row, column]),
                            "fallbackRecommendation": True,
                        }
                    ]

                for candidate in candidates:
                    row, column = candidate["row"], candidate["column"]
                    catchment = np.asarray(
                        grid.catchment(
                            x=column,
                            y=row,
                            fdir=flow_direction,
                            xytype="index",
                            routing="d8",
                        ),
                        dtype=bool,
                    ) & valid
                    if not catchment.any():
                        catchment[row, column] = True
                    candidate["catchmentMask"] = catchment
                    candidate["estimatedCatchmentAreaSqM"] = float(cell_areas[catchment].sum())
                    candidate["catchmentIsTruncated"] = bool(
                        catchment[0, :].any()
                        or catchment[-1, :].any()
                        or catchment[:, 0].any()
                        or catchment[:, -1].any()
                    )
                    candidate["catchmentBoundary"] = _geometry_to_wgs84(
                        _mask_geometry(catchment, source.transform), source.crs
                    )
                    candidate["basinBoundary"] = _geometry_to_wgs84(
                        _mask_geometry(candidate["basinMask"], source.transform)
                        if candidate["basinMask"].any()
                        else None,
                        source.crs,
                    )
                    x, y = raster_xy(source.transform, row, column, offset="center")
                    longitude, latitude = _to_wgs84(float(x), float(y), source.crs)
                    candidate["pondCentroid"] = {"lon": round(longitude, 7), "lat": round(latitude, 7)}
                    candidate["pondElevation"] = round(float(dem_values[row, column]), 3)

                candidates.sort(
                    key=lambda candidate: (
                        candidate["estimatedVolumeM3"],
                        candidate["estimatedCatchmentAreaSqM"],
                    ),
                    reverse=True,
                )
                candidates = candidates[:MAX_DEM_CANDIDATES]
                volumes = [candidate["estimatedVolumeM3"] for candidate in candidates]
                catchment_areas = [candidate["estimatedCatchmentAreaSqM"] for candidate in candidates]
                depths = [candidate["basinDepthM"] for candidate in candidates]

                def normalize(value: float, values: list[float]) -> float:
                    low, high = min(values), max(values)
                    return 1.0 if high - low <= 1e-9 else (value - low) / (high - low)

                payloads = []
                for rank, candidate in enumerate(candidates, start=1):
                    fallback = candidate.get("fallbackRecommendation", False)
                    score = (
                        0.45 * normalize(candidate["estimatedVolumeM3"], volumes)
                        + 0.30 * normalize(candidate["estimatedCatchmentAreaSqM"], catchment_areas)
                        + 0.25 * normalize(candidate["basinDepthM"], depths)
                    ) if not fallback else 0.0
                    payload = {
                        "rank": rank,
                        "recommended": rank == 1,
                        "fallbackRecommendation": fallback,
                        "score": round(score, 4),
                        "pondElevation": candidate["pondElevation"],
                        "pondCentroid": candidate["pondCentroid"],
                        "basinAreaSqM": round(candidate["basinAreaSqM"], 2),
                        "estimatedCatchmentAreaSqM": round(candidate["estimatedCatchmentAreaSqM"], 2),
                        "basinDepthM": round(candidate["basinDepthM"], 3),
                        "estimatedVolumeM3": round(candidate["estimatedVolumeM3"], 2),
                        "compactnessScore": 0.0,
                        "basinBoundary": candidate["basinBoundary"],
                        "catchmentBoundary": candidate["catchmentBoundary"],
                        "catchmentIsTruncated": candidate["catchmentIsTruncated"],
                        "contributingCellCount": int(round(candidate["estimatedCatchmentAreaSqM"] / max(cell_areas.mean(), 1.0))),
                        "fallbackRecommendation": fallback,
                    }
                    payloads.append(payload)

                minimum = float(dem_values[selected].min())
                maximum = float(dem_values[selected].max())
                best = payloads[0]
                result = {
                    "status": "ok",
                    "method": "pysheds D8 flow routing with depression fill-depth storage, connected depression components, and upstream catchment delineation",
                    "fallbackRecommendation": fallback_used,
                    "contourInterval": 0.0,
                    "pondElevation": best["pondElevation"],
                    "pondCentroid": best["pondCentroid"],
                    "basinAreaSqM": best["basinAreaSqM"],
                    "estimatedCatchmentAreaSqM": best["estimatedCatchmentAreaSqM"],
                    "basinDepthM": best["basinDepthM"],
                    "estimatedVolumeM3": best["estimatedVolumeM3"],
                    "compactnessScore": 0.0,
                    "confidenceScore": best["score"],
                    "candidateContours": 0,
                    "basinBoundary": best["basinBoundary"],
                    "terrainSummary": {
                        "minElevation": round(minimum, 3),
                        "maxElevation": round(maximum, 3),
                        "contourCount": 0,
                        "usableLoopCount": 0,
                        "basinCandidateCount": 0 if fallback_used else len(payloads),
                        "demDepressionComponentCount": component_count,
                        "flowRouting": "D8",
                    },
                    "riverAvoidance": {
                        "enabled": False,
                        "filteredRiverLikeLoopCount": 0,
                        "preference": "DEM-based depressions are ranked by storage capacity, contributing area, and depth.",
                    },
                    "pondCandidates": payloads,
                    "alternativeCandidates": [
                        {key: value for key, value in candidate.items() if key not in ("rank", "recommended")}
                        for candidate in payloads[1:]
                    ],
                    "elevationDataset": dataset_name,
                    "fallbackRecommendation": fallback_used,
                }
                logger.info(
                    "DEM hydrology complete: depression_components=%d candidates=%d fallback=%s recommended=(%.6f, %.6f) catchment=%.0f m^2 storage=%.2f m^3",
                    component_count,
                    len(payloads),
                    fallback_used,
                    best["pondCentroid"]["lat"],
                    best["pondCentroid"]["lon"],
                    best["estimatedCatchmentAreaSqM"],
                    best["estimatedVolumeM3"],
                )
                return result
            finally:
                grid_memory.close()