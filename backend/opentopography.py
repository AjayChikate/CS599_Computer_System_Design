from __future__ import annotations

import math
import logging
import os
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence, Tuple
from xml.etree import ElementTree as ET

import contourpy
import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.warp import transform as transform_coordinates, transform_geom
from dotenv import load_dotenv

from .geometry import point_in_polygon, shoelace_area

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("backend").setLevel(logging.INFO)

OPENTOPOGRAPHY_URL = "https://portal.opentopography.org/API/globaldem"
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
MAX_AREA_KM2 = 25.0
MAX_DEM_BYTES = 32 * 1024 * 1024
MAX_GRID_CELLS = 2_000_000
MAX_POLYGON_VERTICES = 200
MAX_CONTOUR_LEVELS = 120
MAX_GENERATED_CONTOURS = 1_000
MAX_CONTOUR_POINTS = 100_000


def validate_area_polygon(area_polygon: Sequence[Tuple[float, float]]) -> tuple[float, float, float, float]:
    if len(area_polygon) < 4 or area_polygon[0] != area_polygon[-1]:
        raise ValueError("Draw a closed polygon before generating contours.")
    if len(area_polygon) > MAX_POLYGON_VERTICES:
        raise ValueError(f"The selected polygon may have at most {MAX_POLYGON_VERTICES} vertices.")

    for longitude, latitude in area_polygon:
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError("The selected polygon contains an invalid coordinate.")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("The selected polygon must use valid WGS84 coordinates.")

    west = min(point[0] for point in area_polygon)
    east = max(point[0] for point in area_polygon)
    south = min(point[1] for point in area_polygon)
    north = max(point[1] for point in area_polygon)
    if east - west > 180:
        raise ValueError("Selections crossing the antimeridian are not supported.")

    width_km = (east - west) * 111.32 * math.cos(math.radians((north + south) / 2))
    height_km = (north - south) * 110.574
    area_km2 = width_km * height_km
    if area_km2 <= 0.01:
        raise ValueError("Select a larger area (at least 0.01 km²) to get useful terrain coverage.")
    if area_km2 > MAX_AREA_KM2:
        logger.warning("Rejected selected bounding area: %.2f km² exceeds %.2f km² app limit", area_km2, MAX_AREA_KM2)
        raise ValueError(f"The selected map area is too large. The maximum is {MAX_AREA_KM2:g} km².")
    if east - west < 0.0005 or north - south < 0.0005:
        raise ValueError("The selected map area is too narrow for 30 m elevation data.")
    return west, south, east, north


def _configured_api_key() -> str:
    return os.getenv("API_Key", "").strip()


def fetch_global_dem(area_polygon: Sequence[Tuple[float, float]], dataset: str = "COP30") -> bytes:
    west, south, east, north = validate_area_polygon(area_polygon)
    width_km = (east - west) * 111.32 * math.cos(math.radians((north + south) / 2))
    height_km = (north - south) * 110.574
    area_km2 = width_km * height_km
    api_key = _configured_api_key()
    if not api_key:
        logger.error("DEM request blocked: API_Key is not configured in the server environment")
        raise ValueError("OpenTopography API key is missing. Set API_Key in the project .env file and restart the app.")
    if dataset not in {"COP30", "SRTMGL1", "AW3D30"}:
        logger.error("DEM request blocked: unsupported dataset=%s", dataset)
        raise ValueError("Unsupported global elevation dataset.")

    params = {
        "demtype": dataset,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key.strip(),
    }
    request_started = perf_counter()
    logger.info(
        "DEM request started: dataset=%s bbox=[%.5f, %.5f, %.5f, %.5f] bbox_area=%.2f km^2",
        dataset,
        west,
        south,
        east,
        north,
        area_km2,
    )
    try:
        with requests.get(OPENTOPOGRAPHY_URL, params=params, stream=True, timeout=(10, 120)) as response:
            logger.info(
                "DEM provider responded: status=%d response_time=%.2f s",
                response.status_code,
                perf_counter() - request_started,
            )
            if response.status_code == 401:
                logger.error("DEM request rejected: provider returned HTTP 401")
                raise ValueError("OpenTopography rejected the API key. Check the key and try again.")
            if response.status_code == 204:
                logger.warning("DEM request returned no data for selected bounds")
                raise ValueError("OpenTopography has no elevation data for this area.")
            if response.status_code == 429:
                logger.warning("DEM request rate limited by provider")
                raise ValueError("OpenTopography rate limit reached. Try again later.")
            if response.status_code != 200:
                logger.error("DEM request failed: provider returned HTTP %d", response.status_code)
                raise ValueError(f"OpenTopography returned HTTP {response.status_code}.")

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    logger.error("DEM response has an invalid Content-Length header")
                    raise ValueError("OpenTopography returned an invalid response size.") from exc
                if declared_size > MAX_DEM_BYTES:
                    logger.warning("DEM response rejected: declared size=%d bytes, limit=%d", declared_size, MAX_DEM_BYTES)
                    raise ValueError("The downloaded elevation grid exceeds the 32 MB processing limit.")

            chunks = []
            total_bytes = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_DEM_BYTES:
                    logger.warning("DEM response exceeded the %d-byte streaming limit", MAX_DEM_BYTES)
                    raise ValueError("The downloaded elevation grid exceeds the 32 MB processing limit.")
                chunks.append(chunk)
            dem_bytes = b"".join(chunks)
            logger.info(
                "DEM download complete: bytes=%d total_time=%.2f s",
                len(dem_bytes),
                perf_counter() - request_started,
            )
            return dem_bytes
    except requests.Timeout as exc:
        logger.warning("DEM request timed out after %.2f s", perf_counter() - request_started)
        raise ValueError("OpenTopography took too long to respond. Try again with a smaller area.") from exc
    except requests.RequestException as exc:
        logger.error("DEM network request failed: %s", type(exc).__name__)
        raise ValueError("Could not connect to OpenTopography. Check the network and try again.") from exc


def _inside_or_on_edge(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    if point_in_polygon(point, polygon):
        return True
    x, y = point
    for start, end in zip(polygon, polygon[1:]):
        x1, y1 = start
        x2, y2 = end
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) <= 1e-10
            and min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10
            and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10
        ):
            return True
    return False


def _contour_interval(minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span <= 0:
        return 1.0
    return max(1.0, math.ceil(span / (MAX_CONTOUR_LEVELS * 5.0)) * 5.0)


def generate_contour_kml(
    dem_bytes: bytes, area_polygon: Sequence[Tuple[float, float]], dataset: str = "COP30"
) -> bytes:
    validate_area_polygon(area_polygon)
    if not dem_bytes:
        raise ValueError("OpenTopography returned an empty elevation grid.")

    processing_started = perf_counter()
    logger.info("DEM processing started: dataset=%s downloaded_bytes=%d", dataset, len(dem_bytes))
    try:
        with MemoryFile(dem_bytes) as memory_file:
            with memory_file.open() as dataset_file:
                if dataset_file.crs is None:
                    raise ValueError("The elevation grid must have a coordinate reference system.")
                transform = dataset_file.transform
                if not math.isclose(transform.b, 0.0) or not math.isclose(transform.d, 0.0):
                    raise ValueError("The downloaded elevation grid uses an unsupported rotated coordinate system.")
                if dataset_file.width * dataset_file.height > MAX_GRID_CELLS:
                    logger.warning(
                        "DEM raster rejected: cells=%d limit=%d",
                        dataset_file.width * dataset_file.height,
                        MAX_GRID_CELLS,
                    )
                    raise ValueError("The downloaded elevation grid is too detailed to process safely.")

                analysis_polygon = list(area_polygon)
                if dataset_file.crs.to_epsg() != 4326:
                    area_geometry = transform_geom(
                        "EPSG:4326",
                        dataset_file.crs,
                        {"type": "Polygon", "coordinates": [[list(point) for point in area_polygon]]},
                    )
                    analysis_polygon = [tuple(point) for point in area_geometry["coordinates"][0]]

                elevations = np.ma.masked_invalid(dataset_file.read(1, masked=True).astype("float64"))
                if elevations.count() < 16:
                    logger.warning("DEM raster rejected: only %d valid elevation cells", elevations.count())
                    raise ValueError("The downloaded elevation grid contains too few valid cells.")
                x_coordinates = transform.c + (np.arange(dataset_file.width) + 0.5) * transform.a
                y_coordinates = transform.f + (np.arange(dataset_file.height) + 0.5) * transform.e
                if transform.a <= 0:
                    raise ValueError("The downloaded elevation grid has an unsupported west-to-east orientation.")
                if y_coordinates[0] > y_coordinates[-1]:
                    y_coordinates = y_coordinates[::-1]
                    elevations = elevations[::-1, :]

                minimum = float(elevations.min())
                maximum = float(elevations.max())
                interval = _contour_interval(minimum, maximum)
                first_level = math.ceil(minimum / interval) * interval
                levels = np.arange(first_level, maximum, interval, dtype="float64")
                logger.info(
                    "DEM raster opened: width=%d height=%d valid_cells=%d elevation_range=%.2f..%.2f m contour_interval=%.2f m",
                    dataset_file.width,
                    dataset_file.height,
                    elevations.count(),
                    minimum,
                    maximum,
                    interval,
                )

                generator = contourpy.contour_generator(
                    x=x_coordinates,
                    y=y_coordinates,
                    z=elevations,
                    name="serial",
                    line_type="Separate",
                )
                root = ET.Element("{http://www.opengis.net/kml/2.2}kml")
                document = ET.SubElement(root, "{http://www.opengis.net/kml/2.2}Document")
                ET.SubElement(document, "{http://www.opengis.net/kml/2.2}name").text = "OpenTopography contours"
                ET.SubElement(document, "{http://www.opengis.net/kml/2.2}description").text = (
                    f"Derived from {dataset} elevation data accessed through the OpenTopography Global Datasets API. "
                    "OpenTopography API and source dataset attribution applies."
                )
                total_points = 0
                total_contours = 0

                for level in levels:
                    for line in generator.lines(float(level)):
                        if len(line) < 4:
                            continue
                        first = (float(line[0][0]), float(line[0][1]))
                        last = (float(line[-1][0]), float(line[-1][1]))
                        if math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-10:
                            continue
                        coordinates = [(float(point[0]), float(point[1])) for point in line]
                        if not all(_inside_or_on_edge(point, analysis_polygon) for point in coordinates):
                            continue
                        total_points += len(coordinates)
                        total_contours += 1
                        if total_points > MAX_CONTOUR_POINTS:
                            logger.warning("Contour generation exceeded %d points", MAX_CONTOUR_POINTS)
                            raise ValueError("The generated contours exceed the 100,000-point processing limit.")
                        if total_contours > MAX_GENERATED_CONTOURS:
                            logger.warning("Contour generation exceeded %d closed contours", MAX_GENERATED_CONTOURS)
                            raise ValueError("The selected area contains too many closed contours to analyze safely.")

                        placemark = ET.SubElement(document, "{http://www.opengis.net/kml/2.2}Placemark")
                        ET.SubElement(placemark, "{http://www.opengis.net/kml/2.2}name").text = f"{level:.3f}"
                        line_string = ET.SubElement(placemark, "{http://www.opengis.net/kml/2.2}LineString")
                        ET.SubElement(line_string, "{http://www.opengis.net/kml/2.2}tessellate").text = "1"
                        if dataset_file.crs.to_epsg() == 4326:
                            geographic_coordinates = coordinates
                        else:
                            longitudes, latitudes = transform_coordinates(
                                dataset_file.crs,
                                "EPSG:4326",
                                [point[0] for point in coordinates],
                                [point[1] for point in coordinates],
                            )
                            geographic_coordinates = list(zip(longitudes, latitudes))
                        ET.SubElement(line_string, "{http://www.opengis.net/kml/2.2}coordinates").text = " ".join(
                            f"{longitude:.7f},{latitude:.7f},0" for longitude, latitude in geographic_coordinates
                        )

                kml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                logger.info(
                    "Contour KML generated: closed_contours=%d coordinate_points=%d kml_bytes=%d processing_time=%.2f s",
                    total_contours,
                    total_points,
                    len(kml_bytes),
                    perf_counter() - processing_started,
                )
                return kml_bytes
    except rasterio.errors.RasterioError as exc:
        logger.error("DEM processing failed: invalid GeoTIFF (%s)", type(exc).__name__)
        raise ValueError("OpenTopography did not return a readable GeoTIFF elevation grid.") from exc


def analyze_dem_area(
    area_polygon: Sequence[Tuple[float, float]], dataset: str = "COP30"
) -> tuple[bytes, dict[str, Any]]:
    from .parsing import parse_kml_text
    from .hydrology import analyze_dem_hydrology

    analysis_started = perf_counter()
    logger.info("Terrain analysis started: dataset=%s polygon_vertices=%d", dataset, len(area_polygon))
    dem_bytes = fetch_global_dem(area_polygon, dataset)
    kml_bytes = generate_contour_kml(dem_bytes, area_polygon, dataset)
    raw_contours = parse_kml_text(kml_bytes.decode("utf-8"))
    logger.info("Terrain analysis received %d closed contours", len(raw_contours))
    result = analyze_dem_hydrology(dem_bytes, area_polygon, dataset)
    if result.get("fallbackRecommendation"):
        logger.warning(
            "Fallback recommendation: lowest DEM point=(%.6f, %.6f); storage=0 m^3; D8 catchment area=%.0f m^2",
            result["pondCentroid"]["lat"],
            result["pondCentroid"]["lon"],
            result["estimatedCatchmentAreaSqM"],
        )
    else:
        logger.info(
            "Terrain analysis complete: candidates=%d recommended=(%.6f, %.6f) catchment=%.0f m^2 storage=%.2f m^3",
            len(result["pondCandidates"]),
            result["pondCentroid"]["lat"],
            result["pondCentroid"]["lon"],
            result["estimatedCatchmentAreaSqM"],
            result["estimatedVolumeM3"],
        )
    logger.info("Terrain analysis pipeline finished in %.2f s", perf_counter() - analysis_started)
    return kml_bytes, result


def _fallback_dem_result(
    dem_bytes: bytes,
    area_polygon: Sequence[Tuple[float, float]],
    dataset_name: str,
    raw_contours: list[dict[str, Any]],
) -> dict[str, Any]:
    with MemoryFile(dem_bytes) as memory_file:
        with memory_file.open() as dataset_file:
            elevations = np.ma.masked_invalid(dataset_file.read(1, masked=True).astype("float64"))
            transform = dataset_file.transform
            x_coordinates = transform.c + (np.arange(dataset_file.width) + 0.5) * transform.a
            y_coordinates = transform.f + (np.arange(dataset_file.height) + 0.5) * transform.e
            longitude_grid = np.broadcast_to(x_coordinates, elevations.shape)
            latitude_grid = np.broadcast_to(y_coordinates[:, np.newaxis], elevations.shape)
            inside = np.zeros(elevations.shape, dtype=bool)
            for start, end in zip(area_polygon, area_polygon[1:]):
                x1, y1 = start
                x2, y2 = end
                crosses = (y1 > latitude_grid) != (y2 > latitude_grid)
                boundary_x = (x2 - x1) * (latitude_grid - y1) / ((y2 - y1) + 1e-30) + x1
                inside ^= crosses & (longitude_grid < boundary_x)

            valid = inside & ~np.ma.getmaskarray(elevations) & np.isfinite(elevations.data)
            if not valid.any():
                raise ValueError("No valid elevation cells fall inside the selected polygon.")
            ranked_values = np.where(valid, elevations.data, np.inf)
            row, column = np.unravel_index(int(np.argmin(ranked_values)), ranked_values.shape)
            pond_longitude = float(longitude_grid[row, column])
            pond_latitude = float(latitude_grid[row, column])
            pond_elevation = float(elevations.data[row, column])
            valid_elevations = elevations.data[valid]

    center_lon = sum(point[0] for point in area_polygon[:-1]) / (len(area_polygon) - 1)
    center_lat = sum(point[1] for point in area_polygon[:-1]) / (len(area_polygon) - 1)
    x_scale = 111_320.0 * math.cos(math.radians(center_lat))
    y_scale = 110_574.0
    local_polygon = [
        ((longitude - center_lon) * x_scale, (latitude - center_lat) * y_scale)
        for longitude, latitude in area_polygon
    ]
    area_proxy_sq_m = shoelace_area(local_polygon)
    coordinates = [[round(longitude, 7), round(latitude, 7)] for longitude, latitude in area_polygon]
    boundary = {"type": "Polygon", "coordinates": [coordinates]}
    candidate = {
        "rank": 1,
        "recommended": True,
        "fallbackRecommendation": True,
        "score": 0.0,
        "pondElevation": round(pond_elevation, 3),
        "pondCentroid": {"lon": round(pond_longitude, 7), "lat": round(pond_latitude, 7)},
        "basinAreaSqM": 0.0,
        "estimatedCatchmentAreaSqM": round(area_proxy_sq_m, 2),
        "basinDepthM": 0.0,
        "estimatedVolumeM3": 0.0,
        "compactnessScore": 0.0,
        "basinBoundary": boundary,
    }
    minimum = float(valid_elevations.min())
    maximum = float(valid_elevations.max())
    return {
        "status": "ok",
        "fallbackRecommendation": True,
        "contourInterval": 0.0,
        "pondElevation": candidate["pondElevation"],
        "pondCentroid": candidate["pondCentroid"],
        "estimatedCatchmentAreaSqM": candidate["estimatedCatchmentAreaSqM"],
        "basinAreaSqM": 0.0,
        "basinDepthM": 0.0,
        "estimatedVolumeM3": 0.0,
        "compactnessScore": 0.0,
        "confidenceScore": 0.0,
        "candidateContours": len(raw_contours),
        "method": (
            "fallback: no enclosed depression was found; recommend the lowest valid DEM cell in the "
            "selected area. The selected area is shown as a search-area proxy, not a delineated catchment, "
            "and storage is zero because no basin capacity could be established"
        ),
        "riverAvoidance": {
            "enabled": True,
            "filteredRiverLikeLoopCount": 0,
            "preference": "Fallback used because no compact enclosed basin could be identified.",
        },
        "basinBoundary": boundary,
        "terrainSummary": {
            "minElevation": round(minimum, 3),
            "maxElevation": round(maximum, 3),
            "contourCount": len(raw_contours),
            "usableLoopCount": 0,
            "basinCandidateCount": 0,
        },
        "pondCandidates": [candidate],
        "alternativeCandidates": [],
        "elevationDataset": dataset_name,
    }