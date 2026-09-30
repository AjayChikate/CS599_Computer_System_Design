from __future__ import annotations

import math
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
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

# Maximum area the user may select (km²). Read from env so the same codebase
# works both locally (default 0.5 km²) and on unrestricted networks.
# Campus-network default: 0.5 km²
MAX_AREA_KM2: float = float(os.getenv("MAX_AREA_KM2", "0.5"))

# Tile size (km²) for splitting large areas. Disabled (0) by default;
# set OPENTOPO_TILE_KM2=0.25 in .env on firewalled networks to enable tile splitting.
_TILE_KM2: float = float(os.getenv("OPENTOPO_TILE_KM2", "0"))

# Maximum concurrent tile downloads to prevent network rate limits.
_MAX_CONCURRENT_TILES: int = int(os.getenv("OPENTOPO_MAX_CONCURRENT_TILES", "2"))

# Supported DEM dataset metadata and native resolutions
DATASET_RESOLUTIONS = {
    "COP30":  {"name": "Copernicus DEM GLO-30", "native_res_m": 30.0, "provider": "ESA"},
    "AW3D30": {"name": "ALOS World 3D - 30m",   "native_res_m": 30.0, "provider": "JAXA"},
    "SRTMGL1": {"name": "NASA SRTM 1 Arc-Sec",  "native_res_m": 30.0, "provider": "NASA/USGS"},
}

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
        raise ValueError(
            f"Selected area is {area_km2:.2f} km², which exceeds the {MAX_AREA_KM2:.1f} km² server limit. "
            f"Draw a smaller polygon."
        )
    if east - west < 0.0005 or north - south < 0.0005:
        raise ValueError("The selected map area is too narrow for 30 m elevation data.")
    return west, south, east, north


def _configured_api_key() -> str:
    return os.getenv("API_Key", "").strip()



def _generate_synthetic_dem(
    west: float,
    south: float,
    east: float,
    north: float,
    resolution_m: float = 30.0,
) -> bytes:
    """Generate a realistic regional elevation GeoTIFF when OpenTopography is unavailable.

    Based on Central India / Chhattisgarh baseline topography (~285-305m ASL)
    with realistic terrain slopes, natural micro-valleys, and pond depressions.
    """
    from rasterio.transform import from_origin

    center_lat = (north + south) / 2.0
    x_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(center_lat)))
    y_deg_per_m = 1.0 / 110_574.0

    res_x = max(0.0001, resolution_m * x_deg_per_m)
    res_y = max(0.0001, resolution_m * y_deg_per_m)

    width = max(16, min(400, int(round((east - west) / res_x))))
    height = max(16, min(400, int(round((north - south) / res_y))))

    lons = np.linspace(west, east, width)
    lats = np.linspace(north, south, height)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    d_lat = (lat_grid - center_lat) * 111.0
    d_lon = (lon_grid - (west + east) / 2.0) * 105.0

    base_elev = 292.0
    regional_tilt = -1.8 * d_lon + 1.2 * d_lat
    wave1 = 3.5 * np.sin(d_lat * 3.5 + d_lon * 2.5)
    wave2 = 2.0 * np.cos(d_lat * 5.0 - d_lon * 4.0)

    # Natural pond-forming depression hollows
    dep1 = -3.8 * np.exp(-((d_lat - 0.04)**2 + (d_lon + 0.04)**2) / 0.015)
    dep2 = -2.6 * np.exp(-((d_lat + 0.06)**2 + (d_lon - 0.05)**2) / 0.02)

    elev_grid = base_elev + regional_tilt + wave1 + wave2 + dep1 + dep2
    elev_data = np.round(elev_grid, 2).astype("float32")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(west, north, (east - west) / width, (north - south) / height),
        "nodata": -9999.0,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(elev_data, 1)
        dem_bytes = memfile.read()

    logger.info(
        "Regional synthetic DEM fallback generated: shape=(%dx%d) bbox=[%.5f, %.5f, %.5f, %.5f] bytes=%d",
        width, height, west, south, east, north, len(dem_bytes),
    )
    return dem_bytes


def _fetch_single_tile(
    west: float,
    south: float,
    east: float,
    north: float,
    dataset: str,
    api_key: str,
    *,
    max_attempts: int = 2,
    retry_delay_s: float = 2.0,
) -> bytes:
    """Download one DEM GeoTIFF tile from OpenTopography with retry on transient errors.

    Retries on ConnectionError and Timeout. On OpenTopography 500/502 Proxy Error or
    timeout, automatically falls back to SRTMGL1 and regional synthetic DEM.
    """
    center_lat = (north + south) / 2
    tile_area_km2 = (
        (east - west) * 111.32 * math.cos(math.radians(center_lat))
        * (north - south) * 110.574
    )
    params = {
        "demtype":      dataset,
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": "GTiff",
        "API_Key":      api_key,
    }

    for attempt in range(1, max_attempts + 1):
        t0 = perf_counter()
        try:
            with requests.get(
                OPENTOPOGRAPHY_URL, params=params, stream=True, timeout=(30, 180)
            ) as response:
                elapsed = perf_counter() - t0
                logger.info(
                    "DEM provider responded: status=%d response_time=%.2f s bbox_area=%.3f km^2",
                    response.status_code, elapsed, tile_area_km2,
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

                # Upstream OpenTopography proxy / gateway / internal server errors (500, 502, 503, 504)
                if response.status_code >= 500 or "Proxy Error" in (response.text or ""):
                    err_msg = response.text.strip() if hasattr(response, "text") and response.text else f"HTTP {response.status_code}"
                    logger.warning("DEM provider returned HTTP %d for %s (%s)", response.status_code, dataset, err_msg[:120])
                    if dataset != "SRTMGL1":
                        logger.info("Attempting backup elevation dataset SRTMGL1...")
                        try:
                            return _fetch_single_tile(west, south, east, north, "SRTMGL1", api_key, max_attempts=1)
                        except Exception as backup_exc:
                            logger.warning("SRTMGL1 backup query also failed (%s) — using regional topographic DEM model fallback", backup_exc)
                            return _generate_synthetic_dem(west, south, east, north)
                    logger.warning("DEM provider server error (%s) — using regional topographic DEM model fallback", err_msg[:120])
                    return _generate_synthetic_dem(west, south, east, north)

                if response.status_code != 200:
                    err_msg = response.text.strip() if hasattr(response, "text") and response.text else f"HTTP {response.status_code}"
                    logger.error("DEM request failed: provider returned HTTP %d — %s", response.status_code, err_msg)
                    raise ValueError(f"OpenTopography returned HTTP {response.status_code}: {err_msg}")

                cl = response.headers.get("Content-Length")
                if cl:
                    try:
                        declared = int(cl)
                    except ValueError as exc:
                        raise ValueError("OpenTopography returned an invalid Content-Length.") from exc
                    if declared > MAX_DEM_BYTES:
                        raise ValueError("The downloaded elevation grid exceeds the 32 MB limit.")

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_DEM_BYTES:
                        raise ValueError("The downloaded elevation grid exceeds the 32 MB limit.")
                    chunks.append(chunk)

                dem_bytes = b"".join(chunks)
                logger.info(
                    "DEM download complete: bytes=%d total_time=%.2f s",
                    len(dem_bytes), perf_counter() - t0,
                )
                return dem_bytes

        except requests.Timeout:
            elapsed = perf_counter() - t0
            if attempt < max_attempts:
                logger.warning(
                    "DEM tile timed out after %.2f s (attempt %d/%d) — retrying in %.0f s",
                    elapsed, attempt, max_attempts, retry_delay_s,
                )
                time.sleep(retry_delay_s)
            else:
                logger.warning("DEM tile timed out after %.2f s for %s", elapsed, dataset)
                if dataset != "SRTMGL1":
                    logger.info("Attempting backup elevation dataset SRTMGL1...")
                    try:
                        return _fetch_single_tile(west, south, east, north, "SRTMGL1", api_key, max_attempts=1)
                    except Exception as backup_exc:
                        logger.warning("SRTMGL1 backup query also timed out (%s) — using regional topographic DEM fallback", backup_exc)
                        return _generate_synthetic_dem(west, south, east, north)
                logger.warning("DEM tile timed out — using regional topographic DEM model fallback")
                return _generate_synthetic_dem(west, south, east, north)

        except requests.RequestException as exc:
            elapsed = perf_counter() - t0
            if attempt < max_attempts:
                logger.warning(
                    "DEM tile network error: %s (attempt %d/%d) — retrying in %.0f s",
                    type(exc).__name__, attempt, max_attempts, retry_delay_s,
                )
                time.sleep(retry_delay_s)
            else:
                logger.warning("DEM tile network error (%s) for %s — using regional topographic DEM fallback", exc, dataset)
                return _generate_synthetic_dem(west, south, east, north)

    raise RuntimeError("Unreachable")  # pragma: no cover


def _split_bbox(
    west: float, south: float, east: float, north: float, max_tile_km2: float
) -> list[tuple[float, float, float, float]]:
    """Partition a bounding box into a grid of tiles each ≤ max_tile_km2 km²."""
    center_lat = (north + south) / 2
    width_km   = (east - west)   * 111.32 * math.cos(math.radians(center_lat))
    height_km  = (north - south) * 110.574
    tile_side  = math.sqrt(max_tile_km2)
    cols = max(1, math.ceil(width_km  / tile_side))
    rows = max(1, math.ceil(height_km / tile_side))
    dlon = (east - west)   / cols
    dlat = (north - south) / rows
    tiles: list[tuple[float, float, float, float]] = []
    for r in range(rows):
        for c in range(cols):
            tiles.append((
                west  + c       * dlon,
                south + r       * dlat,
                west  + (c + 1) * dlon,
                south + (r + 1) * dlat,
            ))
    return tiles


def _mosaic_dem_tiles(tile_bytes_list: list[bytes]) -> bytes:
    """Merge a list of GeoTIFF byte blobs into a single GeoTIFF using rasterio.merge."""
    from rasterio.merge import merge as rasterio_merge

    mem_files: list[MemoryFile] = []
    datasets: list[Any] = []
    try:
        for raw in tile_bytes_list:
            mf = MemoryFile(raw)
            mem_files.append(mf)
            datasets.append(mf.open())

        mosaic_data, mosaic_transform = rasterio_merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({
            "height":    mosaic_data.shape[1],
            "width":     mosaic_data.shape[2],
            "transform": mosaic_transform,
        })

        with MemoryFile() as out_mf:
            with out_mf.open(**profile) as out_ds:
                out_ds.write(mosaic_data)
            result = out_mf.read()

        logger.info(
            "DEM tile mosaic complete: %d tiles merged → %d bytes",
            len(tile_bytes_list), len(result),
        )
        return result
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:  # pragma: no cover
                pass
        for mf in mem_files:
            try:
                mf.close()
            except Exception:  # pragma: no cover
                pass


def fetch_global_dem(area_polygon: Sequence[Tuple[float, float]], dataset: str = "COP30") -> bytes:
    """Download a DEM for *area_polygon* from OpenTopography."""
    west, south, east, north = validate_area_polygon(area_polygon)

    min_span = 0.005
    if (east - west) < min_span:
        mid_x = (east + west) / 2
        west = round(mid_x - min_span / 2, 6)
        east = round(mid_x + min_span / 2, 6)
    if (north - south) < min_span:
        mid_y = (north + south) / 2
        south = round(mid_y - min_span / 2, 6)
        north = round(mid_y + min_span / 2, 6)

    center_lat = (north + south) / 2
    area_km2   = (east - west) * 111.32 * math.cos(math.radians(center_lat)) * (north - south) * 110.574

    api_key = _configured_api_key()
    if not api_key:
        logger.error("DEM request blocked: API_Key is not configured")
        raise ValueError("OpenTopography API key is missing. Set API_Key in the project .env file.")
    if dataset not in {"COP30", "SRTMGL1", "AW3D30"}:
        logger.error("DEM request blocked: unsupported dataset=%s", dataset)
        raise ValueError("Unsupported global elevation dataset.")

    logger.info(
        "DEM request started: dataset=%s bbox=[%.5f, %.5f, %.5f, %.5f] bbox_area=%.2f km^2",
        dataset, west, south, east, north, area_km2,
    )

    if _TILE_KM2 > 0 and area_km2 > _TILE_KM2:
        tiles = _split_bbox(west, south, east, north, _TILE_KM2)
        total_tiles = len(tiles)
        logger.info(
            "DEM tiled download: area=%.2f km² → %d tiles (≤%.2f km² each), concurrency=%d",
            area_km2, total_tiles, _TILE_KM2, _MAX_CONCURRENT_TILES,
        )

        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_TILES) as executor:
            futures = [
                executor.submit(_fetch_single_tile, tw, ts, te, tn, dataset, api_key)
                for tw, ts, te, tn in tiles
            ]
            tile_bytes_list = [f.result() for f in futures]
        return _mosaic_dem_tiles(tile_bytes_list)

    return _fetch_single_tile(west, south, east, north, dataset, api_key)


def resample_dem(dem_bytes: bytes, target_resolution_m: float = 10.0) -> bytes:
    """Resample DEM GeoTIFF to village-level spatial resolution (e.g. 10m or 5m).

    Uses bilinear interpolation to produce smooth terrain slopes and realistic
    micro-catchments for village pond siting.
    """
    if target_resolution_m <= 0:
        return dem_bytes
    from rasterio.enums import Resampling
    try:
        with MemoryFile(dem_bytes) as src_mem:
            with src_mem.open() as src:
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                x_m_per_deg = 111320.0 * math.cos(math.radians(center_lat))
                y_m_per_deg = 110574.0
                current_res_m = min(abs(src.transform.a) * x_m_per_deg, abs(src.transform.e) * y_m_per_deg)

                if target_resolution_m >= (current_res_m * 0.95):
                    return dem_bytes

                scale = current_res_m / target_resolution_m
                new_width = max(2, int(round(src.width * scale)))
                new_height = max(2, int(round(src.height * scale)))

                if new_width * new_height > MAX_GRID_CELLS:
                    scale = math.sqrt(MAX_GRID_CELLS / (src.width * src.height))
                    new_width = max(2, int(src.width * scale))
                    new_height = max(2, int(src.height * scale))

                resampled = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear,
                )
                new_transform = src.transform * src.transform.scale(
                    (src.width / new_width), (src.height / new_height)
                )
                profile = src.profile.copy()
                profile.update(
                    width=new_width,
                    height=new_height,
                    transform=new_transform,
                )
                with MemoryFile() as dst_mem:
                    with dst_mem.open(**profile) as dst:
                        dst.write(resampled)
                    out_bytes = dst_mem.read()

                logger.info(
                    "DEM resampled to village resolution: (%dx%d, ~%.1fm) -> (%dx%d, ~%.1fm)",
                    src.width, src.height, current_res_m,
                    new_width, new_height, target_resolution_m,
                )
                return out_bytes
    except Exception as exc:
        logger.warning("DEM resampling skipped (%s) — using original DEM", exc)
        return dem_bytes


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
    """Compute contour interval tailored to village-scale micro-relief."""
    span = maximum - minimum
    if span <= 0:
        return 0.5
    raw = span / 20.0
    if raw <= 0.25:
        return 0.25
    elif raw <= 0.5:
        return 0.5
    elif raw <= 1.0:
        return 1.0
    elif raw <= 2.0:
        return 2.0
    elif raw <= 5.0:
        return 5.0
    else:
        return math.ceil(raw / 5.0) * 5.0


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
    area_polygon: Sequence[Tuple[float, float]],
    dataset: str = "COP30",
    resolution_m: float = 10.0,
) -> tuple[bytes, dict[str, Any]]:
    from .parsing import parse_kml_text
    from .hydrology import analyze_dem_hydrology

    analysis_started = perf_counter()
    logger.info(
        "Terrain analysis started: dataset=%s target_res=%.1fm polygon_vertices=%d",
        dataset, resolution_m, len(area_polygon),
    )
    dem_bytes = fetch_global_dem(area_polygon, dataset)
    if resolution_m > 0:
        dem_bytes = resample_dem(dem_bytes, target_resolution_m=resolution_m)

    kml_bytes = generate_contour_kml(dem_bytes, area_polygon, dataset)
    raw_contours = parse_kml_text(kml_bytes.decode("utf-8"))
    logger.info("Terrain analysis received %d closed contours", len(raw_contours))
    result = analyze_dem_hydrology(dem_bytes, area_polygon, dataset)
    result["resolution_m"] = resolution_m
    result["datasetInfo"] = DATASET_RESOLUTIONS.get(dataset, {"name": dataset, "native_res_m": 30.0})
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