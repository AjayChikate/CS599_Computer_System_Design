from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import backend.opentopography as opentopography
from backend.hydrology import analyze_dem_hydrology
from backend.geometry import normalize_lonlat_ring
from backend.parsing import parse_kml_text
from backend.service import analyze_contours_in_area


def synthetic_dem() -> tuple[bytes, list[tuple[float, float]]]:
    width = height = 100
    resolution = 0.00027
    west = 81.30
    north = 21.28
    rows, columns = np.mgrid[0:height, 0:width]
    elevations = ((columns - 49.5) ** 2 + (rows - 49.5) ** 2).astype("float32")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(west, north, resolution, resolution),
    }
    with rasterio.io.MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(elevations, 1)
        dem_bytes = memory_file.read()

    east = west + width * resolution
    south = north - height * resolution
    area_polygon = [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
        (west, south),
    ]
    return dem_bytes, area_polygon


def monotonic_dem(flat: bool = False) -> tuple[bytes, list[tuple[float, float]]]:
    width = height = 100
    resolution = 0.00027
    west = 81.30
    north = 21.28
    rows, columns = np.mgrid[0:height, 0:width]
    elevations = np.zeros((height, width), dtype="float32") if flat else (rows + columns).astype("float32")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(west, north, resolution, resolution),
    }
    with rasterio.io.MemoryFile() as memory_file:
        with memory_file.open(**profile) as dataset:
            dataset.write(elevations, 1)
        dem_bytes = memory_file.read()

    east = west + width * resolution
    south = north - height * resolution
    area_polygon = [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
        (west, south),
    ]
    return dem_bytes, area_polygon


def test_dem_generates_closed_kml_and_existing_analysis_result() -> None:
    dem_bytes, area_polygon = synthetic_dem()
    kml_bytes = opentopography.generate_contour_kml(dem_bytes, area_polygon)
    raw_contours = parse_kml_text(kml_bytes.decode("utf-8"))
    result = analyze_contours_in_area(raw_contours, area_polygon)

    assert raw_contours
    assert all(contour["points"][0] == contour["points"][-1] for contour in raw_contours)
    assert result["status"] == "ok"
    assert result["estimatedCatchmentAreaSqM"] > 0
    assert result["estimatedVolumeM3"] > 0


def test_dem_hydrology_finds_depression_storage_and_catchment() -> None:
    dem_bytes, area_polygon = synthetic_dem()

    result = analyze_dem_hydrology(dem_bytes, area_polygon, "synthetic bowl")

    assert result["fallbackRecommendation"] is False
    assert result["terrainSummary"]["flowRouting"] == "D8"
    assert result["pondCandidates"]
    assert result["estimatedVolumeM3"] > 0
    assert result["estimatedCatchmentAreaSqM"] > 0
    assert result["pondCandidates"][0]["catchmentBoundary"]["type"] in {"Polygon", "MultiPolygon"}


def test_dem_area_workflow_fetches_once_and_returns_kml(monkeypatch: pytest.MonkeyPatch) -> None:
    dem_bytes, area_polygon = synthetic_dem()
    calls = []

    def fake_fetch(polygon, dataset):
        calls.append((polygon, dataset))
        return dem_bytes

    monkeypatch.setattr(opentopography, "fetch_global_dem", fake_fetch)
    kml_bytes, result = opentopography.analyze_dem_area(area_polygon, "COP30")

    assert calls == [(area_polygon, "COP30")]
    assert kml_bytes.startswith(b"<?xml")
    assert result["status"] == "ok"


def test_global_dem_selection_enforces_area_limit() -> None:
    oversized_polygon = [
        (81.0, 21.0),
        (82.0, 21.0),
        (82.0, 22.0),
        (81.0, 22.0),
        (81.0, 21.0),
    ]

    with pytest.raises(ValueError, match="maximum is 25 km²"):
        opentopography.validate_area_polygon(oversized_polygon)


def test_polygon_normalization_ignores_map_float_noise_and_closes_ring() -> None:
    first = [(81.123456781, 21.123456781), (81.2, 21.1), (81.123456781, 21.123456781)]
    repeated = [(81.123456784, 21.123456784), (81.2, 21.1), (81.123456784, 21.123456784)]

    assert normalize_lonlat_ring(first) == normalize_lonlat_ring(repeated)
    assert normalize_lonlat_ring(first)[0] == normalize_lonlat_ring(first)[-1]


def test_global_dem_request_uses_documented_parameters_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, area_polygon = synthetic_dem()
    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, traceback):
            return False

        def iter_content(self, chunk_size):
            return iter([b"tiff"])

    def fake_get(url, params, stream, timeout):
        captured.update(url=url, params=params, stream=stream, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(opentopography.requests, "get", fake_get)
    monkeypatch.setenv("API_Key", "test-configured-key")

    result = opentopography.fetch_global_dem(area_polygon, "COP30")

    assert result == b"tiff"
    assert captured["url"] == "https://portal.opentopography.org/API/globaldem"
    assert captured["params"]["demtype"] == "COP30"
    assert captured["params"]["outputFormat"] == "GTiff"
    assert captured["params"]["API_Key"] == "test-configured-key"
    assert captured["timeout"] == (10, 120)
    assert "DEM request started" in caplog.text
    assert "test-configured-key" not in caplog.text


def test_dem_workflow_returns_lowest_point_when_no_basin_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    dem_bytes, area_polygon = monotonic_dem()
    monkeypatch.setattr(opentopography, "fetch_global_dem", lambda polygon, dataset: dem_bytes)

    kml_bytes, result = opentopography.analyze_dem_area(area_polygon, "COP30")

    assert kml_bytes.startswith(b"<?xml")
    assert result["status"] == "ok"
    assert result["fallbackRecommendation"] is True
    assert len(result["pondCandidates"]) == 1
    assert result["pondCandidates"][0]["fallbackRecommendation"] is True
    assert result["estimatedCatchmentAreaSqM"] > 0
    assert result["estimatedVolumeM3"] == 0


def test_flat_dem_still_returns_a_fallback_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    dem_bytes, area_polygon = monotonic_dem(flat=True)
    monkeypatch.setattr(opentopography, "fetch_global_dem", lambda polygon, dataset: dem_bytes)

    _, result = opentopography.analyze_dem_area(area_polygon, "COP30")

    assert result["fallbackRecommendation"] is True
    assert result["pondCandidates"][0]["pondElevation"] == 0
    assert result["estimatedVolumeM3"] == 0


def test_dem_hydrology_uses_lowest_cell_fallback_only_without_depression() -> None:
    dem_bytes, area_polygon = monotonic_dem()

    result = analyze_dem_hydrology(dem_bytes, area_polygon, "synthetic slope")

    assert result["fallbackRecommendation"] is True
    assert result["pondCandidates"][0]["fallbackRecommendation"] is True
    assert result["estimatedVolumeM3"] == 0
    assert result["estimatedCatchmentAreaSqM"] > 0


    boundary = result["pondCandidates"][0]["catchmentBoundary"]
    first_ring = boundary["coordinates"][0][0] if boundary["type"] == "MultiPolygon" else boundary["coordinates"][0]
    assert all(-180 <= longitude <= 180 and -90 <= latitude <= 90 for longitude, latitude in first_ring)