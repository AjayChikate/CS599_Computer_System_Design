from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.parsing import load_raw_contours
from backend.service import analyze_contours_in_area

client = TestClient(app)


def test_analyze_contour_accepts_kml() -> None:
    sample_file = Path(__file__).resolve().parents[1] / "contours_1m.kml"
    with sample_file.open("rb") as handle:
        response = client.post(
            "/analyzeContour",
            files={"file": (sample_file.name, handle.read(), "application/vnd.google-earth.kml+xml")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["estimatedCatchmentAreaSqM"] > 0
    assert payload["pondElevation"] > 0


def test_analyze_contour_rejects_invalid_type() -> None:
    response = client.post(
        "/analyzeContour",
        files={"file": ("sample.txt", b"not a contour file", "text/plain")},
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_selected_area_analysis_includes_complete_contours() -> None:
    sample_file = Path(__file__).resolve().parents[1] / "contours_1m.kml"
    raw_contours = load_raw_contours(sample_file.read_bytes(), sample_file.name)
    points = [point for contour in raw_contours for point in contour["points"]]
    min_lon = min(point[0] for point in points)
    max_lon = max(point[0] for point in points)
    min_lat = min(point[1] for point in points)
    max_lat = max(point[1] for point in points)
    selected_area = [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ]

    result = analyze_contours_in_area(raw_contours, selected_area)

    assert result["status"] == "ok"
    assert result["estimatedCatchmentAreaSqM"] > 0
    assert result["estimatedVolumeM3"] > 0
    assert result["terrainSummary"]["contourCount"] == len(raw_contours)


def test_selected_area_rejects_area_without_complete_contours() -> None:
    sample_file = Path(__file__).resolve().parents[1] / "contours_1m.kml"
    raw_contours = load_raw_contours(sample_file.read_bytes(), sample_file.name)
    longitude, latitude = raw_contours[0]["points"][0]
    offset = 0.00000001
    selected_area = [
        (longitude - offset, latitude - offset),
        (longitude + offset, latitude - offset),
        (longitude + offset, latitude + offset),
        (longitude - offset, latitude + offset),
        (longitude - offset, latitude - offset),
    ]

    with pytest.raises(ValueError, match="No complete contour rings"):
        analyze_contours_in_area(raw_contours, selected_area)
