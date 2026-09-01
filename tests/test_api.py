from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app

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
