from __future__ import annotations

from datetime import date, timedelta

import pytest
import requests

from backend import rainfall


def test_historical_rainfall_aggregates_complete_years(monkeypatch: pytest.MonkeyPatch) -> None:
    end_year = date.today().year - 1
    start_year = end_year - 9
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    dates = []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "latitude": 21.25,
                "longitude": 81.29,
                "daily": {"time": dates, "precipitation_sum": [1.0] * len(dates)},
            }

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(rainfall.requests, "get", fake_get)

    result = rainfall.fetch_historical_rainfall(21.251, 81.291)

    assert result["model"] == "ERA5"
    assert len(result["yearsIncluded"]) == 10
    assert result["meanAnnualPrecipitationM"] == pytest.approx(0.3653, abs=0.0002)
    assert result["meanMonthlyPrecipitationM"][0] == pytest.approx(0.031)
    assert result["meanMonthlyPrecipitationM"][1] == pytest.approx(0.0283, abs=0.0002)
    assert result["meanMonthlyPrecipitationM"][2] == pytest.approx(0.031)
    assert result["gridCell"] == {"latitude": 21.25, "longitude": 81.29}
    assert captured["params"]["daily"] == "precipitation_sum"
    assert captured["params"]["models"] == "era5"
    assert captured["params"]["timezone"] == "auto"


def test_annual_runoff_uses_precipitation_area_and_runoff_coefficient() -> None:
    assert rainfall.estimate_annual_runoff_m3(1.0, 10_000.0, rainfall.RUNOFF_COEFFICIENT) == pytest.approx(2_000.0)


def test_annual_runoff_rejects_changed_runoff_coefficient() -> None:
    with pytest.raises(ValueError, match="fixed at 0.20"):
        rainfall.estimate_annual_runoff_m3(1000.0, 10_000.0, 1.2)


def test_multi_location_rainfall_request_returns_one_series_per_site(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return [
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": {
                        "time": dates,
                        "precipitation_sum": [1.0] * len(dates),
                    },
                }
                for latitude, longitude in [(21.25, 81.25), (21.30, 81.30)]
            ]

    end_year = date.today().year - 1
    start_year = end_year - 1
    dates = []
    current = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params)
        return FakeResponse()

    monkeypatch.setattr(rainfall.requests, "get", fake_get)

    results = rainfall.fetch_historical_rainfall_for_points([(21.25, 81.25), (21.30, 81.30)], years=2)

    assert len(results) == 2
    assert captured["params"]["latitude"] == "21.250000,21.300000"
    assert captured["params"]["longitude"] == "81.250000,81.300000"


def test_rainfall_recommendation_combines_storage_and_runoff() -> None:
    candidates = [
        {"rank": 1, "recommended": True, "score": 0.8, "estimatedCatchmentAreaSqM": 100_000.0, "estimatedVolumeM3": 20.0},
        {"rank": 2, "recommended": False, "score": 0.5, "estimatedCatchmentAreaSqM": 10_000.0, "estimatedVolumeM3": 100.0},
    ]
    rainfall_series = [
        {"source": "Open-Meteo", "model": "ERA5", "gridResolutionKm": 25, "meanAnnualPrecipitationM": 1.0, "meanMonthlyPrecipitationM": [0.1] * 12, "periodStart": "2016-01-01", "periodEnd": "2025-12-31", "yearsIncluded": list(range(2016, 2026)), "gridCell": {"latitude": 21.25, "longitude": 81.25}},
        {"source": "Open-Meteo", "model": "ERA5", "gridResolutionKm": 25, "meanAnnualPrecipitationM": 1.0, "meanMonthlyPrecipitationM": [0.1] * 12, "periodStart": "2016-01-01", "periodEnd": "2025-12-31", "yearsIncluded": list(range(2016, 2026)), "gridCell": {"latitude": 21.30, "longitude": 81.30}},
    ]

    ranked = rainfall.rank_candidates_by_rainfall(candidates, rainfall_series)

    assert ranked[0]["estimatedAnnualFillableWaterM3"] == 100.0
    assert ranked[0]["potentialAnnualRunoffM3"] == 2_000.0
    assert ranked[0]["recommended"] is True
    assert ranked[1]["estimatedAnnualFillableWaterM3"] == 20.0
    assert ranked[1]["recommended"] is False


def test_historical_rainfall_rejects_invalid_coordinates() -> None:
    with pytest.raises(ValueError, match="valid WGS84"):
        rainfall.fetch_historical_rainfall(95.0, 81.0)


def test_historical_rainfall_handles_provider_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class RateLimitedResponse:
        status_code = 429

    monkeypatch.setattr(rainfall.requests, "get", lambda *args, **kwargs: RateLimitedResponse())

    with pytest.raises(ValueError, match="HTTP 429"):
        rainfall.fetch_historical_rainfall(21.25, 81.29)
