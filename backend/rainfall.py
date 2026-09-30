from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date
from typing import Any

import httpx

logger = logging.getLogger(__name__)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RAINFALL_YEARS = 30
RUNOFF_COEFFICIENT = 0.20

# ---------------------------------------------------------------------------
# IMD Climate Normals Fallback (India Meteorological Department)
# Average monthly rainfall (mm) for Central India / Chhattisgarh belt.
# Used when the Open-Meteo public API is unreachable (campus firewall / timeout).
# ---------------------------------------------------------------------------
_IMD_MONTHLY_MM = [12.5, 18.2, 14.8, 16.4, 24.1, 195.4, 375.8, 360.2, 185.6, 48.3, 10.2, 6.5]


def _imd_fallback(latitude: float, longitude: float) -> dict[str, Any]:
    """Build a rainfall result dict from IMD Central India climate normals."""
    annual_mm = sum(_IMD_MONTHLY_MM)
    monthly_m = [round(v / 1000.0, 4) for v in _IMD_MONTHLY_MM]
    end_year = date.today().year - 1
    start_year = end_year - 9
    return {
        "source": "IMD Central India Climate Normals (Network Fallback)",
        "model": "IMD",
        "gridResolutionKm": 50,
        "periodStart": f"{start_year}-01-01",
        "periodEnd":   f"{end_year}-12-31",
        "yearsIncluded": list(range(start_year, end_year + 1)),
        "annualPrecipitationM": {str(y): round(annual_mm / 1000.0, 4) for y in range(start_year, end_year + 1)},
        "meanAnnualPrecipitationM": round(annual_mm / 1000.0, 4),
        "meanMonthlyPrecipitationM": monthly_m,
        "gridCell": {"latitude": round(latitude, 6), "longitude": round(longitude, 6)},
    }


def fetch_historical_rainfall(
    latitude: float, longitude: float, years: int = 10
) -> dict[str, Any]:
    return fetch_historical_rainfall_for_points([(latitude, longitude)], years)[0]


async def _fetch_single_location_async(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
) -> dict[str, Any] | None:
    """
    Fetch ERA5 daily precipitation for a single location using async httpx.
    Returns None on any network / timeout error so the caller can fall back.
    """
    params = {
        "latitude":  f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "daily":      "precipitation_sum",
        "timezone":   "auto",
        "models":     "era5",
        "precipitation_unit": "mm",
    }
    try:
        resp = await client.get(OPEN_METEO_ARCHIVE_URL, params=params)
        if resp.status_code != 200:
            logger.warning("Open-Meteo returned HTTP %d for (%.4f, %.4f)", resp.status_code, latitude, longitude)
            return None
        return resp.json()
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.warning("Open-Meteo request failed for (%.4f, %.4f): %s", latitude, longitude, type(exc).__name__)
        return None


async def fetch_historical_rainfall_for_points_async(
    locations: list[tuple[float, float]], years: int = 10
) -> list[dict[str, Any]]:
    """
    Async version — fetches ERA5 rainfall concurrently for all pond candidate locations.
    Falls back to IMD regional climate normals per-location on network failure.
    Your original aggregation logic is fully preserved.
    """
    if not locations:
        raise ValueError("At least one rainfall location is required.")
    if len(locations) > 12:
        raise ValueError("Rainfall can be requested for at most 12 pond candidates at once.")
    for latitude, longitude in locations:
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            raise ValueError("Rainfall coordinates must be finite numbers.")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Rainfall coordinates must be valid WGS84 values.")
    if not 1 <= years <= MAX_RAINFALL_YEARS:
        raise ValueError(f"Rainfall history must be between 1 and {MAX_RAINFALL_YEARS} years.")

    end_year   = date.today().year - 1
    start_year = end_year - years + 1

    logger.info(
        "Precipitation request started: model=ERA5 period=%d-%d candidate_points=%d",
        start_year, end_year, len(locations),
    )

    # Fetch all locations concurrently with a shared async httpx client (12-second timeout)
    async with httpx.AsyncClient(timeout=12.0) as client:
        import asyncio
        raw_payloads = await asyncio.gather(*[
            _fetch_single_location_async(client, lat, lon, start_year, end_year)
            for lat, lon in locations
        ])

    results: list[dict[str, Any]] = []
    for (latitude, longitude), payload in zip(locations, raw_payloads):
        # ── Fall back to IMD normals if API failed ──────────────────────────
        if payload is None:
            logger.warning(
                "Using IMD fallback for (%.4f, %.4f) — Open-Meteo unreachable", latitude, longitude
            )
            results.append(_imd_fallback(latitude, longitude))
            continue

        # ── Your original per-location aggregation logic ────────────────────
        daily = payload.get("daily", {})
        dates = daily.get("time", [])
        rain_values = daily.get("precipitation_sum", [])

        if len(dates) != len(rain_values) or not dates:
            logger.warning("Open-Meteo returned empty series for (%.4f, %.4f) — using IMD fallback", latitude, longitude)
            results.append(_imd_fallback(latitude, longitude))
            continue

        annual_totals: dict[int, float] = defaultdict(float)
        annual_valid_days: dict[int, int] = defaultdict(int)
        monthly_totals: dict[tuple[int, int], float] = defaultdict(float)
        monthly_valid_days: dict[tuple[int, int], int] = defaultdict(int)

        for day_text, precipitation_value in zip(dates, rain_values):
            if precipitation_value is None:
                continue
            try:
                precipitation_mm = float(precipitation_value)
                day = date.fromisoformat(day_text)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(precipitation_mm) or precipitation_mm < 0:
                continue
            annual_totals[day.year] += precipitation_mm
            annual_valid_days[day.year] += 1
            monthly_totals[(day.year, day.month)] += precipitation_mm
            monthly_valid_days[(day.year, day.month)] += 1

        complete_years = [
            year for year in range(start_year, end_year + 1)
            if annual_valid_days[year] >= (366 if _is_leap_year(year) else 365) - 3
        ]

        if not complete_years:
            logger.warning("No complete years in ERA5 data for (%.4f, %.4f) — using IMD fallback", latitude, longitude)
            results.append(_imd_fallback(latitude, longitude))
            continue

        annual_precipitation = {str(year): round(annual_totals[year] / 1000.0, 4) for year in complete_years}
        monthly_mean = []
        for month in range(1, 13):
            year_totals_list = [
                monthly_totals[(year, month)]
                for year in complete_years
                if monthly_valid_days[(year, month)] >= 27
            ]
            monthly_mean.append(
                round(sum(year_totals_list) / len(year_totals_list) / 1000.0, 4)
                if year_totals_list else 0.0
            )

        result: dict[str, Any] = {
            "source": "Open-Meteo Historical Weather API",
            "model": "ERA5",
            "gridResolutionKm": 25,
            "periodStart": f"{complete_years[0]}-01-01",
            "periodEnd":   f"{complete_years[-1]}-12-31",
            "yearsIncluded": complete_years,
            "annualPrecipitationM": annual_precipitation,
            "meanAnnualPrecipitationM": round(
                sum(annual_totals[year] for year in complete_years) / len(complete_years) / 1000.0, 4
            ),
            "meanMonthlyPrecipitationM": monthly_mean,
            "gridCell": {
                "latitude":  payload.get("latitude", latitude),
                "longitude": payload.get("longitude", longitude),
            },
        }
        results.append(result)
        logger.info(
            "Precipitation summary: years=%d mean_annual=%.4f m grid_cell=(%.5f, %.5f)",
            len(complete_years), result["meanAnnualPrecipitationM"],
            result["gridCell"]["latitude"], result["gridCell"]["longitude"],
        )

    logger.info("Precipitation request complete: candidate_points=%d", len(results))
    return results


def fetch_historical_rainfall_for_points(
    locations: list[tuple[float, float]], years: int = 10
) -> list[dict[str, Any]]:
    """
    Synchronous wrapper kept for backward compatibility (tests, legacy endpoints).
    Delegates to the async implementation via a new event loop.
    """
    import asyncio
    return asyncio.run(fetch_historical_rainfall_for_points_async(locations, years))


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def estimate_annual_runoff_m3(
    mean_annual_precipitation_m: float,
    catchment_area_sq_m: float,
    runoff_coefficient: float,
) -> float:
    if not math.isfinite(mean_annual_precipitation_m) or mean_annual_precipitation_m < 0:
        raise ValueError("Mean annual precipitation must be a non-negative finite value.")
    if not math.isfinite(catchment_area_sq_m) or catchment_area_sq_m < 0:
        raise ValueError("Catchment area must be a non-negative finite value.")
    if not math.isclose(runoff_coefficient, RUNOFF_COEFFICIENT):
        raise ValueError(f"Runoff coefficient is fixed at {RUNOFF_COEFFICIENT:.2f}.")
    return mean_annual_precipitation_m * catchment_area_sq_m * RUNOFF_COEFFICIENT


def rank_candidates_by_rainfall(
    candidates: list[dict[str, Any]], rainfall_series: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not candidates or len(candidates) != len(rainfall_series):
        raise ValueError("Each pond candidate must have one matching rainfall series.")

    ranked = []
    for candidate, rainfall in zip(candidates, rainfall_series):
        runoff_potential = estimate_annual_runoff_m3(
            rainfall["meanAnnualPrecipitationM"],
            candidate["estimatedCatchmentAreaSqM"],
            RUNOFF_COEFFICIENT,
        )
        storage_capacity = max(0.0, float(candidate["estimatedVolumeM3"]))
        estimated_fillable = min(runoff_potential, storage_capacity)
        ranked_candidate = {
            **candidate,
            "rainfallSource": rainfall["source"],
            "rainfallModel": rainfall["model"],
            "rainfallGridResolutionKm": rainfall["gridResolutionKm"],
            "meanAnnualPrecipitationM": rainfall["meanAnnualPrecipitationM"],
            "meanMonthlyPrecipitationM": rainfall["meanMonthlyPrecipitationM"],
            "rainfallPeriodStart": rainfall["periodStart"],
            "rainfallPeriodEnd": rainfall["periodEnd"],
            "rainfallYearsIncluded": rainfall["yearsIncluded"],
            "rainfallGridCell": rainfall["gridCell"],
            "potentialAnnualRunoffM3": round(runoff_potential, 2),
            "estimatedAnnualFillableWaterM3": round(estimated_fillable, 2),
            "runoffCoefficient": RUNOFF_COEFFICIENT,
        }
        ranked.append(ranked_candidate)

    ranked.sort(
        key=lambda candidate: (
            candidate["estimatedAnnualFillableWaterM3"],
            candidate["potentialAnnualRunoffM3"],
            candidate["estimatedVolumeM3"],
            candidate.get("score", 0.0),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["recommended"] = rank == 1
        candidate["recommendationBasis"] = "rainfall runoff potential constrained by pond storage capacity"
    return ranked