from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RAINFALL_YEARS = 30
RUNOFF_COEFFICIENT = 0.20


def fetch_historical_rainfall(
    latitude: float, longitude: float, years: int = 10
) -> dict[str, Any]:
    return fetch_historical_rainfall_for_points([(latitude, longitude)], years)[0]


def fetch_historical_rainfall_for_points(
    locations: list[tuple[float, float]], years: int = 10
) -> list[dict[str, Any]]:
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

    end_year = date.today().year - 1
    start_year = end_year - years + 1
    params = {
        "latitude": ",".join(f"{latitude:.6f}" for latitude, _ in locations),
        "longitude": ",".join(f"{longitude:.6f}" for _, longitude in locations),
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": "precipitation_sum",
        "timezone": "auto",
        "models": "era5",
        "precipitation_unit": "mm",
    }
    logger.info(
        "Precipitation request started: model=ERA5 period=%d-%d candidate_points=%d",
        start_year,
        end_year,
        len(locations),
    )
    max_attempts = 3
    retry_delay_s = 2.0
    payload = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=(30, 90))
            if response.status_code != 200:
                logger.warning("Rainfall provider returned HTTP %d", response.status_code)
                if response.status_code == 429:
                    raise ValueError("Rainfall API rate limit reached. Please try again in a few moments.")
                raise ValueError(f"Rainfall provider returned HTTP {response.status_code}.")
            payload = response.json()
            break
        except requests.Timeout:
            if attempt < max_attempts:
                logger.warning(
                    "Rainfall request timed out (attempt %d/%d) — retrying in %.0f s",
                    attempt, max_attempts, retry_delay_s,
                )
                time.sleep(retry_delay_s)
            else:
                logger.warning("Rainfall request timed out (all %d attempts exhausted)", max_attempts)
                raise ValueError("The rainfall service took too long to respond. Try again shortly.") from None
        except requests.RequestException as exc:
            if attempt < max_attempts:
                logger.warning(
                    "Rainfall network request failed: %s (attempt %d/%d) — retrying in %.0f s",
                    type(exc).__name__, attempt, max_attempts, retry_delay_s,
                )
                time.sleep(retry_delay_s)
            else:
                logger.error("Rainfall network request failed: %s", type(exc).__name__)
                raise ValueError("Could not connect to the rainfall service. Check the network and try again.") from exc
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Rainfall response could not be decoded: %s", type(exc).__name__)
            raise ValueError("The rainfall service returned an unreadable response.") from exc

    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(locations):
        raise ValueError("The rainfall service returned a different number of locations than requested.")

    results = []
    for location, location_payload in zip(locations, payloads):
        latitude, longitude = location
        daily = location_payload.get("daily", {})
        dates = daily.get("time", [])
        rain_values = daily.get("precipitation_sum", [])
        if len(dates) != len(rain_values) or not dates:
            raise ValueError("The rainfall service returned no daily precipitation data for a candidate.")

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
            year
            for year in range(start_year, end_year + 1)
            if annual_valid_days[year] >= (366 if _is_leap_year(year) else 365) - 3
        ]
        if not complete_years:
            raise ValueError("The rainfall service did not return any complete calendar years for a candidate.")

        annual_precipitation = {str(year): round(annual_totals[year] / 1000.0, 4) for year in complete_years}
        monthly_mean = []
        for month in range(1, 13):
            year_totals = [
                monthly_totals[(year, month)]
                for year in complete_years
                if monthly_valid_days[(year, month)] >= 27
            ]
            monthly_mean.append(round(sum(year_totals) / len(year_totals) / 1000.0, 4) if year_totals else 0.0)
        result = {
            "source": "Open-Meteo Historical Weather API",
            "model": "ERA5",
            "gridResolutionKm": 25,
            "periodStart": f"{complete_years[0]}-01-01",
            "periodEnd": f"{complete_years[-1]}-12-31",
            "yearsIncluded": complete_years,
            "annualPrecipitationM": annual_precipitation,
            "meanAnnualPrecipitationM": round(
                sum(annual_totals[year] for year in complete_years) / len(complete_years) / 1000.0,
                4,
            ),
            "meanMonthlyPrecipitationM": monthly_mean,
            "gridCell": {
                "latitude": location_payload.get("latitude", latitude),
                "longitude": location_payload.get("longitude", longitude),
            },
        }
        results.append(result)
        logger.info(
            "Precipitation summary: years=%d mean_annual=%.4f m grid_cell=(%.5f, %.5f)",
            len(complete_years),
            result["meanAnnualPrecipitationM"],
            result["gridCell"]["latitude"],
            result["gridCell"]["longitude"],
        )

    logger.info(
        "Precipitation request complete: candidate_points=%d",
        len(results),
    )
    return results


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