from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class _StreamlitNoiseFilter(logging.Filter):
    """Suppress Streamlit health-check polling noise from uvicorn access logs.

    When a browser has an old Streamlit tab open it repeatedly polls
    /_stcore/health and /_stcore/host-config.  These are harmless 404s but
    they flood the log.  This filter drops them silently.
    """
    _PREFIXES = ("/_stcore/health", "/_stcore/host-config", "/static/media/Source")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        return not any(p in msg for p in self._PREFIXES)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("backend").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").addFilter(_StreamlitNoiseFilter())
logger = logging.getLogger(__name__)

try:
    from .config import MAX_UPLOAD_BYTES
    from .service import analyze_contour_map, analyze_contours_in_area, load_raw_contours
    from .opentopography import analyze_dem_area
    from .rainfall import fetch_historical_rainfall_for_points
except ImportError:
    from config import MAX_UPLOAD_BYTES
    from service import analyze_contour_map, analyze_contours_in_area, load_raw_contours
    from opentopography import analyze_dem_area
    from rainfall import fetch_historical_rainfall_for_points

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Pond Catchment Analysis API",
    description="Geospatial terrain hydrology and rainfall analysis for village pond siting.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic request models for JSON body endpoints
# ---------------------------------------------------------------------------

class DemAnalysisRequest(BaseModel):
    """Request body for /api/analyzeDemArea — download DEM and run full pipeline."""
    area_polygon: List[List[float]]  # [[lon, lat], ...] closed ring, ≤ 0.5 km²
    dataset: str = "COP30"           # AW3D30 | COP30 | SRTMGL1


class RainfallRequest(BaseModel):
    """Request body for /api/fetchRainfall — fetch ERA5 monthly totals."""
    locations: List[List[float]]  # [[lat, lon], ...] one per pond candidate
    years: int = 10               # Number of historical years to average


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


async def _read_upload(contour_map: UploadFile | None, file: UploadFile | None) -> tuple[bytes, str]:
    """Read and validate an uploaded file, returning (bytes, filename)."""
    upload = contour_map or file
    if upload is None or not upload.filename:
        raise ValueError("A file upload is required.")
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        logger.warning("Upload rejected: %d bytes exceeds %d limit", len(contents), MAX_UPLOAD_BYTES)
        raise ValueError("Upload is too large. The maximum supported file size is 25 MB.")
    logger.info("File upload received: filename=%s bytes=%d", upload.filename, len(contents))
    return contents, upload.filename


# ---------------------------------------------------------------------------
# Root — serve HTML frontend
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False, response_model=None)
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        status_code=200,
        content={
            "service": "Pond Catchment Analysis API",
            "docs": "/docs",
            "routes": ["POST /analyzeContour", "POST /api/analyzeDemArea", "POST /api/fetchRainfall"],
        },
    )


# ---------------------------------------------------------------------------
# Legacy KML / KMZ file upload endpoints (kept for backward compatibility)
# ---------------------------------------------------------------------------

async def _run_analysis(contour_map: UploadFile | None, file: UploadFile | None = None) -> dict[str, Any]:
    contents, filename = await _read_upload(contour_map, file)
    return analyze_contour_map(contents, filename)


@app.post("/analyzeContour")
@app.post("/findCatchment")
async def analyze_contour(
    contour_map: UploadFile | None = File(None), file: UploadFile | None = File(None)
) -> JSONResponse:
    try:
        analysis = await _run_analysis(contour_map, file)
        return JSONResponse(status_code=200, content=analysis)
    except ValueError as exc:
        return _error_response(400, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Contour API analysis failed")
        return _error_response(500, f"Unexpected server error: {exc}")


@app.post("/analyzeContour/summary")
async def analyze_contour_summary(
    contour_map: UploadFile | None = File(None), file: UploadFile | None = File(None)
) -> JSONResponse:
    try:
        analysis = await _run_analysis(contour_map, file)
        summary = {
            "status": analysis["status"],
            "pondElevation": analysis["pondElevation"],
            "pondCentroid": analysis["pondCentroid"],
            "estimatedCatchmentAreaSqM": analysis["estimatedCatchmentAreaSqM"],
            "basinDepthM": analysis["basinDepthM"],
            "compactnessScore": analysis["compactnessScore"],
            "confidenceScore": analysis["confidenceScore"],
            "terrainSummary": analysis["terrainSummary"],
            "riverAvoidance": analysis["riverAvoidance"],
        }
        return JSONResponse(status_code=200, content=summary)
    except ValueError as exc:
        return _error_response(400, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Contour API summary failed")
        return _error_response(500, f"Unexpected server error: {exc}")


@app.post("/analyzeContour/candidates")
async def analyze_contour_candidates(
    contour_map: UploadFile | None = File(None), file: UploadFile | None = File(None)
) -> JSONResponse:
    try:
        analysis = await _run_analysis(contour_map, file)
        payload = {
            "status": analysis["status"],
            "recommended": analysis["pondCandidates"][0],
            "alternativeCandidates": analysis["alternativeCandidates"],
            "candidateCount": len(analysis["pondCandidates"]),
        }
        return JSONResponse(status_code=200, content=payload)
    except ValueError as exc:
        return _error_response(400, str(exc))
    except Exception as exc:
        logger.exception("Contour API candidates failed")
        return _error_response(500, f"Unexpected server error: {exc}")


@app.post("/analyzeContour/raw")
async def analyze_contour_raw(
    contour_map: UploadFile | None = File(None), file: UploadFile | None = File(None)
) -> JSONResponse:
    try:
        contents, filename = await _read_upload(contour_map, file)
        contours = load_raw_contours(contents, filename)
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "contourCount": len(contours), "contours": contours},
        )
    except ValueError as exc:
        return _error_response(400, str(exc))
    except Exception as exc:
        logger.exception("Raw contour API failed")
        return _error_response(500, f"Unexpected server error: {exc}")


# ---------------------------------------------------------------------------
# NEW JSON API endpoints — consumed by the HTML/JS frontend
# ---------------------------------------------------------------------------

@app.post("/api/analyzeDemArea")
async def api_analyze_dem_area(request: DemAnalysisRequest) -> JSONResponse:
    """
    Download DEM for the drawn polygon, generate contours, run D8 hydrology,
    and return pond candidate results plus a base64-encoded KML blob.

    Body: { "area_polygon": [[lon, lat], ...], "dataset": "COP30" }
    Response: { "result": {...}, "kml_b64": "<base64 KML string>" }
    """
    try:
        polygon = [tuple(pt) for pt in request.area_polygon]
        if len(polygon) < 4:
            raise ValueError("area_polygon must have at least 4 vertices forming a closed ring.")
        logger.info(
            "DEM area API request: dataset=%s polygon_vertices=%d",
            request.dataset, len(polygon),
        )
        kml_bytes, result = analyze_dem_area(polygon, request.dataset)
        logger.info(
            "DEM area API completed: status=%s candidates=%d kml_bytes=%d",
            result.get("status"), len(result.get("pondCandidates", [])), len(kml_bytes),
        )
        return JSONResponse(
            status_code=200,
            content={
                "result": result,
                "kml_b64": base64.b64encode(kml_bytes).decode("utf-8"),
            },
        )
    except ValueError as exc:
        logger.warning("DEM area API rejected: %s", str(exc))
        return _error_response(400, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("DEM area API failed unexpectedly")
        return _error_response(500, f"Server error: {exc}")


@app.post("/api/fetchRainfall")
async def api_fetch_rainfall(request: RainfallRequest) -> JSONResponse:
    """
    Fetch ERA5-Land historical monthly precipitation for pond candidate locations.

    Body: { "locations": [[lat, lon], ...], "years": 10 }
    Response: list of rainfall series objects (one per location)
    """
    try:
        locations = [tuple(pt) for pt in request.locations]
        if not locations:
            raise ValueError("At least one location is required.")
        logger.info(
            "Rainfall API request: points=%d years=%d", len(locations), request.years,
        )
        series = fetch_historical_rainfall_for_points(locations, request.years)
        logger.info("Rainfall API completed: returned %d series", len(series))
        return JSONResponse(status_code=200, content=series)
    except ValueError as exc:
        logger.warning("Rainfall API rejected: %s", str(exc))
        return _error_response(400, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Rainfall API failed unexpectedly")
        return _error_response(500, f"Server error: {exc}")


@app.post("/api/analyzeContourWithArea")
async def api_analyze_contour_with_area(
    file: UploadFile = File(...),
    area_polygon_json: str = Form(...),
) -> JSONResponse:
    """
    Parse a KML/KMZ upload and run analysis restricted to the drawn polygon.

    Multipart form: file=<KML/KMZ binary>, area_polygon_json='[[lon,lat],...]'
    Response: full analysis dict (same schema as /analyzeContour)
    """
    try:
        contents, filename = await _read_upload(file, None)
        try:
            raw_area = json.loads(area_polygon_json)
            area = [tuple(pt) for pt in raw_area]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"area_polygon_json is not valid JSON: {exc}") from exc

        raw_contours = load_raw_contours(contents, filename)
        logger.info(
            "Contour+area API request: filename=%s bytes=%d polygon_vertices=%d loaded_contours=%d",
            filename, len(contents), len(area), len(raw_contours),
        )
        result = analyze_contours_in_area(raw_contours, area)
        logger.info(
            "Contour+area API completed: status=%s candidates=%d",
            result.get("status"), len(result.get("pondCandidates", [])),
        )
        return JSONResponse(status_code=200, content=result)
    except ValueError as exc:
        logger.warning("Contour+area API rejected: %s", str(exc))
        return _error_response(400, str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Contour+area API failed unexpectedly")
        return _error_response(500, f"Server error: {exc}")


# ---------------------------------------------------------------------------
# Static file mount — serves frontend CSS / JS (must come AFTER all routes)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    logger.info("Frontend static files mounted from: %s", FRONTEND_DIR)
else:
    logger.warning("Frontend directory not found at %s — static serving disabled", FRONTEND_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)