from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .service import analyze_contour_map, load_raw_contours
except ImportError: 
    from service import analyze_contour_map, load_raw_contours

app = FastAPI(
    title="Pond Catchment Analysis API",
    description="Upload a KML or KMZ contour map and estimate suitable pond basins and watershed areas.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


async def _run_analysis(file: UploadFile) -> dict[str, Any]:
    if not file.filename:
        raise ValueError("A file upload is required.")
    contents = await file.read()
    return analyze_contour_map(contents, file.filename)


@app.get("/")
def read_root() -> dict[str, Any]:
    return {
        "service": "Pond Catchment Analysis API",
        "routes": [
            "POST /analyzeContour",
            "POST /findCatchment",
            "POST /analyzeContour/summary",
            "POST /analyzeContour/candidates",
            "POST /analyzeContour/raw",
        ],
        "description": "Upload a KML or KMZ contour map to estimate pond basin and catchment area.",
    }


@app.post("/analyzeContour")
@app.post("/findCatchment")
async def analyze_contour(contour_map: UploadFile = File(...)) -> JSONResponse:
    try:
        analysis = await _run_analysis(contour_map)
        return JSONResponse(status_code=200, content=analysis)
    except ValueError as exc:
        return _error_response(400, str(exc))
    except Exception as exc:  # pragma: no cover
        return _error_response(500, f"Unexpected server error: {exc}")


@app.post("/analyzeContour/summary")
async def analyze_contour_summary(contour_map: UploadFile = File(...)) -> JSONResponse:
    try:
        analysis = await _run_analysis(contour_map)
        summary = {
            "status": analysis["status"],
            "pondElevation": analysis["pondElevation"],
            "pondCentroid": analysis["pondCentroid"],
            "estimatedCatchmentAreaSqM": analysis["estimatedCatchmentAreaSqM"],
            "estimatedCatchmentAreaHectares": analysis["estimatedCatchmentAreaHectares"],
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
        return _error_response(500, f"Unexpected server error: {exc}")


@app.post("/analyzeContour/candidates")
async def analyze_contour_candidates(contour_map: UploadFile = File(...)) -> JSONResponse:
    try:
        analysis = await _run_analysis(contour_map)
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
        return _error_response(500, f"Unexpected server error: {exc}")


@app.post("/analyzeContour/raw")
async def analyze_contour_raw(contour_map: UploadFile = File(...)) -> JSONResponse:
    try:
        if not contour_map.filename:
            raise ValueError("A file upload is required.")
        contents = await contour_map.read()
        contours = load_raw_contours(contents, contour_map.filename)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "contourCount": len(contours),
                "contours": contours,
            },
        )
    except ValueError as exc:
        return _error_response(400, str(exc))
    except Exception as exc:  
        return _error_response(500, f"Unexpected server error: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)