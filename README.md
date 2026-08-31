# Assignment 1 - Phase 2: Pond Catchment Analysis Backend

A contour-based pond detection and catchment analysis project that reads KML/KMZ survey data, finds enclosed basin candidates, rejects river-like depressions, and returns pond recommendations with catchment metrics.

## Description

This project analyzes contour maps to estimate where a pond could realistically be built. It reads elevation contours, identifies nested depressions, filters out river/valley-like loops, and ranks the best pond sites by basin depth, compactness, and catchment area.

## Directory structure

```text
Pond-Catchment-Analysis/
├─ backend/
│  ├─ __init__.py              # Package exports (analyze_contour_map, load_raw_contours)
│  ├─ app.py                   # FastAPI application and route definitions
│  ├─ config.py                # Tunable thresholds and constants
│  ├─ geometry.py              # Coordinate and polygon utilities
│  ├─ models.py                # Loop and basin candidate dataclasses
│  ├─ parsing.py               # KML/KMZ parsing helpers
│  └─ service.py               # Core contour analysis pipeline
├─ frontend/
│  └─ streamlit_app.py         # Streamlit dashboard for map and results
├─ tests/
│  └─ test_api.py              # API tests
├─ contours_1m.kml             # Sample contour dataset
├─ requirements.txt            # Python dependencies
├─ README.md                   # Documentation
└─ .venv/                      # Optional local virtual environment
```

## How to set up the project

1. Go to the project root.
2. Create and activate a virtual environment if needed.

```bash
python -m venv .venv
.venv/Scripts/Activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the API:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

5. Start the interactive frontend:

```bash
streamlit run frontend/streamlit_app.py
```

## API endpoints

### 1) Full analysis endpoint

`POST /analyzeContour`

Returns the complete analysis result, including the recommended pond, alternative candidates, pond boundary, terrain summary, and watershed metrics.

for local testing

```bash
curl.exe -X POST "http://127.0.0.1:8000/analyzeContour" -F "file=@contours_1m.kml"
```
for testing on server

```bash
curl.exe -X POST "http://10.1.75.51:4297/analyzeContour" -F "file=@contours_1m.kml"
```

### 2) Summary endpoint

`POST /analyzeContour/summary`

Returns a compact version with the main results only.

for local testing

```bash
curl.exe -X POST "http://127.0.0.1:8000/analyzeContour/summary" -F "file=@contours_1m.kml"
```

for testing on server

```bash
curl.exe -X POST "http://10.1.75.51:4297/analyzeContour/summary" -F "file=@contours_1m.kml"
```

### 3) Candidate endpoint

`POST /analyzeContour/candidates`

Returns the recommended site and alternative pond candidates only.

for local testing

```bash
curl.exe -X POST "http://127.0.0.1:8000/analyzeContour/candidates" -F "file=@contours_1m.kml"
```

for testing on server

```bash
curl.exe -X POST "http://10.1.75.51:4297/analyzeContour/candidates" -F "file=@contours_1m.kml"
```


### 4) Raw contours endpoint

`POST /analyzeContour/raw`

Returns raw parsed contour lines without analysis, useful for map rendering or inspection.

for local testing

```bash
curl.exe -X POST "http://127.0.0.1:8000/analyzeContour/raw" -F "file=@contours_1m.kml"
```

for testing on server

```bash
curl.exe -X POST "http://10.1.75.51:4297/analyzeContour/raw" -F "file=@contours_1m.kml"
```

### Legacy compatibility endpoint

`POST /findCatchment`

an alias for the main analysis route.

## How to use the interactive app

Run the Streamlit dashboard:

```bash
streamlit run frontend/streamlit_app.py
```

Then upload a `.kml` or `.kmz` contour file in the UI. The app displays:

- contour lines on a map
- suggested pond location(s)
- terrain metrics
- basin depth and catchment area
- a downloadable JSON result

## API response overview

The main response contains fields such as:

- `status`
- `pondElevation`
- `pondCentroid`
- `estimatedCatchmentAreaSqM`
- `estimatedCatchmentAreaHectares`
- `basinAreaSqM`
- `basinDepthM`
- `compactnessScore`
- `confidenceScore`
- `terrainSummary`
- `pondCandidates`
- `alternativeCandidates`
- `basinBoundary`

## Production-ready modular structure

The backend was split into smaller modules to improve readability, reusability, and maintainability:

- `__init__.py` exports the public API (`analyze_contour_map`, `load_raw_contours`) for clean package-level imports.
- `parsing.py` handles input extraction from KML/KMZ files.
- `geometry.py` contains the coordinate transform and polygon calculations.
- `models.py` defines the geometry entities used in the analysis.
- `config.py` contains all threshold constants.
- `service.py` actual pond detection logic.
- `app.py` exposes the HTTP interface and the API routes.

## Notes

- The app uploads only `.kml` and `.kmz` files.
- River-like elongated contour loops are filtered out to avoid false pond suggestions.

## Quick example

for local testing

```bash
curl.exe -X POST "http://127.0.0.1:8000/analyzeContour" -F "file=@contours_1m.kml"
```

for testing on server

```bash
curl.exe -X POST "http://10.1.75.51:4297/analyzeContour" -F "file=@contours_1m.kml"
```
