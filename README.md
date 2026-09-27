# Phase 3: Pond Catchment and Rainfall Analysis

A Streamlit map and FastAPI service for finding pond-site candidates from contour surveys or global elevation data, delineating DEM catchments, and estimating rainfall-driven runoff and pond storage.

## End-to-end algorithm

1. **Choose an input mode.** Upload a KML/KMZ contour survey, or draw an area on the map and choose an OpenTopography global DEM dataset.
2. **Prepare terrain.** Upload mode parses contour elevations and keeps closed contour rings within the selected polygon. Map mode requests a GeoTIFF DEM from OpenTopography, creates contour KML for download, and analyzes the raster directly.
3. **Find pond depressions.** For DEM input, `pysheds` fills pits and depressions. The difference between original and filled elevations identifies connected depression cells; components at least 1 m deep and 3 cells in size are candidates. Estimated storage is the sum of cell fill-depth times cell area.
4. **Delineate catchments.** D8 flow direction, accumulation, and catchment tracing estimate the upslope area contributing to each candidate. Uploaded KML mode retains the nested-contour basin analysis from the earlier phase.
5. **Add rainfall and rank sites.** Once terrain candidates exist, the app automatically makes one batched request for all candidates to Open-Meteo ERA5 daily precipitation. It averages the last 10 complete calendar years, estimates annual runoff with a fixed 0.20 coefficient, and estimates annual fillable water as the lesser of runoff potential and pond storage capacity. Candidates are ranked by fillable water, then runoff and storage.
6. **Display results.** The map shows pond markers, depression basins, and D8 catchment boundaries. Results include area in m², depth in m, storage/runoff/fillable water in m³, plus a monthly precipitation chart. Generated contour KML and the complete JSON result can be downloaded.

## Setup

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Map-selected terrain analysis requires an OpenTopography key in the root `.env` file:

```text
API_Key=your-personal-key
```

The server reads the key; it is not requested in the frontend. Keep `.env` out of version control. OpenTopography prohibits sharing one user's key with other users and applies daily quotas. Rainfall comes from the Open-Meteo Historical Weather API, which has separate usage terms and limits.

Start the frontend:

```powershell
streamlit run frontend/streamlit_app.py
```

Streamlit defaults to port 8501. Select another port if needed:

```powershell
streamlit run frontend/streamlit_app.py --server.port 8502
```

Start the API separately when needed:

```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Run tests:

```powershell
python -m pytest -q
```

## Limits and interpretation

- KML/KMZ uploads are limited to 25 MB. Global DEM requests are limited to a 25 km² bounding area, 32 MB, 2 million raster cells, and generated-contour processing limits.
- The fallback returns the lowest valid DEM cell if no depression passes the candidate thresholds; it reports zero storage and is not a detected pond basin.
- Global elevation datasets are roughly 30 m resolution and may miss small features. ERA5 rainfall is roughly 25 km resolution; the weather grid cell can be offset from a site and is not a local gauge measurement.
- Runoff assumes a fixed 0.20 coefficient and omits site-specific infiltration, evaporation, soil/land-cover calibration, conveyance loss, and operating rules. Annual fillable water is an estimate, not a guaranteed yield.
- Outputs are preliminary screening estimates, not construction or safety design recommendations.
- Backend progress logs appear in the terminal running Streamlit/Uvicorn. Logs include request bounds, provider status, raster/contour counts, and analysis status; API keys are not logged.

## API endpoints

Contour uploads use multipart field `contour_map` (the legacy `file` field is also accepted):

- `POST /analyzeContour`: complete contour-analysis result.
- `POST /analyzeContour/summary`: compact analysis summary.
- `POST /analyzeContour/candidates`: recommended and alternative contour candidates.
- `POST /analyzeContour/raw`: parsed contour lines.
- `POST /findCatchment`: legacy alias for `/analyzeContour`.

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyzeContour" -F "contour_map=@contours_1m.kml"
```