/**
 * app.js — Pond Catchment Analysis frontend
 *
 * Architecture:
 *   - Leaflet map with Leaflet.draw for polygon selection
 *   - Two modes:  "dem"    → POST /api/analyzeDemArea  then /api/fetchRainfall
 *                "upload" → POST /api/analyzeContourWithArea then /api/fetchRainfall
 *   - Chart.js bar chart for monthly ERA5 precipitation
 *   - All area calculations done client-side (no extra library needed)
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants & Config
// ---------------------------------------------------------------------------
let maxAreaKm2      = 25.0;     // Dynamically synced with /api/config
const INDIA_CENTER  = [21.25, 81.29];
const INDIA_ZOOM    = 6;
const RUNOFF_COEFF  = 0.20;     // Must match backend/rainfall.py

const MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const CANDIDATE_COLORS = ["#3b82f6","#f59e0b","#8b5cf6","#10b981","#ef4444","#ec4899","#06b6d4"];

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (res.ok) {
      const cfg = await res.json();
      if (typeof cfg.max_area_km2 === "number") {
        maxAreaKm2 = cfg.max_area_km2;
        const hintEl = document.getElementById("max-area-hint");
        if (hintEl) hintEl.textContent = `${maxAreaKm2} km²`;
      }
    }
  } catch (err) {
    console.debug("Config sync fallback to default:", err);
  }
}


// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
const state = {
  mode:          "dem",      // "dem" | "upload"
  drawnLayer:    null,       // Leaflet polygon layer
  polygon:       null,       // [[lon, lat], ...] closed ring
  areaKm2:       null,       // number
  uploadedFile:  null,       // File object
  result:        null,       // Latest analysis result dict
  rainfallSeries: null,      // Latest rainfall series array
  kmlB64:        null,       // Base64 KML string for download
  resultLayers:  [],         // Leaflet layers added for results
  rainfallChart: null,       // Chart.js instance
};

// ---------------------------------------------------------------------------
// Map setup
// ---------------------------------------------------------------------------
let map, drawnItems, drawControl;

function initMap() {
  map = L.map("map", { center: INDIA_CENTER, zoom: INDIA_ZOOM });

  // Tile layer — OpenStreetMap
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>",
    maxZoom: 20,
  }).addTo(map);

  // Feature group to hold drawn polygons
  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  // Draw control — polygon only
  drawControl = new L.Control.Draw({
    draw: {
      polygon: {
        allowIntersection: false,
        showArea: true,
        drawError: { color: "#ef4444", message: "Self-intersection not allowed" },
        shapeOptions: { color: "#3b82f6", weight: 2, fillOpacity: 0.10 },
      },
      polyline: false, rectangle: false, circle: false,
      circlemarker: false, marker: false,
    },
    edit: {
      featureGroup: drawnItems,
      edit: { selectedPathOptions: { maintainColor: true } },
    },
  });
  map.addControl(drawControl);

  // Events
  map.on(L.Draw.Event.CREATED,   onPolygonCreated);
  map.on(L.Draw.Event.EDITED,    onPolygonEdited);
  map.on(L.Draw.Event.DELETED,   onPolygonDeleted);
}

// ---------------------------------------------------------------------------
// Area calculation (equirectangular, accurate for small areas < 10 km²)
// ---------------------------------------------------------------------------
function calculateAreaKm2(latlngs) {
  if (!latlngs || latlngs.length < 3) return 0;
  const centerLat = latlngs.reduce((s, p) => s + p.lat, 0) / latlngs.length;
  const xScale = 111.32 * Math.cos(centerLat * Math.PI / 180); // km per degree lon
  const yScale = 110.574;                                        // km per degree lat
  const pts = latlngs.map(p => [p.lng * xScale, p.lat * yScale]);
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
  }
  return Math.abs(area) / 2;
}

// Convert Leaflet LatLng array → [[lon, lat], ...] closed ring for API
function latlngsToClosedRing(latlngs) {
  const ring = latlngs.map(p => [p.lng, p.lat]);
  if (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1]) {
    ring.push([...ring[0]]);
  }
  return ring;
}

// ---------------------------------------------------------------------------
// Draw event handlers
// ---------------------------------------------------------------------------
function onPolygonCreated(e) {
  // Remove previous drawing
  drawnItems.clearLayers();
  state.drawnLayer = e.layer;
  drawnItems.addLayer(e.layer);
  updatePolygonState(e.layer.getLatLngs()[0]);
}

function onPolygonEdited(e) {
  e.layers.eachLayer(layer => updatePolygonState(layer.getLatLngs()[0]));
}

function onPolygonDeleted() {
  state.polygon = null;
  state.drawnLayer = null;
  state.areaKm2 = null;
  updateAreaDisplay(null);
  setAnalyzeEnabled(false);
  showClearBtn(false);
}

function updatePolygonState(latlngs) {
  const area = calculateAreaKm2(latlngs);
  state.polygon  = latlngsToClosedRing(latlngs);
  state.areaKm2  = area;
  updateAreaDisplay(area);
  const overLimit = area > maxAreaKm2;
  setAnalyzeEnabled(!overLimit && canAnalyze());
  showClearBtn(true);
  if (overLimit) {
    setStatus(
      `Area ${area.toFixed(2)} km² exceeds the ${maxAreaKm2} km² limit — shrink the polygon.`,
      "error", false
    );
  } else {
    clearStatus();
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function updateAreaDisplay(area) {
  const demDisplay    = document.getElementById("dem-area-display");
  const demValue      = document.getElementById("dem-area-value");
  const uploadDisplay = document.getElementById("upload-area-display");
  const uploadValue   = document.getElementById("upload-area-value");

  if (area === null) {
    demDisplay.classList.add("hidden");
    uploadDisplay.classList.add("hidden");
    return;
  }

  const over = area > maxAreaKm2;
  const text = `${area.toFixed(2)} km² ${over ? "⚠ too large" : "✓ OK"}`;
  const cls  = over ? "over-limit" : "ok";

  demDisplay.classList.remove("hidden", "over-limit", "ok");
  demDisplay.classList.add(cls);
  demValue.textContent = text;

  uploadDisplay.classList.remove("hidden", "over-limit", "ok");
  uploadDisplay.classList.add(cls);
  uploadValue.textContent = text;
}

function canAnalyze() {
  if (state.mode === "dem") {
    return !!state.polygon;
  }
  // upload mode: need a file; polygon is optional (analyzes entire file if absent)
  return !!state.uploadedFile;
}

function setAnalyzeEnabled(enabled) {
  document.getElementById("analyze-dem-btn").disabled    = !enabled;
  document.getElementById("analyze-upload-btn").disabled = !enabled;
}

function showClearBtn(show) {
  document.getElementById("clear-dem-btn").classList.toggle("hidden", !show);
  document.getElementById("clear-upload-btn").classList.toggle("hidden", !show);
}

function setStatus(msg, type = "info", spinner = true) {
  const bar  = document.getElementById("status-bar");
  const text = document.getElementById("status-text");
  const spin = document.getElementById("status-spinner");
  bar.classList.remove("hidden", "error", "success");
  if (type === "error")   bar.classList.add("error");
  if (type === "success") bar.classList.add("success");
  spin.classList.toggle("hidden", !spinner);
  text.textContent = msg;
  bar.classList.remove("hidden");
}

function clearStatus() {
  document.getElementById("status-bar").classList.add("hidden");
}

function clearResultLayers() {
  state.resultLayers.forEach(l => map.removeLayer(l));
  state.resultLayers = [];
}

function clearResults() {
  clearResultLayers();
  document.getElementById("results-panel").classList.add("hidden");
  document.getElementById("metrics-grid").innerHTML = "";
  document.getElementById("runoff-summary").classList.add("hidden");
  if (state.rainfallChart) {
    state.rainfallChart.destroy();
    state.rainfallChart = null;
  }
  document.getElementById("rainfall-loading").classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------
function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.mode === mode));
  document.getElementById("dem-panel").classList.toggle("hidden",    mode !== "dem");
  document.getElementById("upload-panel").classList.toggle("hidden", mode !== "upload");
  setAnalyzeEnabled(canAnalyze() && !!state.polygon && state.areaKm2 <= maxAreaKm2);
  clearResults();
  clearStatus();
}

// ---------------------------------------------------------------------------
// Polygon clear
// ---------------------------------------------------------------------------
function clearPolygon() {
  drawnItems.clearLayers();
  state.drawnLayer = null;
  state.polygon    = null;
  state.areaKm2    = null;
  updateAreaDisplay(null);
  setAnalyzeEnabled(canAnalyze() && false);
  showClearBtn(false);
  clearStatus();
}

// ---------------------------------------------------------------------------
// DEM analysis
// ---------------------------------------------------------------------------
async function analyzeDem() {
  if (!state.polygon) { setStatus("Draw a polygon on the map first.", "error", false); return; }
  if (state.areaKm2 > maxAreaKm2) {
    setStatus(`Area ${state.areaKm2.toFixed(2)} km² is too large. Draw a smaller polygon.`, "error", false);
    return;
  }

  const dataset = document.getElementById("dataset-select").value;
  clearResults();
  setStatus("Downloading elevation data from OpenTopography…", "info", true);
  document.getElementById("analyze-dem-btn").disabled = true;

  try {
    const resp = await fetch("/api/analyzeDemArea", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ area_polygon: state.polygon, dataset }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

    state.result = data.result;
    state.kmlB64 = data.kml_b64;

    renderResultsOnMap(data.result);
    renderMetrics(data.result);
    setStatus("Terrain analysis complete. Fetching rainfall…", "success", true);

    await fetchAndRenderRainfall(data.result);
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, "error", false);
  } finally {
    document.getElementById("analyze-dem-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Upload + area analysis
// ---------------------------------------------------------------------------
async function analyzeUpload() {
  if (!state.uploadedFile) { setStatus("Choose a KML or KMZ file first.", "error", false); return; }

  clearResults();
  setStatus("Parsing survey file…", "info", true);
  document.getElementById("analyze-upload-btn").disabled = true;

  const formData = new FormData();
  formData.append("file", state.uploadedFile);
  // Send polygon if drawn; otherwise send empty so backend uses full file extent
  formData.append("area_polygon_json", JSON.stringify(state.polygon || []));

  try {
    const resp = await fetch("/api/analyzeContourWithArea", {
      method: "POST",
      body:   formData,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

    state.result = data;
    state.kmlB64 = null; // upload mode doesn't return KML

    renderResultsOnMap(data);
    renderMetrics(data);
    setStatus("Survey analysis complete. Fetching rainfall…", "success", true);

    await fetchAndRenderRainfall(data);
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, "error", false);
  } finally {
    document.getElementById("analyze-upload-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Rainfall fetch + render
// ---------------------------------------------------------------------------
async function fetchAndRenderRainfall(result) {
  const candidates = result.pondCandidates || [];
  if (!candidates.length) { setStatus("No candidates found in this area.", "error", false); return; }

  const locations = candidates.map(c => [c.pondCentroid.lat, c.pondCentroid.lon]);
  try {
    const resp = await fetch("/api/fetchRainfall", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ locations, years: 10 }),
    });
    const series = await resp.json();
    if (!resp.ok) throw new Error(series.error || `HTTP ${resp.status}`);

    state.rainfallSeries = series;
    renderRainfallChart(series[0]);               // chart for recommended site
    renderRunoffSummary(result, series[0]);
    setStatus("Done.", "success", false);
    setTimeout(clearStatus, 3000);
  } catch (err) {
    document.getElementById("rainfall-loading").textContent = `Rainfall unavailable: ${err.message}`;
    setStatus(`Rainfall fetch failed: ${err.message}`, "error", false);
  }
}

// ---------------------------------------------------------------------------
// Map result rendering
// ---------------------------------------------------------------------------
const RESULT_STYLE = {
  catchment: { color: "#3b82f6", weight: 1.5, fillColor: "#3b82f6", fillOpacity: 0.07, dashArray: "5,4" },
  basin:     { color: "#10b981", weight: 2,   fillColor: "#10b981", fillOpacity: 0.12 },
};

function renderResultsOnMap(result) {
  clearResultLayers();
  const candidates = result.pondCandidates || [];
  const bounds = [];

  candidates.forEach((c, i) => {
    const color     = CANDIDATE_COLORS[i % CANDIDATE_COLORS.length];
    const isRec     = i === 0;
    const weight    = isRec ? 3 : 1.5;
    const radius    = isRec ? 12 : 8;

    // Basin boundary
    if (c.basinBoundaryGeoJSON) {
      try {
        const layer = L.geoJSON(c.basinBoundaryGeoJSON, {
          style: { ...RESULT_STYLE.basin, color, fillColor: color },
        }).addTo(map);
        state.resultLayers.push(layer);
        layer.eachLayer(l => { if (l.getBounds) bounds.push(...Object.values(l.getBounds())); });
      } catch { /* ignore bad GeoJSON */ }
    }

    // Catchment boundary
    if (c.catchmentBoundaryGeoJSON) {
      try {
        const layer = L.geoJSON(c.catchmentBoundaryGeoJSON, {
          style: { ...RESULT_STYLE.catchment, color },
        }).addTo(map);
        state.resultLayers.push(layer);
      } catch { /* ignore */ }
    }

    // Pond site marker
    const { lat, lon } = c.pondCentroid;
    const marker = L.circleMarker([lat, lon], {
      radius,
      color:       "#fff",
      weight:      2,
      fillColor:   color,
      fillOpacity: 0.9,
    }).addTo(map);

    const tag    = isRec ? "⭐ Recommended" : `Alt #${i}`;
    const depth  = (c.basinDepthM || 0).toFixed(1);
    const vol    = formatVolume(c.estimatedVolumeM3 || 0);
    const area   = formatArea(c.estimatedCatchmentAreaSqM || 0);
    marker.bindPopup(
      `<b>${tag}</b><br>` +
      `Elevation: <b>${(c.pondElevation || 0).toFixed(1)} m</b><br>` +
      `Basin depth: <b>${depth} m</b><br>` +
      `Storage: <b>${vol}</b><br>` +
      `Catchment: <b>${area}</b><br>` +
      `Confidence: <b>${((c.confidenceScore || 0) * 100).toFixed(0)}%</b>`
    );
    if (isRec) marker.openPopup();
    state.resultLayers.push(marker);
    bounds.push([lat, lon]);
  });

  if (bounds.length) {
    try { map.fitBounds(L.latLngBounds(bounds), { padding: [40, 40] }); } catch { /* ignore */ }
  }

  document.getElementById("results-panel").classList.remove("hidden");
  document.getElementById("download-row").classList.toggle("hidden", !state.kmlB64);
}

// ---------------------------------------------------------------------------
// Metrics cards
// ---------------------------------------------------------------------------
function renderMetrics(result) {
  const rec = (result.pondCandidates || [])[0];
  if (!rec) return;

  const depth   = (rec.basinDepthM        || 0).toFixed(1);
  const vol     = formatVolume(rec.estimatedVolumeM3 || 0);
  const cat     = formatArea(rec.estimatedCatchmentAreaSqM || 0);
  const elev    = (rec.pondElevation       || 0).toFixed(1);
  const conf    = `${((rec.confidenceScore || 0) * 100).toFixed(0)}%`;
  const lat     = (rec.pondCentroid?.lat   || 0).toFixed(5);
  const lon     = (rec.pondCentroid?.lon   || 0).toFixed(5);

  const grid = document.getElementById("metrics-grid");
  grid.innerHTML = [
    { label: "Elevation",       value: `${elev}`,  unit: "m" },
    { label: "Basin Depth",     value: `${depth}`, unit: "m" },
    { label: "Est. Storage",    value: vol,         unit: "" },
    { label: "Catchment Area",  value: cat,         unit: "" },
    { label: "Confidence",      value: conf,        unit: "" },
    { label: "Coordinates",     value: `${lat}, ${lon}`, unit: "" },
  ].map(m => `
    <div class="metric-card">
      <div class="metric-label">${m.label}</div>
      <div class="metric-value">${m.value} <span class="metric-unit">${m.unit}</span></div>
    </div>
  `).join("");
}

// ---------------------------------------------------------------------------
// Rainfall chart
// ---------------------------------------------------------------------------
function renderRainfallChart(series) {
  const loadingEl = document.getElementById("rainfall-loading");
  loadingEl.classList.add("hidden");

  if (!series || !series.monthlyMeanPrecipitationM) return;

  const monthly_mm = series.monthlyMeanPrecipitationM.map(v => +(v * 1000).toFixed(1));
  const ctx = document.getElementById("rainfall-chart").getContext("2d");

  if (state.rainfallChart) state.rainfallChart.destroy();

  state.rainfallChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels:   MONTH_LABELS,
      datasets: [{
        label:           "Avg monthly rainfall (mm)",
        data:            monthly_mm,
        backgroundColor: monthly_mm.map(v => v > 80 ? "#3b82f6" : v > 30 ? "#60a5fa" : "#93c5fd"),
        borderRadius:    3,
        borderSkipped:   false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y} mm`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", font: { size: 9 } },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
        y: {
          ticks: { color: "#94a3b8", font: { size: 9 } },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Runoff summary card
// ---------------------------------------------------------------------------
function renderRunoffSummary(result, series) {
  const rec = (result.pondCandidates || [])[0];
  if (!rec || !series) return;

  const annualMm    = (series.meanAnnualPrecipitationM || 0) * 1000;
  const catchSqM    = rec.estimatedCatchmentAreaSqM || 0;
  const storageCuM  = Math.max(0, rec.estimatedVolumeM3 || 0);
  const runoffCuM   = series.meanAnnualPrecipitationM * catchSqM * RUNOFF_COEFF;
  const fillable    = Math.min(runoffCuM, storageCuM);

  const el = document.getElementById("runoff-summary");
  el.innerHTML =
    `Annual rainfall: <strong>${annualMm.toFixed(0)} mm</strong><br>` +
    `Potential runoff: <strong>${formatVolume(runoffCuM)}</strong> (${(RUNOFF_COEFF * 100).toFixed(0)}% coefficient)<br>` +
    `Est. fillable water: <strong>${formatVolume(fillable)}</strong>`;
  el.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------
function downloadKML() {
  if (!state.kmlB64) return;
  const bytes = atob(state.kmlB64);
  const arr   = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  const blob  = new Blob([arr], { type: "application/vnd.google-earth.kml+xml" });
  triggerDownload(blob, "pond_analysis.kml");
}

function downloadJSON() {
  if (!state.result) return;
  const payload = { result: state.result, rainfall: state.rainfallSeries };
  const blob    = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  triggerDownload(blob, "pond_analysis.json");
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement("a");
  a.href    = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------
function formatVolume(m3) {
  if (m3 >= 1_000_000) return `${(m3 / 1_000_000).toFixed(2)} Mm³`;
  if (m3 >= 1_000)     return `${(m3 / 1_000).toFixed(1)} k m³`;
  return `${m3.toFixed(0)} m³`;
}

function formatArea(m2) {
  if (m2 >= 10_000) return `${(m2 / 10_000).toFixed(2)} ha`;
  return `${m2.toFixed(0)} m²`;
}

// ---------------------------------------------------------------------------
// Wire up event listeners
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadConfig();

  // Mode tab switch
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => switchMode(tab.dataset.mode));
  });

  // DEM analyze
  document.getElementById("analyze-dem-btn").addEventListener("click", analyzeDem);

  // Upload analyze
  document.getElementById("analyze-upload-btn").addEventListener("click", analyzeUpload);

  // Clear polygon buttons
  document.getElementById("clear-dem-btn").addEventListener("click",    clearPolygon);
  document.getElementById("clear-upload-btn").addEventListener("click", clearPolygon);

  // File input
  const fileInput = document.getElementById("file-input");
  const fileLabel = document.getElementById("file-label");
  const fileDrop  = document.getElementById("file-drop");

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    state.uploadedFile = file;
    fileLabel.textContent = `📄 ${file.name}`;
    fileDrop.classList.add("has-file");
    setAnalyzeEnabled(canAnalyze());
  });

  // Drag-and-drop onto file zone
  fileDrop.addEventListener("dragover", e => { e.preventDefault(); fileDrop.style.borderColor = "var(--primary)"; });
  fileDrop.addEventListener("dragleave", ()  => { fileDrop.style.borderColor = ""; });
  fileDrop.addEventListener("drop", e => {
    e.preventDefault();
    fileDrop.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (!file) return;
    fileInput.files = e.dataTransfer.files;
    state.uploadedFile = file;
    fileLabel.textContent = `📄 ${file.name}`;
    fileDrop.classList.add("has-file");
    setAnalyzeEnabled(canAnalyze());
  });

  // Downloads
  document.getElementById("dl-kml-btn").addEventListener("click",  downloadKML);
  document.getElementById("dl-json-btn").addEventListener("click",  downloadJSON);
});
