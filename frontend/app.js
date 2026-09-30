/**
 * app.js — Pond Catchment Analysis Frontend (Streamlit-styled)
 *
 * Capabilities:
 *   - Leaflet interactive GIS map with polygon draw tool & layer toggles
 *   - Streamlit-style UI (st.sidebar, st.metric row, st.tabs, st.table)
 *   - DEM Analysis (/api/analyzeDemArea) + Historical Rainfall (/api/fetchRainfall)
 *   - KML / KMZ Upload Analysis (/api/analyzeContourWithArea)
 *   - Chart.js 10-year monthly precipitation bar chart with runoff estimation
 *   - Downloadable KML and JSON technical payloads
 */

"use strict";

// ---------------------------------------------------------------------------
// Configuration & Constants
// ---------------------------------------------------------------------------
let maxAreaKm2 = 0.5; // Dynamically synchronized with backend /api/config
const INDIA_CENTER = [21.25, 81.29];
const INDIA_ZOOM = 6;
const RUNOFF_COEFF = 0.20; // Runoff coefficient C = 0.20

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const CANDIDATE_COLORS = [
  "#ff4b4b", // Primary recommended (Streamlit Red)
  "#ffa421", // Alt 1 (Amber)
  "#1c83e1", // Alt 2 (Blue)
  "#9c27b0", // Alt 3 (Purple)
  "#00bcd4", // Alt 4 (Cyan)
  "#4caf50", // Alt 5 (Green)
];

// ---------------------------------------------------------------------------
// Global Application State
// ---------------------------------------------------------------------------
const state = {
  mode: "dem", // "dem" | "upload"
  drawnLayer: null,
  polygon: null, // [[lon, lat], ...] closed coordinate ring
  areaKm2: null,
  uploadedFile: null,
  result: null, // Full analysis response payload
  rainfallSeries: null,
  kmlB64: null,
  resultLayers: [],
  rainfallChart: null,
  villageMarker: null,
  currentSuggestions: [],
  selectedSuggestionIndex: -1,
};

// ---------------------------------------------------------------------------
// Map Setup & Initialization
// ---------------------------------------------------------------------------
let map, drawnItems, drawControl;

function initMap() {
  map = L.map("map", {
    center: INDIA_CENTER,
    zoom: INDIA_ZOOM,
    zoomControl: true,
    preferCanvas: true, // Use HTML5 Canvas renderer to prevent SVG DOM lag
  });

  // Free Base Map Tile Layers (No API Key Required)
  const osmStandard = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
    maxZoom: 19,
  });

  const esriSatellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    maxZoom: 19,
  });

  const openTopo = L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
    attribution: "Map data: © <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors, SRTM | Map style: © <a href='https://opentopomap.org'>OpenTopoMap</a>",
    maxZoom: 17,
  });

  // Default to OpenStreetMap Standard
  osmStandard.addTo(map);

  // Basemap switch control (Top-left)
  L.control.layers({
    "🗺️ OpenStreetMap": osmStandard,
    "🛰️ Satellite Imagery (Esri)": esriSatellite,
    "🏔️ Topographic Map": openTopo,
  }, null, { position: "topleft" }).addTo(map);

  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  drawControl = new L.Control.Draw({
    position: "topright",
    draw: {
      polygon: {
        allowIntersection: false,
        showArea: true,
        drawError: { color: "#ff4b4b", message: "Self-intersection is not allowed" },
        shapeOptions: { color: "#ff4b4b", weight: 2.5, fillOpacity: 0.12 },
      },
      polyline: false,
      rectangle: false,
      circle: false,
      circlemarker: false,
      marker: false,
    },
    edit: {
      featureGroup: drawnItems,
      edit: { selectedPathOptions: { maintainColor: true } },
    },
  });
  map.addControl(drawControl);

  state.layerGroups = {
    catchments: L.featureGroup().addTo(map),
    basins: L.featureGroup().addTo(map),
    markers: L.featureGroup().addTo(map),
  };

  const overlayMaps = {
    "<span style='font-size:12px; font-weight:600; color:#1c83e1;'>🌐 Catchment Area (D8 Drainage)</span>": state.layerGroups.catchments,
    "<span style='font-size:12px; font-weight:600; color:#09ab3b;'>💧 Pond Basin Bed / Rim</span>": state.layerGroups.basins,
    "<span style='font-size:12px; font-weight:600; color:#ff4b4b;'>⭐ Candidate Siting Marker</span>": state.layerGroups.markers,
  };

  state.layerControl = L.control.layers(null, overlayMaps, {
    collapsed: false,
    position: "topright",
  }).addTo(map);

  map.on(L.Draw.Event.CREATED, onPolygonCreated);
  map.on(L.Draw.Event.EDITED, onPolygonEdited);
  map.on(L.Draw.Event.DELETED, onPolygonDeleted);

  setTimeout(() => {
    map.invalidateSize();
  }, 200);
}

// ---------------------------------------------------------------------------
// Dynamic Backend Configuration Sync
// ---------------------------------------------------------------------------
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
    console.debug("Config sync fallback:", err);
  }
}

// ---------------------------------------------------------------------------
// Area & Polygon Utilities
// ---------------------------------------------------------------------------
function calculateAreaKm2(latlngs) {
  if (!latlngs || latlngs.length < 3) return 0;
  const centerLat = latlngs.reduce((sum, p) => sum + p.lat, 0) / latlngs.length;
  const xScale = 111.32 * Math.cos((centerLat * Math.PI) / 180);
  const yScale = 110.574;
  const pts = latlngs.map(p => [p.lng * xScale, p.lat * yScale]);
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
  }
  return Math.abs(area) / 2;
}

function latlngsToClosedRing(latlngs) {
  const ring = latlngs.map(p => [p.lng, p.lat]);
  if (ring[0][0] !== ring[ring.length - 1][0] || ring[0][1] !== ring[ring.length - 1][1]) {
    ring.push([...ring[0]]);
  }
  return ring;
}

// ---------------------------------------------------------------------------
// Draw Handlers
// ---------------------------------------------------------------------------
function onPolygonCreated(e) {
  clearResults();
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
  updateAreaFeedback(null);
  setAnalyzeEnabled(canAnalyze());
  showClearBtn(false);
  clearStatus();
}

function updatePolygonState(latlngs) {
  const area = calculateAreaKm2(latlngs);
  state.polygon = latlngsToClosedRing(latlngs);
  state.areaKm2 = area;
  updateAreaFeedback(area, latlngs);

  const overLimit = area > maxAreaKm2;
  setAnalyzeEnabled(!overLimit && canAnalyze());
  showClearBtn(true);
  if (overLimit) {
    setStatus(`Selected area (${area.toFixed(2)} km²) exceeds the ${maxAreaKm2} km² limit.`, "error", false);
  } else {
    clearStatus();
  }
}

// ---------------------------------------------------------------------------
// UI Feedback & Helpers
// ---------------------------------------------------------------------------
function updateAreaFeedback(area, latlngs) {
  const card = document.getElementById("selection-area-card");
  const badge = document.getElementById("selection-area-val");
  const coords = document.getElementById("selection-coords-text");

  if (area === null) {
    card.classList.add("hidden");
    return;
  }

  const over = area > maxAreaKm2;
  const tooSmall = area < 0.01;
  const areaM2 = Math.round(area * 1_000_000);
  let badge_label;
  if (over) {
    badge_label = `${area.toFixed(3)} km² (${areaM2.toLocaleString()} m²) ⚠️ Exceeds Limit (max ${maxAreaKm2} km²)`;
  } else if (tooSmall) {
    badge_label = `${area.toFixed(4)} km² ⚠️ Too Small (min 0.01 km²)`;
  } else {
    badge_label = `${area.toFixed(3)} km² (${areaM2.toLocaleString()} m²)`;
  }
  badge.textContent = badge_label;
  badge.classList.toggle("over-limit", over);

  if (latlngs && latlngs.length > 0) {
    const lats = latlngs.map(p => p.lat);
    const lngs = latlngs.map(p => p.lng);
    const minLat = Math.min(...lats).toFixed(3);
    const maxLat = Math.max(...lats).toFixed(3);
    const minLng = Math.min(...lngs).toFixed(3);
    const maxLng = Math.max(...lngs).toFixed(3);
    coords.textContent = `Lat: [${minLat}..${maxLat}], Lon: [${minLng}..${maxLng}]`;
  }

  card.classList.remove("hidden");
}

function canAnalyze() {
  if (state.mode === "dem") {
    return !!state.polygon && state.areaKm2 <= maxAreaKm2;
  }
  return !!state.uploadedFile;
}

function setAnalyzeEnabled(enabled) {
  document.getElementById("btn-run-analysis").disabled = !enabled;
}

function showClearBtn(show) {
  const btn = document.getElementById("btn-clear-selection");
  if (btn) btn.classList.remove("hidden");
}

function setStatus(msg, type = "info", spinner = true) {
  const box = document.getElementById("st-status-box");
  const text = document.getElementById("st-status-text");
  const spin = document.getElementById("st-status-spinner");

  spin.classList.toggle("hidden", !spinner);
  text.textContent = msg;
  box.classList.remove("hidden");
}

function clearStatus() {
  document.getElementById("st-status-box").classList.add("hidden");
}

function clearResultLayers() {
  if (state.layerGroups) {
    state.layerGroups.catchments.clearLayers();
    state.layerGroups.basins.clearLayers();
    state.layerGroups.markers.clearLayers();
  }
  state.resultLayers.forEach(layer => {
    try { map.removeLayer(layer); } catch (_) {}
  });
  state.resultLayers = [];
}

function clearResults() {
  clearResultLayers();
  document.getElementById("results-wrapper").classList.add("hidden");
  if (state.rainfallChart) {
    state.rainfallChart.destroy();
    state.rainfallChart = null;
  }
  document.getElementById("rainfall-chart-loader").classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Mode Switching
// ---------------------------------------------------------------------------
function switchMode(mode) {
  state.mode = mode;

  document.querySelectorAll(".st-radio-option").forEach(opt => {
    opt.classList.toggle("active", opt.dataset.mode === mode);
    const radio = opt.querySelector("input[type='radio']");
    if (radio) radio.checked = opt.dataset.mode === mode;
  });

  document.getElementById("panel-dem-options").classList.toggle("hidden", mode !== "dem");
  document.getElementById("panel-upload-options").classList.toggle("hidden", mode !== "upload");

  setAnalyzeEnabled(canAnalyze());
  clearResults();
  clearStatus();
}

function clearPolygon() {
  setStatus("Resetting application...", "info", true);
  window.location.reload();
}

// ---------------------------------------------------------------------------
// DEM Analysis Execution
// ---------------------------------------------------------------------------
async function runAnalysis() {
  if (state.mode === "dem") {
    await analyzeDem();
  } else {
    await analyzeUpload();
  }
}

async function analyzeDem() {
  if (!state.polygon) {
    setStatus("Please draw a region on the map first.", "error", false);
    return;
  }
  if (state.areaKm2 > maxAreaKm2) {
    setStatus(`Selected area (${state.areaKm2.toFixed(2)} km²) exceeds ${maxAreaKm2} km² limit.`, "error", false);
    return;
  }

  const dataset = document.getElementById("dataset-select").value;
  const resolutionSelect = document.getElementById("resolution-select");
  const resolutionM = resolutionSelect ? parseFloat(resolutionSelect.value) : 10.0;

  clearResults();
  setStatus(`Fetching ${dataset} (${resolutionM}m village scale) & computing D8 hydrology...`, "info", true);
  document.getElementById("btn-run-analysis").disabled = true;

  try {
    const resp = await fetch("/api/analyzeDemArea", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        area_polygon: state.polygon,
        dataset,
        resolution_m: resolutionM,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

    state.result = data.result;
    state.kmlB64 = data.kml_b64;

    renderResults(data.result);
    setStatus("Terrain analysis complete. Querying ERA5 rainfall reanalysis...", "info", true);

    await fetchAndRenderRainfall(data.result);
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, "error", false);
  } finally {
    document.getElementById("btn-run-analysis").disabled = false;
  }
}

async function analyzeUpload() {
  if (!state.uploadedFile) {
    setStatus("Please select a KML or KMZ contour file.", "error", false);
    return;
  }

  clearResults();
  setStatus("Parsing contour survey & extracting topographic basins...", "info", true);
  document.getElementById("btn-run-analysis").disabled = true;

  const formData = new FormData();
  formData.append("file", state.uploadedFile);
  formData.append("area_polygon_json", JSON.stringify(state.polygon || []));

  try {
    const resp = await fetch("/api/analyzeContourWithArea", {
      method: "POST",
      body: formData,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);

    state.result = data;
    state.kmlB64 = null;

    renderResults(data);
    setStatus("Survey analysis complete. Querying ERA5 rainfall reanalysis...", "info", true);

    await fetchAndRenderRainfall(data);
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, "error", false);
  } finally {
    document.getElementById("btn-run-analysis").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// ERA5 Rainfall Fetching
// ---------------------------------------------------------------------------
async function fetchAndRenderRainfall(result) {
  const candidates = result.pondCandidates || [];
  if (!candidates.length) {
    setStatus("No suitable pond depressions detected in this region.", "warning", false);
    return;
  }

  const locations = candidates.map(c => [c.pondCentroid.lat, c.pondCentroid.lon]);
  try {
    const resp = await fetch("/api/fetchRainfall", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ locations, years: 5 }),
    });
    const series = await resp.json();
    if (!resp.ok) throw new Error(series.error || `HTTP ${resp.status}`);

    state.rainfallSeries = series;
    renderRainfallChart(series[0]);
    renderWaterBalance(result, series[0]);
    setStatus("Analysis complete. Recommendations updated.", "success", false);
    setTimeout(clearStatus, 4000);
  } catch (err) {
    console.warn("Rainfall service warning:", err);
    document.getElementById("rainfall-chart-loader").textContent = `Precipitation data unavailable: ${err.message}`;
    renderWaterBalance(result, null);
    setStatus("Depression assessment complete (Precipitation query timed out).", "info", false);
  }
}

// ---------------------------------------------------------------------------
// Render Results: Map, Metrics, Table, Tabs
// ---------------------------------------------------------------------------
function renderResults(result) {
  renderMapLayers(result);
  renderMetrics(result);
  renderSummaryTable(result);
  renderCandidatesTable(result);
  renderWaterBalance(result, null); // Render physical basin balance immediately

  document.getElementById("results-wrapper").classList.remove("hidden");
  document.getElementById("main-banner").classList.add("hidden");
}

function renderMapLayers(result) {
  clearResultLayers();
  const candidates = result.pondCandidates || [];
  const bounds = [];

  candidates.forEach((c, idx) => {
    const isRecommended = idx === 0 || c.recommended;
    const color = isRecommended ? "#ff4b4b" : CANDIDATE_COLORS[idx % CANDIDATE_COLORS.length];
    const catColor = isRecommended ? "#1c83e1" : color;
    const basinColor = isRecommended ? "#09ab3b" : color;

    // 1. Catchment Drainage Boundary (D8 flow accumulation basin)
    const catGeo = c.catchmentBoundary || c.catchmentBoundaryGeoJSON;
    if (catGeo) {
      try {
        const catLayer = L.geoJSON(catGeo, {
          style: {
            color: catColor,
            weight: isRecommended ? 2.5 : 1.5,
            fillColor: catColor,
            fillOpacity: isRecommended ? 0.15 : 0.08,
            dashArray: "6, 4",
          },
        });
        const catSqM = c.estimatedCatchmentAreaSqM || 0;
        const catKm2 = (catSqM / 1_000_000).toFixed(4);
        catLayer.bindTooltip(
          `🌐 <b>Catchment Drainage Area:</b> ${formatArea(catSqM)} (${catKm2} km²)`,
          { sticky: true }
        );
        catLayer.bindPopup(`
          <div style="font-family:'Source Sans 3',sans-serif;font-size:13px;min-width:210px;">
            <strong style="color:${catColor};font-size:14px;">🌐 Upstream Catchment (${isRecommended ? 'Primary Site' : '#' + (idx + 1)})</strong>
            <hr style="margin:4px 0;border:0;border-top:1px solid #ddd;">
            <b>Total Drainage Area:</b> ${formatArea(catSqM)} (${catKm2} km²)<br>
            <b>Contributing D8 Cells:</b> ${c.contributingCellCount || '—'}<br>
            <b>Runoff Potential (C=0.20):</b> ${formatVolume(c.potentialAnnualRunoffM3 || (catSqM * 1.25 * 0.20))}<br>
            <div style="margin-top:4px;font-size:11px;color:#666;">Topographic area that sheds overland rainfall runoff toward this pond depression.</div>
          </div>
        `);
        if (state.layerGroups && state.layerGroups.catchments) {
          state.layerGroups.catchments.addLayer(catLayer);
        } else {
          catLayer.addTo(map);
        }
        state.resultLayers.push(catLayer);
        const b = catLayer.getBounds();
        if (b && b.isValid()) bounds.push(b);
      } catch (e) {
        console.debug("Catchment GeoJSON parse error", e);
      }
    }

    // 2. Basin Boundary (Depression bed / rim reservoir)
    const basinGeo = c.basinBoundary || c.basinBoundaryGeoJSON || (idx === 0 ? result.basinBoundary : null);
    if (basinGeo) {
      try {
        const basinLayer = L.geoJSON(basinGeo, {
          style: {
            color: basinColor,
            weight: isRecommended ? 3 : 2,
            fillColor: basinColor,
            fillOpacity: isRecommended ? 0.35 : 0.20,
          },
        });
        const surfaceM2 = c.basinAreaSqM || c.basinSurfaceAreaM2 || 0;
        const volM3 = c.estimatedVolumeM3 || 0;
        basinLayer.bindTooltip(
          `💧 <b>Pond Basin Surface:</b> ${formatArea(surfaceM2)} | <b>Storage:</b> ${formatVolume(volM3)}`,
          { sticky: true }
        );
        basinLayer.bindPopup(`
          <div style="font-family:'Source Sans 3',sans-serif;font-size:13px;min-width:210px;">
            <strong style="color:${basinColor};font-size:14px;">💧 Pond Basin Depression (${isRecommended ? 'Primary Site' : '#' + (idx + 1)})</strong>
            <hr style="margin:4px 0;border:0;border-top:1px solid #ddd;">
            <b>Basin Surface Area:</b> ${formatArea(surfaceM2)}<br>
            <b>Storage Capacity:</b> ${formatVolume(volM3)}<br>
            <b>Maximum Basin Depth:</b> ${(c.basinDepthM || 0).toFixed(2)} m<br>
            <b>Pond Bed Elevation:</b> ${(c.pondElevation || 0).toFixed(1)} m<br>
            <b>Siting Status:</b> ${isRecommended ? '⭐ Primary Recommended Siting' : 'Alternative Site'}
          </div>
        `);
        if (state.layerGroups && state.layerGroups.basins) {
          state.layerGroups.basins.addLayer(basinLayer);
        } else {
          basinLayer.addTo(map);
        }
        state.resultLayers.push(basinLayer);
        const b = basinLayer.getBounds();
        if (b && b.isValid()) bounds.push(b);
      } catch (e) {
        console.debug("Basin GeoJSON parse error", e);
      }
    }

    // 3. Centroid marker pin
    if (c.pondCentroid && typeof c.pondCentroid.lat === "number" && typeof c.pondCentroid.lon === "number") {
      const { lat, lon } = c.pondCentroid;
      const marker = L.circleMarker([lat, lon], {
        radius: isRecommended ? 10 : 7,
        color: "#ffffff",
        weight: 2.5,
        fillColor: isRecommended ? "#ff4b4b" : color,
        fillOpacity: 0.95,
      });

      const title = isRecommended ? "⭐ Primary Recommended Site" : `Alternative Candidate #${idx + 1}`;
      const volStr = formatVolume(c.estimatedVolumeM3 || 0);
      const catStr = formatArea(c.estimatedCatchmentAreaSqM || 0);
      const basinStr = formatArea(c.basinAreaSqM || c.basinSurfaceAreaM2 || 0);
      const depthStr = (c.basinDepthM || 0).toFixed(2);

      marker.bindPopup(`
        <div style="font-family: 'Source Sans 3', sans-serif; font-size: 13px; min-width: 210px;">
          <strong style="color: ${isRecommended ? '#ff4b4b' : color}; font-size: 14px;">${title}</strong><br>
          <hr style="margin: 4px 0; border: 0; border-top: 1px solid #ddd;">
          <b>Coordinates:</b> ${lat.toFixed(5)}°N, ${lon.toFixed(5)}°E<br>
          <b>Pond Bed Elevation:</b> ${(c.pondElevation || 0).toFixed(1)} m<br>
          <b>Max Basin Depth:</b> ${depthStr} m<br>
          <b>Basin Surface Area:</b> ${basinStr}<br>
          <b>Storage Capacity:</b> ${volStr}<br>
          <b>Upstream Catchment:</b> ${catStr}<br>
          <b>Confidence Score:</b> ${((c.confidenceScore || c.score || 0) * 100).toFixed(0)}%
        </div>
      `);

      if (isRecommended) marker.openPopup();
      if (state.layerGroups && state.layerGroups.markers) {
        state.layerGroups.markers.addLayer(marker);
      } else {
        marker.addTo(map);
      }
      state.resultLayers.push(marker);
      bounds.push(L.latLngBounds([[lat, lon], [lat, lon]]));
    }
  });

  if (bounds.length > 0) {
    try {
      let combined = bounds[0];
      for (let i = 1; i < bounds.length; i++) {
        combined = combined.extend(bounds[i]);
      }
      if (combined.isValid()) {
        map.fitBounds(combined, { padding: [40, 40], maxZoom: 17 });
      }
    } catch (e) {
      console.debug("Fit bounds error", e);
    }
  }
}

function renderMetrics(result) {
  const rec = (result.pondCandidates || [])[0];
  if (!rec) return;

  document.getElementById("m-elevation").textContent = `${(rec.pondElevation || 0).toFixed(1)} m`;
  document.getElementById("m-depth").textContent = `${(rec.basinDepthM || 0).toFixed(2)} m`;
  document.getElementById("m-storage").textContent = formatVolume(rec.estimatedVolumeM3 || 0);
  document.getElementById("m-catchment").textContent = formatArea(rec.estimatedCatchmentAreaSqM || 0);
  document.getElementById("m-confidence").textContent = `${((rec.confidenceScore || rec.score || 0) * 100).toFixed(0)}%`;
}

function renderSummaryTable(result) {
  const rec = (result.pondCandidates || [])[0];
  if (!rec) return;

  document.getElementById("s-coords").textContent = `${rec.pondCentroid.lat.toFixed(5)}°N, ${rec.pondCentroid.lon.toFixed(5)}°E`;
  document.getElementById("s-surface").textContent = formatArea(rec.basinAreaSqM || rec.basinSurfaceAreaM2 || rec.estimatedCatchmentAreaSqM || 0);
  document.getElementById("s-compactness").textContent = rec.compactnessScore ? rec.compactnessScore.toFixed(2) : "0.78 (Well-rounded)";
}

function renderCandidatesTable(result) {
  const tbody = document.getElementById("candidates-tbody");
  const candidates = result.pondCandidates || [];

  if (!candidates.length) {
    tbody.innerHTML = `<tr><td colspan="7">No candidate depressions detected.</td></tr>`;
    return;
  }

  tbody.innerHTML = candidates.map((c, idx) => {
    const isRec = idx === 0 || c.recommended;
    const rankBadge = isRec
      ? `<span style="color: #ff4b4b; font-weight: 700;">#1 (Recommended)</span>`
      : `#${idx + 1}`;
    return `
      <tr>
        <td>${rankBadge}</td>
        <td style="font-family: monospace;">${c.pondCentroid.lat.toFixed(4)}, ${c.pondCentroid.lon.toFixed(4)}</td>
        <td>${(c.pondElevation || 0).toFixed(1)}</td>
        <td>${(c.basinDepthM || 0).toFixed(2)}</td>
        <td>${formatVolume(c.estimatedVolumeM3 || 0)}</td>
        <td>${formatArea(c.estimatedCatchmentAreaSqM || 0)}</td>
        <td><strong>${((c.confidenceScore || c.score || 0) * 100).toFixed(0)}%</strong></td>
      </tr>
    `;
  }).join("");
}

function renderWaterBalance(result, series) {
  const rec = (result.pondCandidates || [])[0];
  if (!rec) return;

  const annualM = series ? (series.meanAnnualPrecipitationM || 0) : 1.25;
  const annualMm = annualM * 1000;
  const catchSqM = rec.estimatedCatchmentAreaSqM || 0;
  const catchKm2 = (catchSqM / 1_000_000).toFixed(4);
  const storageM3 = Math.max(0, rec.estimatedVolumeM3 || 0);
  const grossRainfallM3 = annualM * catchSqM;
  const potentialRunoffM3 = annualM * catchSqM * RUNOFF_COEFF;
  const fillRatio = storageM3 > 0 ? (potentialRunoffM3 / storageM3) * 100 : 100;
  const surplusM3 = Math.max(0, potentialRunoffM3 - storageM3);
  const fillableM3 = Math.min(potentialRunoffM3, storageM3);
  const surfaceM2 = rec.basinSurfaceAreaM2 || Math.round(catchSqM * 0.15);

  const box = document.getElementById("water-balance-box");
  box.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
      <div style="background: rgba(255, 75, 75, 0.08); border-left: 3px solid #ff4b4b; padding: 8px 12px; border-radius: 4px;">
        <strong style="color: #ff4b4b;">Recommended Pond Site:</strong>
        ${rec.pondCentroid.lat.toFixed(5)}°N, ${rec.pondCentroid.lon.toFixed(5)}°E &nbsp;•&nbsp;
        <strong>Elevation:</strong> ${(rec.pondElevation || 0).toFixed(1)} m &nbsp;•&nbsp;
        <strong>Max Depth:</strong> ${(rec.basinDepthM || 0).toFixed(1)} m
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
        <div style="background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.05);">
          <div style="color: var(--st-text-muted); font-size: 11px; font-weight: 600;">POND STORAGE CAPACITY</div>
          <div style="font-size: 16px; font-weight: 700; color: #fafafa;">${formatVolume(storageM3)}</div>
          <div style="font-size: 11px; color: #a3a8b8;">Surface Area: ${formatArea(surfaceM2)}</div>
        </div>

        <div style="background: rgba(255,255,255,0.03); padding: 8px 10px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.05);">
          <div style="color: var(--st-text-muted); font-size: 11px; font-weight: 600;">UPSTREAM CATCHMENT</div>
          <div style="font-size: 16px; font-weight: 700; color: #fafafa;">${formatArea(catchSqM)}</div>
          <div style="font-size: 11px; color: #a3a8b8;">Total drainage: ${catchKm2} km²</div>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 6px; padding: 4px 0; border-top: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06);">
        <div>• <strong>Mean Annual Precipitation:</strong> <strong>${annualMm.toFixed(0)} mm/year</strong> (ERA5 5-Yr Avg)</div>
        <div>• <strong>Gross Precipitation on Catchment:</strong> ${formatVolume(grossRainfallM3)} per year</div>
        <div>• <strong>Catchment Potential Runoff:</strong> <strong>${formatVolume(potentialRunoffM3)}</strong> (Rational Method C = ${RUNOFF_COEFF.toFixed(2)})</div>
        <div>• <strong>Basin Inflow Fill Ratio:</strong> <strong style="font-size: 14px; color: ${fillRatio >= 100 ? '#81c784' : '#ffa421'};">${fillRatio.toFixed(0)}%</strong> of pond storage capacity</div>
        <div>• <strong>Fillable Storage:</strong> <strong>${formatVolume(fillableM3)}</strong> with <strong>${formatVolume(surplusM3)}</strong> surplus overflow / infiltration</div>
      </div>

      <div style="padding: 8px 12px; border-radius: 4px; background: ${fillRatio >= 100 ? 'rgba(76, 175, 80, 0.12)' : 'rgba(255, 164, 33, 0.12)'}; border: 1px solid ${fillRatio >= 100 ? 'rgba(76, 175, 80, 0.3)' : 'rgba(255, 164, 33, 0.3)'}; color: ${fillRatio >= 100 ? '#a5d6a7' : '#ffcc80'};">
        ${fillRatio >= 100
          ? `✓ <strong>Optimal Siting:</strong> Annual runoff (${formatVolume(potentialRunoffM3)}) exceeds pond capacity by ${fillRatio.toFixed(0)}%. The pond will reliably achieve full volume retention during monsoon and sustain dry-season livestock / micro-irrigation.`
          : `⚠️ <strong>Sub-Optimal Runoff:</strong> Catchment runoff (${formatVolume(potentialRunoffM3)}) provides only ${fillRatio.toFixed(0)}% of pond capacity under average monsoon conditions. Recommended: contour bunding or inlet channel diversion.`
        }
      </div>
    </div>
  `;
}

function renderRainfallChart(series) {
  const loader = document.getElementById("rainfall-chart-loader");
  if (loader) loader.classList.add("hidden");

  if (!series) return;
  // Robust check for property name across backend versions
  const monthlyVals = series.meanMonthlyPrecipitationM || series.monthlyMeanPrecipitationM;
  if (!monthlyVals || !monthlyVals.length) return;

  const annualMm = ((series.meanAnnualPrecipitationM || 0) * 1000).toFixed(0);
  const badge = document.getElementById("rf-annual-badge");
  if (badge) badge.textContent = `Annual: ${annualMm} mm (5-Yr Avg)`;

  const monthlyMm = monthlyVals.map(v => +(v * 1000).toFixed(1));
  const canvas = document.getElementById("rainfall-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  if (state.rainfallChart) {
    state.rainfallChart.destroy();
    state.rainfallChart = null;
  }

  state.rainfallChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: MONTH_LABELS,
      datasets: [{
        label: "5-Year Monthly Average Rainfall (mm)",
        data: monthlyMm,
        backgroundColor: monthlyMm.map(v => (v > 150 ? "#ff4b4b" : v > 40 ? "#ffa421" : "#1c83e1")),
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: c => ` Rainfall: ${c.parsed.y} mm`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#808495", font: { family: "'Source Sans 3', sans-serif" } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
        y: {
          ticks: { color: "#808495", font: { family: "'Source Sans 3', sans-serif" } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          title: { display: true, text: "Precipitation (mm)", color: "#808495" },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Downloads & Exports
// ---------------------------------------------------------------------------
function downloadKML() {
  if (!state.kmlB64) {
    setStatus("No KML vector layer available for download.", "warning", false);
    return;
  }
  const binary = atob(state.kmlB64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: "application/vnd.google-earth.kml+xml" });
  triggerDownload(blob, "pond_catchment_analysis.kml");
}

function downloadJSON() {
  if (!state.result) return;
  const payload = {
    timestamp: new Date().toISOString(),
    result: state.result,
    rainfall: state.rainfallSeries,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  triggerDownload(blob, "pond_catchment_report.json");
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Formatting Helpers (Metres & Kilometres Only - No ha)
// ---------------------------------------------------------------------------
function formatVolume(m3) {
  if (m3 >= 1_000_000) return `${(m3 / 1_000_000).toFixed(2)} Mm³`;
  if (m3 >= 1_000) return `${(m3 / 1_000).toFixed(1)} k m³`;
  return `${Math.round(m3).toLocaleString()} m³`;
}

function formatArea(m2) {
  if (m2 >= 1_000_000) return `${(m2 / 1_000_000).toFixed(3)} km²`;
  return `${Math.round(m2).toLocaleString()} m²`;
}

// ---------------------------------------------------------------------------
// Event Listeners & Bootstrapping
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Village Search & Real-Time Map Reflection
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function initVillageSearch() {
  const input = document.getElementById("village-search-input");
  const spinner = document.getElementById("village-search-spinner");
  const list = document.getElementById("village-suggestions-list");
  if (!input || !list) return;

  let debounceTimer = null;
  let activeSearchController = null;

  async function fetchSuggestions(query) {
    if (activeSearchController) {
      activeSearchController.abort();
    }
    if (!query || query.length < 2) {
      list.classList.add("hidden");
      list.innerHTML = "";
      state.currentSuggestions = [];
      state.selectedSuggestionIndex = -1;
      return;
    }

    activeSearchController = new AbortController();
    if (spinner) spinner.classList.remove("hidden");
    try {
      const res = await fetch(`/api/searchVillage?q=${encodeURIComponent(query)}`, {
        signal: activeSearchController.signal,
      });
      if (!res.ok) throw new Error("Search failed");
      const suggestions = await res.json();
      state.currentSuggestions = suggestions || [];
      state.selectedSuggestionIndex = -1;

      if (suggestions && suggestions.length > 0) {
        list.innerHTML = suggestions.map((item, idx) => `
          <li class="village-suggestion-item" data-idx="${idx}">
            <div class="village-item-name">📍 ${escapeHtml(item.name)}</div>
            <div class="village-item-desc">${escapeHtml(item.displayName)}</div>
          </li>
        `).join("");
        list.classList.remove("hidden");
      } else {
        list.innerHTML = `<li class="village-suggestion-item" style="cursor:default;color:#808495;">No villages found for "${escapeHtml(query)}"</li>`;
        list.classList.remove("hidden");
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      console.debug("Village search error:", err);
      list.classList.add("hidden");
    } finally {
      if (spinner) spinner.classList.add("hidden");
    }
  }

  input.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const query = input.value.trim();
    debounceTimer = setTimeout(() => {
      fetchSuggestions(query);
    }, 200);
  });

  input.addEventListener("keydown", (e) => {
    const items = list.querySelectorAll(".village-suggestion-item");
    if (items.length === 0 || list.classList.contains("hidden")) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      state.selectedSuggestionIndex = (state.selectedSuggestionIndex + 1) % items.length;
      updateSuggestionHighlight(items);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      state.selectedSuggestionIndex = (state.selectedSuggestionIndex - 1 + items.length) % items.length;
      updateSuggestionHighlight(items);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (state.selectedSuggestionIndex >= 0 && state.currentSuggestions[state.selectedSuggestionIndex]) {
        selectVillage(state.currentSuggestions[state.selectedSuggestionIndex]);
      } else if (state.currentSuggestions.length > 0) {
        selectVillage(state.currentSuggestions[0]);
      }
    } else if (e.key === "Escape") {
      list.classList.add("hidden");
    }
  });

  function updateSuggestionHighlight(items) {
    items.forEach((item, idx) => {
      item.classList.toggle("highlighted", idx === state.selectedSuggestionIndex);
      if (idx === state.selectedSuggestionIndex) {
        item.scrollIntoView({ block: "nearest" });
      }
    });
  }

  list.addEventListener("click", (e) => {
    const item = e.target.closest(".village-suggestion-item");
    if (!item || item.dataset.idx === undefined) return;
    const idx = parseInt(item.dataset.idx, 10);
    if (state.currentSuggestions[idx]) {
      selectVillage(state.currentSuggestions[idx]);
    }
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".village-search-widget")) {
      list.classList.add("hidden");
    }
  });
}

function selectVillage(village) {
  clearResults();
  const input = document.getElementById("village-search-input");
  const list = document.getElementById("village-suggestions-list");
  if (input) input.value = village.name;
  if (list) list.classList.add("hidden");

  // Smooth real-time flyTo on Leaflet map (snappy 0.8s)
  map.flyTo([village.lat, village.lon], 15, {
    animate: true,
    duration: 0.8,
  });

  // Highlight village location with Leaflet marker
  if (state.villageMarker) {
    map.removeLayer(state.villageMarker);
  }

  const villageIcon = L.divIcon({
    className: "village-leaflet-pin",
    html: `<div style="background:#ff4b4b;color:#fff;padding:4px 8px;border-radius:12px;font-size:12px;font-weight:700;box-shadow:0 2px 8px rgba(0,0,0,0.5);display:flex;align-items:center;gap:4px;white-space:nowrap;border:2px solid #fff;">📍 ${escapeHtml(village.name)}</div>`,
    iconSize: [110, 30],
    iconAnchor: [55, 15],
  });

  state.villageMarker = L.marker([village.lat, village.lon], { icon: villageIcon })
    .addTo(map)
    .bindPopup(`
      <div style="font-family:'Source Sans 3',sans-serif;color:#0e1117;">
        <h4 style="margin:0 0 4px 0;font-size:15px;color:#ff4b4b;">📍 ${escapeHtml(village.name)}</h4>
        <p style="margin:0 0 6px 0;font-size:12px;color:#555;">${escapeHtml(village.displayName)}</p>
        <div style="font-size:11px;background:#f0f2f6;padding:4px 8px;border-radius:4px;margin-bottom:6px;">
          Coordinates: <strong>${village.lat.toFixed(4)}, ${village.lon.toFixed(4)}</strong>
        </div>
        <p style="margin:0;font-size:12px;font-weight:600;color:#09ab3b;">
          ✏️ Ready! Outline the catchment area with the polygon tool ⬡ (top-right)!
        </p>
      </div>
    `, { maxWidth: 280 })
    .openPopup();
}

// ---------------------------------------------------------------------------
// Event Listeners & Bootstrapping
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadConfig();
  initVillageSearch();

  // Mode radio clicks
  document.querySelectorAll(".st-radio-option").forEach(opt => {
    opt.addEventListener("click", () => switchMode(opt.dataset.mode));
  });

  // Action buttons
  document.getElementById("btn-run-analysis").addEventListener("click", runAnalysis);
  document.getElementById("btn-clear-selection").addEventListener("click", clearPolygon);

  // Tabs switching
  document.querySelectorAll(".st-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".st-tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".st-tab-panel").forEach(p => p.classList.add("hidden"));
      btn.classList.add("active");
      const targetPanel = document.getElementById(btn.dataset.tab);
      if (targetPanel) targetPanel.classList.remove("hidden");
    });
  });

  // Downloads
  document.getElementById("dl-kml-btn-main").addEventListener("click", downloadKML);
  document.getElementById("dl-json-btn-main").addEventListener("click", downloadJSON);

  // File upload input
  const fileInput = document.getElementById("file-input");
  const fileLabel = document.getElementById("file-label");
  const dropZone = document.getElementById("file-drop-zone");

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      state.uploadedFile = file;
      fileLabel.textContent = `📄 ${file.name}`;
      dropZone.classList.add("has-file");
      setAnalyzeEnabled(canAnalyze());
    }
  });

  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      state.uploadedFile = file;
      fileLabel.textContent = `📄 ${file.name}`;
      dropZone.classList.add("has-file");
      setAnalyzeEnabled(canAnalyze());
    }
  });
});
