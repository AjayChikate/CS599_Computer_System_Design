from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import analyze_contours_in_area, load_raw_contours
from backend.config import MAX_UPLOAD_BYTES
from backend.geometry import normalize_lonlat_ring
from backend.opentopography import MAX_AREA_KM2, analyze_dem_area
from backend.rainfall import (
    RUNOFF_COEFFICIENT,
    fetch_historical_rainfall_for_points,
    rank_candidates_by_rainfall,
)

st.set_page_config(page_title="Contour & Catchment", layout="wide")

RECOMMENDED_COLOR = "#2563EB"
ALTERNATIVE_COLORS = ["#B9622C", "#7B5EA7", "#C9A227", "#3F7D5C", "#A3352B", "#4F8FB0", "#8A6D3B"]


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def cached_raw_contours(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    return load_raw_contours(file_bytes, filename)


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def cached_area_analysis(file_bytes: bytes, filename: str, area_json: str) -> dict[str, Any]:
    area = [tuple(point) for point in json.loads(area_json)]
    return analyze_contours_in_area(cached_raw_contours(file_bytes, filename), area)


@st.cache_data(ttl=86400, max_entries=32, show_spinner=False)
def cached_candidate_rainfall(coordinates_json: str) -> list[dict[str, Any]]:
    locations = [tuple(point) for point in json.loads(coordinates_json)]
    return fetch_historical_rainfall_for_points(locations, years=10)


def apply_rainfall_ranking(
    result: dict[str, Any], rainfall_series: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates = rank_candidates_by_rainfall(result["pondCandidates"], rainfall_series)
    ranked_result = {**result, "pondCandidates": candidates, "rainfallAdjustedRecommendation": True}
    ranked_result["alternativeCandidates"] = [
        {key: value for key, value in candidate.items() if key not in ("rank", "recommended")}
        for candidate in candidates[1:]
    ]
    best = candidates[0]
    for field in (
        "pondElevation",
        "pondCentroid",
        "basinAreaSqM",
        "estimatedCatchmentAreaSqM",
        "basinDepthM",
        "estimatedVolumeM3",
        "basinBoundary",
        "fallbackRecommendation",
        "score",
    ):
        if field in best:
            ranked_result[field] = best[field]
    ranked_result["meanAnnualPrecipitationM"] = best["meanAnnualPrecipitationM"]
    ranked_result["potentialAnnualRunoffM3"] = best["potentialAnnualRunoffM3"]
    ranked_result["estimatedAnnualFillableWaterM3"] = best["estimatedAnnualFillableWaterM3"]
    return ranked_result


def complete_rainfall_analysis(
    result: dict[str, Any], state_key: str
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    coordinates = [
        [candidate["pondCentroid"]["lat"], candidate["pondCentroid"]["lon"]]
        for candidate in result["pondCandidates"]
    ]
    coordinates_json = json.dumps(coordinates, separators=(",", ":"))
    signature = hashlib.sha256(coordinates_json.encode("utf-8")).hexdigest()
    rainfall_state_key = f"{state_key}_rainfall_state"
    rainfall_state = st.session_state.get(rainfall_state_key, {})

    if rainfall_state.get("signature") == signature:
        if "error" in rainfall_state:
            return result, None, rainfall_state["error"]
        rainfall_series = rainfall_state["series"]
    else:
        try:
            with st.spinner("Completing analysis with historical rainfall for all candidates..."):
                rainfall_series = cached_candidate_rainfall(coordinates_json)
            st.session_state[rainfall_state_key] = {"signature": signature, "series": rainfall_series}
        except ValueError as exc:
            message = str(exc)
            st.session_state[rainfall_state_key] = {"signature": signature, "error": message}
            return result, None, message

    ranked_result = apply_rainfall_ranking(result, rainfall_series)
    best = ranked_result["pondCandidates"][0]
    rainfall_estimate = {
        "source": best["rainfallSource"],
        "model": best["rainfallModel"],
        "gridResolutionKm": best["rainfallGridResolutionKm"],
        "periodStart": best["rainfallPeriodStart"],
        "periodEnd": best["rainfallPeriodEnd"],
        "runoffCoefficient": RUNOFF_COEFFICIENT,
        "recommendationRule": "rank by minimum of annual runoff potential and pond storage capacity",
        "candidateEstimates": [
            {
                "rank": candidate["rank"],
                "location": candidate["pondCentroid"],
                "meanAnnualPrecipitationM": candidate["meanAnnualPrecipitationM"],
                "yearsIncluded": candidate["rainfallYearsIncluded"],
                "potentialAnnualRunoffM3": candidate["potentialAnnualRunoffM3"],
                "storageCapacityM3": candidate["estimatedVolumeM3"],
                "estimatedAnnualFillableWaterM3": candidate["estimatedAnnualFillableWaterM3"],
            }
            for candidate in ranked_result["pondCandidates"]
        ],
    }
    return ranked_result, rainfall_estimate, None


def render_rainfall_panel(
    result: dict[str, Any], rainfall_estimate: dict[str, Any] | None, error: str | None = None
) -> None:
    st.subheader("Rainfall-based recommendation")
    if error:
        st.warning(f"Rainfall data is unavailable; showing the terrain-based recommendation. {error}")
        return
    if not rainfall_estimate or not result.get("rainfallAdjustedRecommendation"):
        return

    candidate = result["pondCandidates"][0]
    first_year = candidate["rainfallPeriodStart"][:4]
    last_year = candidate["rainfallPeriodEnd"][:4]
    metrics = st.columns(3)
    metrics[0].metric("Potential annual runoff", f'{candidate["potentialAnnualRunoffM3"]:,.0f} m³')
    metrics[1].metric("Annual fillable water", f'{candidate["estimatedAnnualFillableWaterM3"]:,.0f} m³')
    metrics[2].metric("Pond storage capacity", f'{candidate["estimatedVolumeM3"]:,.0f} m³')

    monthly_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.bar_chart(
        {"Month": monthly_names, "Mean precipitation (m)": candidate["meanMonthlyPrecipitationM"]},
        x="Month",
        y="Mean precipitation (m)",
        y_label="m",
    )
    grid_cell = candidate["rainfallGridCell"]
    st.caption(
        f"Ranked by min(annual runoff potential, pond storage capacity). "
        f"ERA5, {first_year}–{last_year}, {len(candidate['rainfallYearsIncluded'])} complete years. "
        f"Grid cell used: {grid_cell['latitude']:.3f}, {grid_cell['longitude']:.3f}. "
        f"Fixed runoff coefficient: {RUNOFF_COEFFICIENT:.2f}. "
        "Annual fillable water is an estimate before evaporation and seepage."
    )
def candidate_color(candidate: dict[str, Any]) -> str:
    if candidate["recommended"]:
        return RECOMMENDED_COLOR
    return ALTERNATIVE_COLORS[(candidate["rank"] - 2) % len(ALTERNATIVE_COLORS)]


def coordinate_bounds(points: list[tuple[float, float]]) -> list[list[float]]:
    longitudes = [point[0] for point in points]
    latitudes = [point[1] for point in points]
    return [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]


def contour_bounds(contours: list[dict[str, Any]]) -> list[list[float]]:
    min_lon = min(point[0] for contour in contours for point in contour["points"])
    max_lon = max(point[0] for contour in contours for point in contour["points"])
    min_lat = min(point[1] for contour in contours for point in contour["points"])
    max_lat = max(point[1] for contour in contours for point in contour["points"])
    return [[min_lat, min_lon], [max_lat, max_lon]]


def build_map(
    raw_contours: list[dict[str, Any]],
    result: dict[str, Any] | None = None,
    selected_area: list[tuple[float, float]] | None = None,
) -> folium.Map:
    bounds = coordinate_bounds(selected_area) if selected_area else contour_bounds(raw_contours) if raw_contours else None
    center = (
        [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
        if bounds
        else [21.2517, 81.2970]
    )
    fmap = folium.Map(
        location=center,
        zoom_start=15 if raw_contours else 7 if not selected_area else 15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    fit_points = list(selected_area) if selected_area else None
    if selected_area:
        folium.Polygon(
            [(lat, lon) for lon, lat in selected_area],
            color="#111827",
            weight=2,
            fill=True,
            fill_color="#F59E0B",
            fill_opacity=0.12,
            tooltip="Selected land area",
        ).add_to(folium.FeatureGroup(name="Selected land area", show=True).add_to(fmap))

    if result:
        pond_group = folium.FeatureGroup(name="Pond and catchment results", show=True)
        candidates = result["pondCandidates"]
        ordered_candidates = [candidate for candidate in candidates if not candidate["recommended"]]
        ordered_candidates += [candidate for candidate in candidates if candidate["recommended"]]

        for candidate in ordered_candidates:
            is_best = candidate["recommended"]
            is_fallback = candidate.get("fallbackRecommendation", False)
            color = candidate_color(candidate)
            basin_boundary = candidate.get("basinBoundary")
            catchment_boundary = candidate.get("catchmentBoundary")
            boundary_latlon = []
            if basin_boundary and basin_boundary.get("type") == "Polygon":
                boundary = basin_boundary["coordinates"][0]
                boundary_latlon = [(lat, lon) for lon, lat in boundary]
                if fit_points is not None and not catchment_boundary:
                    fit_points.extend(boundary)
            label = (
                "Lowest DEM point (no depression found)"
                if is_fallback
                else "Recommended pond" if is_best else f'Alternative #{candidate["rank"]}'
            )
            catchment = candidate["estimatedCatchmentAreaSqM"]
            volume = candidate["estimatedVolumeM3"]
            if catchment_boundary:
                folium.GeoJson(
                    catchment_boundary,
                    style_function=lambda _feature, color=color, is_best=is_best, candidate=candidate: {
                        "color": color,
                        "weight": 2 if is_best else 1,
                        "fillColor": color,
                        "fillOpacity": 0.12 if is_best else 0.06,
                        "dashArray": "5, 5" if candidate.get("catchmentIsTruncated") else None,
                    },
                    tooltip=(
                        f"D8 catchment | {catchment:,.0f} m²"
                        + (" | clipped at DEM edge" if candidate.get("catchmentIsTruncated") else "")
                    ),
                ).add_to(pond_group)
            elif boundary_latlon:
                folium.Polygon(
                    boundary_latlon,
                    color="#B45309" if is_fallback else color,
                    weight=2 if is_fallback else 3 if is_best else 1.5,
                    fill=True,
                    fill_color="#F59E0B" if is_fallback else color,
                    fill_opacity=0.12 if is_fallback else 0.28 if is_best else 0.14,
                    tooltip=(
                        f"Fallback search area | {catchment:,.0f} m² | no basin found"
                        if is_fallback
                        else f"{label} catchment | {catchment:,.0f} m² | {volume:,.0f} m³ storage"
                    ),
                ).add_to(pond_group)

            if basin_boundary and catchment_boundary:
                folium.GeoJson(
                    basin_boundary,
                    style_function=lambda _feature, color=color: {
                        "color": color,
                        "weight": 2,
                        "fillColor": color,
                        "fillOpacity": 0.25,
                    },
                    tooltip=f"Depression basin | estimated storage {volume:,.0f} m³",
                ).add_to(pond_group)

            latitude = candidate["pondCentroid"]["lat"]
            longitude = candidate["pondCentroid"]["lon"]
            popup_html = (
                f"<b>{label}</b><br>"
                f"Suggested location: {latitude:.6f}, {longitude:.6f}<br>"
                f"{'D8 catchment area' if catchment_boundary else 'Search area proxy' if is_fallback else 'Catchment area'}: {catchment:,.0f} m²<br>"
                f"Catchment touches DEM edge: {candidate.get('catchmentIsTruncated', False)}<br>"
                f"Expected storage volume: {volume:,.0f} m³<br>"
                f"Pond elevation: {candidate['pondElevation']:.1f} m"
            )
            folium.CircleMarker(
                location=(latitude, longitude),
                radius=9 if is_best else 6,
                color="white",
                weight=2,
                fill=True,
                fill_color="#B45309" if is_fallback else color,
                fill_opacity=1.0,
                tooltip=label,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(pond_group)
        pond_group.add_to(fmap)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {"allowIntersection": False, "showArea": True},
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    if fit_points or bounds:
        fmap.fit_bounds(coordinate_bounds(fit_points) if fit_points else bounds, padding=(24, 24))
    return fmap


def render_summary(result: dict[str, Any]) -> None:
    terrain = result["terrainSummary"]
    columns = st.columns(4)
    if terrain.get("flowRouting"):
        columns[0].metric("Flow routing", terrain["flowRouting"])
    else:
        columns[0].metric("Contour interval", f'{result["contourInterval"]:g} m')
    columns[1].metric(
        "Elevation range",
        f'{terrain["minElevation"]:g}–{terrain["maxElevation"]:g} m',
    )
    columns[2].metric("Basin candidates found", terrain["basinCandidateCount"])
    columns[3].metric("River-like loops filtered", result["riverAvoidance"]["filteredRiverLikeLoopCount"])


def pond_table_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in result["pondCandidates"]:
        row = {
            "Rank": candidate["rank"],
            "Site": (
                "Fallback: lowest DEM point"
                if candidate.get("fallbackRecommendation")
                else "Recommended" if candidate["recommended"] else f'Alternative #{candidate["rank"]}'
            ),
            "Elevation (m)": candidate["pondElevation"],
            "Basin area (m²)": candidate["basinAreaSqM"],
            "Depth (m)": candidate["basinDepthM"],
            "Expected volume (m³)": candidate["estimatedVolumeM3"],
            "Catchment area (m²)": candidate["estimatedCatchmentAreaSqM"],
            "Score": candidate["score"],
            "Latitude": candidate["pondCentroid"]["lat"],
            "Longitude": candidate["pondCentroid"]["lon"],
        }
        if "potentialAnnualRunoffM3" in candidate:
            row["Potential annual runoff (m³)"] = candidate["potentialAnnualRunoffM3"]
            row["Annual fillable water (m³)"] = candidate["estimatedAnnualFillableWaterM3"]
        rows.append(row)
    return rows


st.title("Contour & Catchment")
st.caption("Find pond locations and estimate storage from contour surveys or elevation data.")
st.session_state.pop("opentopo_api_key", None)
analysis_mode = st.segmented_control(
    "Analysis source",
    options=["Upload contours", "Select map area"],
    default="Upload contours",
    selection_mode="single",
    width="stretch",
    key="analysis_workflow",
)

if analysis_mode == "Select map area":
    st.caption("Select a region on the map and analyze it with OpenTopography elevation data.")
    selected_area = st.session_state.get("opentopo_selected_area")
    result = st.session_state.get("opentopo_result")
    generated_kml = st.session_state.get("opentopo_generated_kml")
    opentopo_contours = st.session_state.get("opentopo_contours", [])

    with st.sidebar:
        st.subheader("OpenTopography")
        dataset = st.selectbox("Elevation dataset", ["AW3D30", "SRTMGL1", "COP30"], index=0)
        st.caption(
            f"Maximum selected bounding area: {MAX_AREA_KM2:g} km². "
            "The server uses API_Key from .env; OpenTopography daily limits apply."
        )

    source_signature = f"OpenTopography:{dataset}"

    if st.session_state.get("opentopo_result_source_signature") not in (None, source_signature):
        st.session_state.pop("opentopo_result", None)
        st.session_state.pop("opentopo_generated_kml", None)
        st.session_state.pop("opentopo_contours", None)
        st.session_state.pop("opentopo_rainfall_state", None)
        result = None
        generated_kml = None
        opentopo_contours = []

    rainfall_estimate = None
    rainfall_error = None
    if result:
        result, rainfall_estimate, rainfall_error = complete_rainfall_analysis(result, "opentopo")

    st.subheader("Select an area")
    st.caption("Draw a polygon. Elevation data is fetched and contours are generated only after you click the button below.")
    map_key = f"opentopo_map_{st.session_state.get('opentopo_map_reset', 0)}"
    map_data = st_folium(
        build_map(opentopo_contours, result, selected_area),
        use_container_width=True,
        height=650,
        key=map_key,
        returned_objects=["last_active_drawing"],
    )
    drawing = (map_data or {}).get("last_active_drawing")
    if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
        ring = drawing["geometry"].get("coordinates", [[]])[0]
        drawn_area = normalize_lonlat_ring([(point[0], point[1]) for point in ring])
        current_area = normalize_lonlat_ring(selected_area) if selected_area else []
        if len(drawn_area) >= 4 and drawn_area != current_area:
            st.session_state["opentopo_selected_area"] = drawn_area
            st.session_state.pop("opentopo_result", None)
            st.session_state.pop("opentopo_generated_kml", None)
            st.session_state.pop("opentopo_contours", None)
            st.session_state.pop("opentopo_rainfall_state", None)
            st.rerun()

    if selected_area and st.button("Clear selected area"):
        st.session_state.pop("opentopo_selected_area", None)
        st.session_state.pop("opentopo_result", None)
        st.session_state.pop("opentopo_generated_kml", None)
        st.session_state.pop("opentopo_contours", None)
        st.session_state.pop("opentopo_rainfall_state", None)
        st.session_state["opentopo_map_reset"] = st.session_state.get("opentopo_map_reset", 0) + 1
        st.rerun()

    if not selected_area:
        st.info("Draw a polygon on the map to select a region.")
    elif st.button("Generate contours and analyze", type="primary"):
        try:
            with st.spinner("Fetching global elevation data, generating contours, and analyzing the selected area..."):
                kml_bytes, result = analyze_dem_area(selected_area, dataset)
            generated_contours = load_raw_contours(kml_bytes, "generated_contours.kml")
            st.session_state["opentopo_generated_kml"] = kml_bytes
            st.session_state["opentopo_contours"] = generated_contours
            st.session_state["opentopo_result"] = result
            st.session_state["opentopo_result_source_signature"] = source_signature
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The terrain request could not be processed. Try a smaller area or another dataset.")

    if result:
        st.success("DEM analysis complete.")
        recommended = result["pondCandidates"][0]
        location = recommended["pondCentroid"]
        if result.get("fallbackRecommendation"):
            st.warning(
                "No depression met the DEM depth and size criteria. The marker is the lowest DEM cell in the selected area; "
                "storage is 0 m³."
            )
        summary_columns = st.columns(3)
        summary_columns[0].metric("Suggested pond location", f'{location["lat"]:.6f}, {location["lon"]:.6f}')
        summary_columns[1].metric(
            "D8 catchment area" if result.get("fallbackRecommendation") else "Catchment area",
            f'{recommended["estimatedCatchmentAreaSqM"]:,.0f} m²',
        )
        summary_columns[2].metric(
            "Storage estimate (no basin found)" if result.get("fallbackRecommendation") else "Estimated storage volume",
            f'{recommended["estimatedVolumeM3"]:,.0f} m³',
        )
        render_summary(result)
        render_rainfall_panel(result, rainfall_estimate, rainfall_error)
        st.dataframe(pond_table_rows(result), width="stretch", hide_index=True)
        st.download_button(
            "Download generated contour KML",
            data=generated_kml,
            file_name="dem_contours.kml",
            mime="application/vnd.google-earth.kml+xml",
        )
        st.download_button(
            "Download analysis JSON",
            data=json.dumps(
                {**result, **({"rainfallEstimate": rainfall_estimate} if rainfall_estimate else {})},
                indent=2,
            ),
            file_name="pond_catchment_analysis.json",
            mime="application/json",
        )
        st.caption("Data: OpenTopography Global Datasets API. 30 m resolution may miss small field-scale depressions.")
    st.stop()

uploaded = st.file_uploader(
    "Contour survey (.kml or .kmz)",
    type=["kml", "kmz"],
    max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
)

if uploaded is None:
    st.info("Upload a KML or KMZ contour survey to begin.")
    st.stop()

with st.spinner("Preparing contour map: reading and parsing survey lines..."):
    file_bytes = uploaded.getvalue()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        st.error("Upload is too large. The maximum supported file size is 25 MB.")
        st.stop()

    upload_fingerprint = hashlib.sha256(file_bytes).hexdigest()[:16]
    if st.session_state.get("active_upload") != upload_fingerprint:
        st.session_state["active_upload"] = upload_fingerprint
        st.session_state["selected_land_area"] = None
        st.session_state["map_reset_counter"] = 0
        st.session_state.pop("upload_rainfall_state", None)

    try:
        raw_contours = cached_raw_contours(file_bytes, uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

selected_area = st.session_state.get("selected_land_area")
if selected_area and st.button("Clear selected area"):
    st.session_state["selected_land_area"] = None
    st.session_state.pop("upload_rainfall_state", None)
    st.session_state["map_reset_counter"] = st.session_state.get("map_reset_counter", 0) + 1
    st.rerun()

result = None
analysis_error = None
if selected_area:
    area_json = json.dumps(selected_area, separators=(",", ":"))
    try:
        with st.spinner("Analyzing contours inside the selected area..."):
            result = cached_area_analysis(file_bytes, uploaded.name, area_json)
    except ValueError as exc:
        analysis_error = str(exc)

rainfall_estimate = None
rainfall_error = None
if result:
    result, rainfall_estimate, rainfall_error = complete_rainfall_analysis(result, "upload")

map_key = f"pond_map_{upload_fingerprint}_{st.session_state.get('map_reset_counter', 0)}"
st.subheader("Map")
st.caption(
    "Draw a polygon to select land. Pond markers and shaded catchments appear after analysis."
    if result is None
    else "The selected land, recommended pond marker, and shaded catchment are shown below."
)
with st.spinner("Rendering contour map..."):
    contour_map = build_map(raw_contours, result, selected_area)

map_data = st_folium(
    contour_map,
    use_container_width=True,
    height=610,
    key=map_key,
    returned_objects=["last_active_drawing"],
)

drawing = (map_data or {}).get("last_active_drawing")
if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
    ring = drawing["geometry"].get("coordinates", [[]])[0]
    drawn_area = normalize_lonlat_ring([(point[0], point[1]) for point in ring])
    current_area = normalize_lonlat_ring(selected_area) if selected_area else []
    if len(drawn_area) >= 4 and drawn_area != current_area:
        st.session_state["selected_land_area"] = drawn_area
        st.session_state.pop("upload_rainfall_state", None)
        st.rerun()

if result is None:
    if analysis_error:
        st.warning(analysis_error)
    else:
        st.info("Draw a polygon on the map to select the land area for analysis.")
    st.stop()

st.success(f'Analysis completed using {result["terrainSummary"]["contourCount"]} selected contour rings.')
recommended = result["pondCandidates"][0]
location = recommended["pondCentroid"]
summary_columns = st.columns(3)
summary_columns[0].metric("Suggested pond location", f'{location["lat"]:.6f}, {location["lon"]:.6f}')
summary_columns[1].metric(
    "Catchment area",
    f'{recommended["estimatedCatchmentAreaSqM"]:,.0f} m²',
)
summary_columns[2].metric("Expected water volume", f'{recommended["estimatedVolumeM3"]:,.0f} m³')

render_summary(result)
render_rainfall_panel(result, rainfall_estimate, rainfall_error)
st.subheader("Suggested ponds")
for candidate in result["pondCandidates"]:
    color = candidate_color(candidate)
    label = "Recommended" if candidate["recommended"] else f'Alternative #{candidate["rank"]}'
    with st.container(border=True):
        st.markdown(
            f'<span style="display:inline-block;width:11px;height:11px;'
            f'border-radius:50%;background:{color};margin-right:7px;"></span>'
            f'<b>{label}</b>',
            unsafe_allow_html=True,
        )
        st.caption(
            f'Location: {candidate["pondCentroid"]["lat"]:.6f}, '
            f'{candidate["pondCentroid"]["lon"]:.6f}'
        )
        first, second = st.columns(2)
        first.metric("Catchment", f'{candidate["estimatedCatchmentAreaSqM"]:,.0f} m²')
        second.metric("Expected volume", f'{candidate["estimatedVolumeM3"]:,.0f} m³')

st.subheader("All suggested ponds")
st.dataframe(pond_table_rows(result), width="stretch", hide_index=True)
with st.expander("Full JSON response"):
    st.json(result)
st.download_button(
    "Download JSON",
    data=json.dumps(
        {**result, **({"rainfallEstimate": rainfall_estimate} if rainfall_estimate else {})},
        indent=2,
    ),
    file_name="pond_catchment_analysis.json",
    mime="application/json",
)