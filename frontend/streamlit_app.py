from __future__ import annotations

import hashlib
import json
import math
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

st.set_page_config(page_title="Contour & Catchment", layout="wide")

RECOMMENDED_COLOR = "#2563EB"
ALTERNATIVE_COLORS = ["#B9622C", "#7B5EA7", "#C9A227", "#3F7D5C", "#A3352B", "#4F8FB0", "#8A6D3B"]
MAX_DISPLAY_POINTS_PER_CONTOUR = 400


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def cached_raw_contours(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    return load_raw_contours(file_bytes, filename)


@st.cache_data(ttl=3600, max_entries=3, show_spinner=False)
def cached_area_analysis(file_bytes: bytes, filename: str, area_json: str) -> dict[str, Any]:
    area = [tuple(point) for point in json.loads(area_json)]
    return analyze_contours_in_area(cached_raw_contours(file_bytes, filename), area)


def candidate_color(candidate: dict[str, Any]) -> str:
    if candidate["recommended"]:
        return RECOMMENDED_COLOR
    return ALTERNATIVE_COLORS[(candidate["rank"] - 2) % len(ALTERNATIVE_COLORS)]


def elevation_color(elevation: float, min_elevation: float, max_elevation: float) -> str:
    t = (elevation - min_elevation) / (max_elevation - min_elevation) if max_elevation > min_elevation else 0.5
    red = round(47 + t * (185 - 47))
    green = round(111 + t * (98 - 111))
    blue = round(94 + t * (44 - 94))
    return f"rgb({red},{green},{blue})"


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
    bounds = coordinate_bounds(selected_area) if selected_area else contour_bounds(raw_contours)
    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True)

    elevations = [contour["elevation"] for contour in raw_contours]
    min_elevation, max_elevation = (min(elevations), max(elevations)) if elevations else (0.0, 1.0)
    contour_group = folium.FeatureGroup(name="Contour lines", show=True)
    for contour in raw_contours:
        points = contour["points"]
        stride = max(1, math.ceil(len(points) / MAX_DISPLAY_POINTS_PER_CONTOUR))
        display_points = points[::stride]
        if display_points[-1] != points[-1]:
            display_points.append(points[-1])
        folium.PolyLine(
            [(lat, lon) for lon, lat in display_points],
            color=elevation_color(contour["elevation"], min_elevation, max_elevation),
            weight=1,
            opacity=0.6,
            smooth_factor=1.5,
            tooltip=f'{contour["elevation"]} m',
        ).add_to(contour_group)
    contour_group.add_to(fmap)

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
            color = candidate_color(candidate)
            boundary = candidate["basinBoundary"]["coordinates"][0]
            boundary_latlon = [(lat, lon) for lon, lat in boundary]
            if fit_points is not None:
                fit_points.extend(boundary)
            label = "Recommended pond" if is_best else f'Alternative #{candidate["rank"]}'
            catchment = candidate["estimatedCatchmentAreaHectares"]
            volume = candidate["estimatedVolumeM3"]
            folium.Polygon(
                boundary_latlon,
                color=color,
                weight=3 if is_best else 1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.28 if is_best else 0.14,
                tooltip=f"{label} catchment | {catchment:.2f} ha | {volume:,.0f} m³ storage",
            ).add_to(pond_group)

            latitude = candidate["pondCentroid"]["lat"]
            longitude = candidate["pondCentroid"]["lon"]
            popup_html = (
                f"<b>{label}</b><br>"
                f"Suggested location: {latitude:.6f}, {longitude:.6f}<br>"
                f"Catchment area: {catchment:.2f} ha<br>"
                f"Expected storage volume: {volume:,.0f} m³<br>"
                f"Pond elevation: {candidate['pondElevation']:.1f} m"
            )
            folium.CircleMarker(
                location=(latitude, longitude),
                radius=9 if is_best else 6,
                color="white",
                weight=2,
                fill=True,
                fill_color=color,
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
    fmap.fit_bounds(coordinate_bounds(fit_points) if fit_points else bounds, padding=(24, 24))
    return fmap


def render_summary(result: dict[str, Any]) -> None:
    terrain = result["terrainSummary"]
    columns = st.columns(4)
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
        rows.append(
            {
                "Rank": candidate["rank"],
                "Site": "Recommended" if candidate["recommended"] else f'Alternative #{candidate["rank"]}',
                "Elevation (m)": candidate["pondElevation"],
                "Basin area (m²)": candidate["basinAreaSqM"],
                "Depth (m)": candidate["basinDepthM"],
                "Expected volume (m³)": candidate["estimatedVolumeM3"],
                "Catchment (ha)": candidate["estimatedCatchmentAreaHectares"],
                "Compactness": candidate["compactnessScore"],
                "Score": candidate["score"],
                "Latitude": candidate["pondCentroid"]["lat"],
                "Longitude": candidate["pondCentroid"]["lon"],
            }
        )
    return rows


st.title("Contour & Catchment")
st.caption("Contour-based pond siting and storage estimates")
uploaded = st.file_uploader(
    "Contour survey (.kml or .kmz)",
    type=["kml", "kmz"],
    max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
)

if uploaded is None:
    st.info("Upload a KML or KMZ contour survey to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
if len(file_bytes) > MAX_UPLOAD_BYTES:
    st.error("Upload is too large. The maximum supported file size is 25 MB.")
    st.stop()

upload_fingerprint = hashlib.sha256(file_bytes).hexdigest()[:16]
if st.session_state.get("active_upload") != upload_fingerprint:
    st.session_state["active_upload"] = upload_fingerprint
    st.session_state["selected_land_area"] = None
    st.session_state["map_reset_counter"] = 0

try:
    raw_contours = cached_raw_contours(file_bytes, uploaded.name)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

selected_area = st.session_state.get("selected_land_area")
if selected_area and st.button("Clear selected area"):
    st.session_state["selected_land_area"] = None
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

map_key = f"pond_map_{upload_fingerprint}_{st.session_state.get('map_reset_counter', 0)}"
st.subheader("Map")
st.caption(
    "Draw a polygon to select land. Pond markers and shaded catchments appear after analysis."
    if result is None
    else "The selected land, recommended pond marker, and shaded catchment are shown below."
)
map_data = st_folium(
    build_map(raw_contours, result, selected_area),
    use_container_width=True,
    height=610,
    key=map_key,
    returned_objects=["last_active_drawing"],
)

drawing = (map_data or {}).get("last_active_drawing")
if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
    ring = drawing["geometry"].get("coordinates", [[]])[0]
    drawn_area = [(float(point[0]), float(point[1])) for point in ring]
    if len(drawn_area) >= 4 and drawn_area != selected_area:
        st.session_state["selected_land_area"] = drawn_area
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
    f'{recommended["estimatedCatchmentAreaHectares"]:.2f} ha',
    f'{recommended["estimatedCatchmentAreaSqM"]:,.0f} m²',
)
summary_columns[2].metric("Expected water volume", f'{recommended["estimatedVolumeM3"]:,.0f} m³')
st.caption("Water volume is estimated pond storage capacity from contour geometry; actual collected yield depends on rainfall and runoff.")

render_summary(result)
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
        first.metric("Catchment", f'{candidate["estimatedCatchmentAreaHectares"]:.2f} ha')
        second.metric("Expected volume", f'{candidate["estimatedVolumeM3"]:,.0f} m³')

st.subheader("All suggested ponds")
st.dataframe(pond_table_rows(result), width="stretch", hide_index=True)
with st.expander("Full JSON response"):
    st.json(result)
st.download_button(
    "Download JSON",
    data=json.dumps(result, indent=2),
    file_name="pond_catchment_analysis.json",
    mime="application/json",
)