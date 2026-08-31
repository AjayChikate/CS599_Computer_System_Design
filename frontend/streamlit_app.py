from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import folium
import streamlit as st
from streamlit_folium import st_folium


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import analyze_contour_map, load_raw_contours

st.set_page_config(page_title="Contour & Catchment", layout="wide")

RECOMMENDED_COLOR = "#2563EB"   # blue
ALTERNATIVE_COLORS = [
    "#B9622C",  # clay
    "#7B5EA7",  # purple
    "#C9A227",  # gold
    "#3F7D5C",  # green
    "#A3352B",  # brick red
    "#4F8FB0",  # steel blue
    "#8A6D3B",  # bronze
]


def candidate_color(cand: dict[str, Any]) -> str:
    if cand["recommended"]:
        return RECOMMENDED_COLOR
    return ALTERNATIVE_COLORS[(cand["rank"] - 2) % len(ALTERNATIVE_COLORS)]


def elevation_color(elevation: float, min_e: float, max_e: float) -> str:
    t = (elevation - min_e) / (max_e - min_e) if max_e > min_e else 0.5
    r = round(47 + t * (185 - 47))
    g = round(111 + t * (98 - 111))
    b = round(94 + t * (44 - 94))
    return f"rgb({r},{g},{b})"


def build_map(raw_contours: list[dict[str, Any]], result: dict[str, Any]) -> folium.Map:
    candidates = result["pondCandidates"]
    center_lat = result["pondCentroid"]["lat"]
    center_lon = result["pondCentroid"]["lon"]
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")
    elevations = [c["elevation"] for c in raw_contours]
    min_e, max_e = (min(elevations), max(elevations)) if elevations else (0.0, 1.0)
    contour_group = folium.FeatureGroup(name="Contour lines", show=True)
    for contour in raw_contours:
        latlon_points = [(lat, lon) for lon, lat in contour["points"]]
        folium.PolyLine(
            latlon_points,
            color=elevation_color(contour["elevation"], min_e, max_e),
            weight=1,
            opacity=0.6,
            tooltip=f'{contour["elevation"]} m',
        ).add_to(contour_group)
    contour_group.add_to(fmap)
    pond_group = folium.FeatureGroup(name="Suggested ponds", show=True)
    bounds = []
    for cand in candidates:
        is_best = cand["recommended"]
        color = candidate_color(cand)
        boundary_latlon = [(lat, lon) for lon, lat in cand["basinBoundary"]["coordinates"][0]]
        bounds.extend(boundary_latlon)

        folium.Polygon(
            boundary_latlon,
            color=color,
            weight=3 if is_best else 1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.32 if is_best else 0.18,
            tooltip=("Recommended pond" if is_best else f'Alternative #{cand["rank"]}'),
        ).add_to(pond_group)

        popup_html = f"""
            <b>{'Recommended pond' if is_best else 'Alternative pond #' + str(cand['rank'])}</b><br>
            Elevation: {cand['pondElevation']:.1f} m<br>
            Basin area: {cand['basinAreaSqM']:,.0f} m²<br>
            Catchment area: {cand['estimatedCatchmentAreaHectares']:.2f} ha<br>
            Basin depth: {cand['basinDepthM']:.1f} m<br>
            Estimated volume: {cand['estimatedVolumeM3']:,.0f} m³<br>
            Compactness: {cand['compactnessScore']:.2f}
        """
        folium.CircleMarker(
            location=(cand["pondCentroid"]["lat"], cand["pondCentroid"]["lon"]),
            radius=8 if is_best else 6,
            color="white",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(pond_group)

    pond_group.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    if bounds:
        fmap.fit_bounds(bounds, padding=(40, 40))
    return fmap


def render_summary(result: dict[str, Any]) -> None:
    terrain = result["terrainSummary"]
    cols = st.columns(4)
    cols[0].metric("Contour interval", f'{result["contourInterval"]:g} m')
    cols[1].metric("Elevation range", f'{terrain["minElevation"]:g}–{terrain["maxElevation"]:g} m')
    cols[2].metric("Basin candidates found", terrain["basinCandidateCount"])
    cols[3].metric("River-like loops filtered", result["riverAvoidance"]["filteredRiverLikeLoopCount"])


def pond_table_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for c in result["pondCandidates"]:
        rows.append(
            {
                "Rank": c["rank"],
                "Site": "Recommended" if c["recommended"] else f'Alternative #{c["rank"]}',
                "Elevation (m)": c["pondElevation"],
                "Basin area (m²)": c["basinAreaSqM"],
                "Depth (m)": c["basinDepthM"],
                "Volume (m³)": c["estimatedVolumeM3"],
                "Catchment (ha)": c["estimatedCatchmentAreaHectares"],
                "Compactness": c["compactnessScore"],
                "Score": c["score"],
            }
        )
    return rows


def render_pond_table(result: dict[str, Any]) -> None:
    st.dataframe(pond_table_rows(result), width="stretch", hide_index=True)

st.title("Contour & Catchment")
st.caption("Upload a contour survey and find where the water would collect.")

uploaded = st.file_uploader("Contour map (.kml or .kmz)", type=["kml", "kmz"])

if uploaded is not None:
    file_bytes = uploaded.getvalue()

    with st.spinner("Analyzing contours…"):
        try:
            result = analyze_contour_map(file_bytes, uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  
            st.error(f"Unexpected error: {exc}")
            st.stop()

        raw_contours = load_raw_contours(file_bytes, uploaded.name)

    st.success(f'{len(result["pondCandidates"])} pond site(s) suggested.')

    render_summary(result)

    map_col, list_col = st.columns([2, 1])

    with map_col:
        st.subheader("Map")
        fmap = build_map(raw_contours, result)
        st_folium(fmap, use_container_width=True, height=560, key="pond_map", returned_objects=[])

    with list_col:
        st.subheader("Suggested ponds")
        for c in result["pondCandidates"]:
            color = candidate_color(c)
            label = "Recommended" if c["recommended"] else f'Alternative #{c["rank"]}'
            with st.container(border=True):
                st.markdown(
                    f'<span style="display:inline-block;width:11px;height:11px;'
                    f'border-radius:50%;background:{color};margin-right:7px;"></span>'
                    f'<b>{label}</b>',
                    unsafe_allow_html=True,
                )
                m1, m2 = st.columns(2)
                m1.metric("Area", f'{c["basinAreaSqM"]:,.0f} m²')
                m2.metric("Depth", f'{c["basinDepthM"]:.1f} m')
                m3, m4 = st.columns(2)
                m3.metric("Volume", f'{c["estimatedVolumeM3"]:,.0f} m³')
                m4.metric("Catchment", f'{c["estimatedCatchmentAreaHectares"]:.2f} ha')

    st.subheader("All suggested ponds")
    render_pond_table(result)

    with st.expander("Full JSON response"):
        st.json(result)

    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2),
        file_name="pond_catchment_analysis.json",
        mime="application/json",
    )
else:
    st.info("Upload a .kml or .kmz contour export to run the analysis.")