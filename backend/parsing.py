from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Any, List
from xml.etree import ElementTree as ET


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_kml_text(kml_text: str) -> List[dict[str, Any]]:
    root = ET.fromstring(kml_text)
    ns = {
        "k": "http://www.opengis.net/kml/2.2",
        "gx": "http://www.google.com/kml/ext/2.2",
    }
    contours: List[dict[str, Any]] = []
    for placemark in root.findall(".//k:Placemark", ns):
        name = placemark.findtext("k:name", default="", namespaces=ns)
        elevation = _parse_float(name)
        if elevation is None:
            continue
        line_string = placemark.find(".//k:LineString", ns)
        if line_string is None:
            continue
        coordinates_text = line_string.findtext("k:coordinates", default="", namespaces=ns)
        if not coordinates_text or not coordinates_text.strip():
            continue

        points: List[tuple[float, float]] = []
        for coordinate in coordinates_text.strip().split():
            parts = coordinate.split(",")
            if len(parts) < 2:
                continue
            try:
                lon_val = float(parts[0])
                lat_val = float(parts[1])
            except ValueError:
                continue
            points.append((lon_val, lat_val))
        if len(points) < 3:
            continue
        if points[0] != points[-1]:
            points.append(points[0])
        contours.append({"elevation": elevation, "points": points})
    return contours


def parse_kmz_bytes(file_bytes: bytes) -> List[dict[str, Any]]:
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as zf:
            kml_files = [name for name in zf.namelist() if name.lower().endswith(".kml")]
            if not kml_files:
                raise ValueError("KMZ upload does not contain a KML file.")
            kml_bytes = zf.read(kml_files[0])
            try:
                return parse_kml_text(kml_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                return parse_kml_text(kml_bytes.decode("utf-8", errors="ignore"))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded KMZ file is invalid.") from exc


def load_raw_contours(file_bytes: bytes, filename: str) -> List[dict[str, Any]]:
    if filename.lower().endswith(".kmz"):
        return parse_kmz_bytes(file_bytes)
    if filename.lower().endswith(".kml"):
        return parse_kml_text(file_bytes.decode("utf-8"))
    raise ValueError("Unsupported contour file type. Please upload a KML or KMZ file.")
