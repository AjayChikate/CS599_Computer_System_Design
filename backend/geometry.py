from __future__ import annotations

import math
from typing import Sequence, Tuple


def shoelace_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def centroid(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area /= 2.0
    if abs(area) < 1e-9:
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    return (cx / (6.0 * area), cy / (6.0 * area))


def polygon_perimeter(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    perimeter = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        perimeter += math.hypot(x2 - x1, y2 - y1)
    return perimeter


def bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_contains(outer: Tuple[float, float, float, float], inner: Tuple[float, float, float, float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def principal_axis_elongation(points: Sequence[Tuple[float, float]], centroid_point: Tuple[float, float]) -> float:
    cx, cy = centroid_point
    sxx = syy = sxy = 0.0
    n = len(points)
    if n == 0:
        return 1.0
    for x, y in points:
        dx, dy = x - cx, y - cy
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    trace = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = max(trace * trace / 4.0 - det, 0.0)
    root = math.sqrt(disc)
    lambda1 = trace / 2.0 + root
    lambda2 = trace / 2.0 - root
    if lambda2 <= 1e-9:
        return 10.0
    return math.sqrt(lambda1 / lambda2)


def to_shared_local_xy(lon0: float, lat0: float, x_scale: float, y_scale: float, points: Sequence[Tuple[float, float]]) -> list[Tuple[float, float]]:
    return [((lon - lon0) * x_scale, (lat - lat0) * y_scale) for lon, lat in points]


def from_shared_local_xy(lon0: float, lat0: float, x_scale: float, y_scale: float, point: Tuple[float, float]) -> Tuple[float, float]:
    x, y = point
    return (lon0 + x / x_scale, lat0 + y / y_scale)
