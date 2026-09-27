from __future__ import annotations

import math
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import MAX_ALTERNATIVE_CANDIDATES, MIN_LOOP_AREA_M2, RIVER_ELONGATION_MIN, RIVER_ROUNDNESS_MAX
from .geometry import (
    bbox,
    bbox_contains,
    centroid,
    from_shared_local_xy,
    point_in_polygon,
    polygon_perimeter,
    principal_axis_elongation,
    shoelace_area,
    to_shared_local_xy,
)
from .models import BasinCandidate, Loop
from .parsing import load_raw_contours

logger = logging.getLogger(__name__)
logging.getLogger("backend").setLevel(logging.INFO)


def _build_containment_tree(loops: List[Loop]) -> None:
    by_area_desc = sorted(loops, key=lambda l: l.area_m2, reverse=True)
    for loop in loops:
        best_parent: Optional[Loop] = None
        best_parent_area = math.inf
        for candidate in by_area_desc:
            if candidate is loop:
                continue
            if candidate.area_m2 <= loop.area_m2:
                break
            if candidate.area_m2 >= best_parent_area:
                continue
            if not bbox_contains(candidate.bbox, loop.bbox):
                continue
            if point_in_polygon(loop.centroid_xy, candidate.xy):
                best_parent = candidate
                best_parent_area = candidate.area_m2
        if best_parent is not None:
            loop.parent = best_parent
            best_parent.children.append(loop)


def _extract_basin_candidates(roots: List[Loop]) -> List[BasinCandidate]:
    candidates: List[BasinCandidate] = []

    def walk(node: Loop, open_chain: List[Loop]) -> None:
        if open_chain and node.elevation < open_chain[-1].elevation:
            chain = open_chain + [node]
        else:
            chain = [node]

        if not node.children:
            if len(chain) >= 1:
                rim, floor = chain[0], chain[-1]
                if rim is not floor:
                    depth = rim.elevation - floor.elevation
                    avg_round = sum(c.roundness for c in chain) / len(chain)
                    candidates.append(
                        BasinCandidate(
                            rim=rim,
                            floor=floor,
                            chain=chain,
                            catchment_area_m2=rim.area_m2,
                            depth_m=depth,
                            avg_roundness=avg_round,
                        )
                    )
        for child in node.children:
            walk(child, chain)

    for root in roots:
        walk(root, [])
    return candidates


def _basin_volume_m3(chain: List[Loop]) -> float:
    if len(chain) < 2:
        return 0.0
    volume = 0.0
    for i in range(len(chain) - 1):
        h = chain[i].elevation - chain[i + 1].elevation
        a1, a2 = chain[i].area_m2, chain[i + 1].area_m2
        volume += (h / 3.0) * (a1 + a2 + math.sqrt(max(a1 * a2, 0.0)))
    last_h = chain[-2].elevation - chain[-1].elevation
    volume += (last_h / 3.0) * chain[-1].area_m2
    return volume


def _score_candidates(candidates: List[BasinCandidate]) -> List[Tuple[float, BasinCandidate]]:
    if not candidates:
        return []

    depths = [c.depth_m for c in candidates]
    areas = [c.catchment_area_m2 for c in candidates]
    rounds = [c.avg_roundness for c in candidates]

    def norm(value: float, values: Sequence[float]) -> float:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return 1.0
        return (value - lo) / (hi - lo)

    scored: List[Tuple[float, BasinCandidate]] = []
    for candidate in candidates:
        score = (
            0.4 * norm(candidate.depth_m, depths)
            + 0.3 * norm(candidate.catchment_area_m2, areas)
            + 0.3 * norm(candidate.avg_roundness, rounds)
        )
        scored.append((score, candidate))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def _point_on_segment(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    epsilon = 1e-10
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return (
        abs(cross) <= epsilon
        and min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon
        and min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon
    )


def _inside_or_on_boundary(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    if any(_point_on_segment(point, polygon[index], polygon[index + 1]) for index in range(len(polygon) - 1)):
        return True
    return point_in_polygon(point, polygon)


def analyze_contours(raw_contours: List[dict[str, Any]]) -> dict[str, Any]:
    analysis_started = time.perf_counter()
    if not raw_contours:
        logger.warning("Contour analysis rejected: no contour lines provided")
        raise ValueError("No contour lines were found in the uploaded file.")

    all_points = [p for contour in raw_contours for p in contour["points"]]
    logger.info("Contour analysis started: contours=%d coordinate_points=%d", len(raw_contours), len(all_points))
    lon0 = sum(p[0] for p in all_points) / len(all_points)
    lat0 = sum(p[1] for p in all_points) / len(all_points)
    lat_rad = math.radians(lat0)
    x_scale = 111_320.0 * math.cos(lat_rad)
    y_scale = 110_574.0

    loops: List[Loop] = []
    for idx, contour in enumerate(raw_contours):
        lonlat = contour["points"]
        xy = to_shared_local_xy(lon0, lat0, x_scale, y_scale, lonlat)
        area = shoelace_area(xy)
        if area <= MIN_LOOP_AREA_M2:
            continue

        centroid_xy = centroid(xy)
        perimeter = polygon_perimeter(xy)
        roundness = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        elongation = principal_axis_elongation(xy, centroid_xy)
        river_like = (
            roundness < RIVER_ROUNDNESS_MAX
            or (perimeter / math.sqrt(max(area, 1.0))) > RIVER_ELONGATION_MIN
            or elongation > 3.0
        )

        loops.append(
            Loop(
                index=idx,
                elevation=float(contour["elevation"]),
                lonlat=lonlat,
                xy=xy,
                area_m2=area,
                centroid_xy=centroid_xy,
                perimeter_m=perimeter,
                roundness=roundness,
                elongation=elongation,
                bbox=bbox(xy),
                is_river_like=river_like,
            )
        )

    if not loops:
        logger.warning("Contour analysis found no usable closed loops")
        raise ValueError("The uploaded contours do not contain a usable closed basin.")

    _build_containment_tree(loops)
    roots = [loop for loop in loops if loop.parent is None]
    all_candidates = _extract_basin_candidates(roots)
    basin_candidates = [
        candidate for candidate in all_candidates
        if not candidate.floor.is_river_like and not any(link.is_river_like for link in candidate.chain)
    ]
    river_like_count = sum(1 for loop in loops if loop.is_river_like)

    if not basin_candidates:
        logger.warning("No basin candidates passed compactness and river filters: loops=%d", len(loops))
        raise ValueError("No compact, non-river closed basin was detected in this contour map.")

    ranked = _score_candidates(basin_candidates)
    best_score, best = ranked[0]

    unique_elevations = sorted({loop.elevation for loop in loops})
    contour_interval = 1.0
    if len(unique_elevations) > 1:
        intervals = [second - first for first, second in zip(unique_elevations, unique_elevations[1:])]
        contour_interval = sum(intervals) / len(intervals)

    def to_lonlat(xy_point: Tuple[float, float]) -> Tuple[float, float]:
        return from_shared_local_xy(lon0, lat0, x_scale, y_scale, xy_point)

    def boundary_geojson(loop: Loop) -> dict[str, Any]:
        return {
            "type": "Polygon",
            "coordinates": [[[round(lon, 6), round(lat, 6)] for lon, lat in loop.lonlat]],
        }

    def candidate_payload(rank: int, score: float, candidate: BasinCandidate, recommended: bool) -> dict[str, Any]:
        c_lon, c_lat = to_lonlat(candidate.floor.centroid_xy)
        volume_m3 = _basin_volume_m3(candidate.chain)
        return {
            "rank": rank,
            "recommended": recommended,
            "score": round(score, 4),
            "pondElevation": round(candidate.floor.elevation, 3),
            "pondCentroid": {"lon": round(c_lon, 6), "lat": round(c_lat, 6)},
            "basinAreaSqM": round(candidate.floor.area_m2, 2),
            "estimatedCatchmentAreaSqM": round(candidate.catchment_area_m2, 2),
            "basinDepthM": round(candidate.depth_m, 3),
            "estimatedVolumeM3": round(volume_m3, 2),
            "compactnessScore": round(candidate.avg_roundness, 4),
            "basinBoundary": boundary_geojson(candidate.rim),
        }

    best_volume_m3 = _basin_volume_m3(best.chain)
    pond_lon, pond_lat = to_lonlat(best.floor.centroid_xy)

    pond_candidates = [candidate_payload(1, best_score, best, True)]
    alternatives: List[Dict[str, Any]] = []
    seen_floor_indices = {best.floor.index}
    rank = 2
    for score, candidate in ranked[1:]:
        if candidate.floor.index in seen_floor_indices:
            continue
        seen_floor_indices.add(candidate.floor.index)
        payload = candidate_payload(rank, score, candidate, False)
        pond_candidates.append(payload)
        alternatives.append({k: v for k, v in payload.items() if k not in ("rank", "recommended")})
        rank += 1
        if len(alternatives) >= MAX_ALTERNATIVE_CANDIDATES:
            break

    result = {
        "status": "ok",
        "contourInterval": round(contour_interval, 3),
        "pondElevation": round(best.floor.elevation, 3),
        "estimatedCatchmentAreaSqM": round(best.catchment_area_m2, 2),
        "basinAreaSqM": round(best.floor.area_m2, 2),
        "basinDepthM": round(best.depth_m, 3),
        "estimatedVolumeM3": round(best_volume_m3, 2),
        "compactnessScore": round(best.avg_roundness, 4),
        "confidenceScore": round(best_score, 4),
        "candidateContours": len(loops),
        "pondCentroid": {"lon": round(pond_lon, 6), "lat": round(pond_lat, 6)},
        "method": (
            "containment-tree basin detection: nested contour loops are linked into a "
            "parent/child forest, maximal runs of strictly decreasing elevation toward the "
            "interior are treated as topographic basins, basins are ranked by depth, "
            "catchment area, and compactness, and volume is estimated by integrating "
            "ring area over elevation as stacked conical frustums"
        ),
        "riverAvoidance": {
            "enabled": True,
            "filteredRiverLikeLoopCount": river_like_count,
            "preference": (
                "compact, round enclosed basins are preferred; nested runs that pass through "
                "an elongated (river/valley-like) ring at any point are excluded from pond "
                "candidacy"
            ),
        },
        "basinBoundary": boundary_geojson(best.rim),
        "terrainSummary": {
            "minElevation": round(min(loop.elevation for loop in loops), 3),
            "maxElevation": round(max(loop.elevation for loop in loops), 3),
            "contourCount": len(raw_contours),
            "usableLoopCount": len(loops),
            "basinCandidateCount": len(basin_candidates),
        },
        "pondCandidates": pond_candidates,
        "alternativeCandidates": alternatives,
    }
    logger.info(
        "Contour analysis complete: loops=%d candidates=%d alternatives=%d recommended=(%.6f, %.6f) runtime=%.2f s",
        len(loops),
        len(pond_candidates),
        len(alternatives),
        pond_lat,
        pond_lon,
        time.perf_counter() - analysis_started,
    )
    return result


def analyze_contours_in_area(
    raw_contours: List[dict[str, Any]], area_polygon: List[Tuple[float, float]]
) -> dict[str, Any]:
    if len(area_polygon) < 4 or area_polygon[0] != area_polygon[-1]:
        raise ValueError("Draw a closed polygon around the land area to analyze.")

    area_bounds = (
        min(point[0] for point in area_polygon),
        min(point[1] for point in area_polygon),
        max(point[0] for point in area_polygon),
        max(point[1] for point in area_polygon),
    )
    selected_contours = []
    for contour in raw_contours:
        points = contour["points"]
        contour_bounds = (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        )
        if (
            contour_bounds[0] < area_bounds[0]
            or contour_bounds[1] < area_bounds[1]
            or contour_bounds[2] > area_bounds[2]
            or contour_bounds[3] > area_bounds[3]
        ):
            continue
        if all(_inside_or_on_boundary(point, area_polygon) for point in points):
            selected_contours.append(contour)

    if not selected_contours:
        raise ValueError("No complete contour rings fit inside the selected land area.")
    return analyze_contours(selected_contours)


def analyze_contour_map(file_bytes: bytes, filename: str) -> dict[str, Any]:
    return analyze_contours(load_raw_contours(file_bytes, filename))
