from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Loop:
    index: int
    elevation: float
    lonlat: List[Tuple[float, float]]
    xy: List[Tuple[float, float]]
    area_m2: float
    centroid_xy: Tuple[float, float]
    perimeter_m: float
    roundness: float
    elongation: float
    bbox: Tuple[float, float, float, float]
    is_river_like: bool
    parent: Optional["Loop"] = None
    children: List["Loop"] = field(default_factory=list)


@dataclass
class BasinCandidate:
    rim: Loop
    floor: Loop
    chain: List[Loop]
    catchment_area_m2: float
    depth_m: float
    avg_roundness: float
