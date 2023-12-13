from dataclasses import dataclass
from typing import List, Tuple

from shapely import LineString


@dataclass
class Vector:
    diameter: float
    geom: LineString
    node_id: int
    stem_id: int
    vector: List[Tuple[float, float]]
