from dataclasses import dataclass
from typing import List, Tuple

from shapely import Point


@dataclass
class Node:
    diameter: float
    geom: Point
    node_id: int
    stem_id: int
    vector: List[Tuple[float, float]]
