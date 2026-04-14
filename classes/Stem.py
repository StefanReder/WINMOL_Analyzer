# DataClass representing stem objects
from dataclasses import dataclass
from typing import List, Tuple, Optional

from shapely import LineString, Point

from classes.Node import Node
from classes.Vector import Vector


@dataclass
class Stem:
    start: Point
    stop: Point
    path: LineString
    vector: List[Tuple[float, float]]
    segment_diameter_list: List[float]
    segment_length_list: List[float]
    segment_volume_list: List[float]
    stem_id: Optional[str] = None
    crs: Optional[str] = None
    tree_x: Optional[float] = None
    tree_y: Optional[float] = None
    direction_x: float = 0.0
    direction_y: float = 0.0
    direction_deg: Optional[float] = None
    direction_confidence: float = 0.0
    owner_partition_id: Optional[str] = None
    source_tile_id: Optional[str] = None
    is_border_candidate: bool = False

    def __post_init__(self):
        if self.stem_id is not None:
            self.stem_id = str(self.stem_id)
        if self.owner_partition_id is not None:
            self.owner_partition_id = str(self.owner_partition_id)
        if self.source_tile_id is not None:
            self.source_tile_id = str(self.source_tile_id)

    def __eq__(self, other):
        return (self.start == other.start and self.stop == other.stop
                and self.path == other.path and self.crs == other.crs)

    def __hash__(self):
        return hash(
            ('start', tuple(list(self.start.coords)),
             'stop', tuple(list(self.stop.coords)),
             'path', tuple(list(self.path.coords)),
             'crs', self.crs)
        )

    @property
    def length(self):
        if len(self.segment_length_list) == 0:
            if self.path is not None:
                try:
                    return float(self.path.length)
                except Exception:
                    pass
            if self.start is not None and self.stop is not None:
                return self.start.distance(self.stop)
            return 0
        return sum(self.segment_length_list)

    @property
    def volume(self):
        if len(self.segment_volume_list) == 0:
            return 0
        return sum(self.segment_volume_list)

    def get_nodes(self) -> List[Node]:
        if self.path.coords is None:
            return []
        node_list = []
        for j in range(len(self.path.coords)):
            diameter = None
            try:
                diameter = self.segment_diameter_list[j]
            except Exception:
                diameter = None
            node_list.append(Node(
                diameter=diameter,
                geom=Point(self.path.coords[j]),
                vector=self.vector,
                stem_id=self.stem_id,
                node_id=j,
            ))
        return node_list

    def get_vectors(self) -> List[Vector]:
        if self.path.coords is None:
            return []
        vector_list = []
        for j in range(len(self.path.coords)):
            diameter = None
            try:
                diameter = self.segment_diameter_list[j]
            except Exception:
                diameter = None
            vector_list.append(Vector(
                diameter=diameter,
                geom=self.path,
                vector=self.vector,
                stem_id=self.stem_id,
                node_id=j,
            ))
        return vector_list
