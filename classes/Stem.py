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
    crs: Optional[str] = None  # Default to undefined

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
            if self.start is not None and self.stop is not None:
                return self.start.distance(self.stop)
            return 0
        return sum(self.segment_length_list)

    @property
    def volume(self):
        if len(self.segment_volume_list) == 0:
            return 0
        return sum(self.segment_volume_list)

    # TBD: what's the difference between nodes and vectors beside the geometry?
    def get_nodes(self) -> List[Node]:
        if self.path.coords is None:
            return []
        node_list = []
        for j in range(len(self.path.coords)):
            node_list.append(Node(
                diameter=self.segment_diameter_list[j],
                geom=Point(self.path.coords[j]),
                # TODO: check this
                vector=self.vector,
                stem_id=self.stem_id,
                node_id=j
            ))
        return node_list

    def get_vectors(self) -> List[Vector]:
        if self.path.coords is None:
            return []
        vector_list = []
        for j in range(len(self.path.coords)):
            vector_list.append(Vector(
                diameter=self.segment_diameter_list[j],
                geom=self.path,
                # TODO: check this
                vector=self.vector,
                stem_id=self.stem_id,
                node_id=j
            ))
        return vector_list
