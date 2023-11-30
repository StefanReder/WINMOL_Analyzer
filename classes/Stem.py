# DataClass representing stem objects
from dataclasses import dataclass
from typing import List, Tuple

from shapely import LineString, Point


@dataclass
class Stem:
    start: Point
    stop: Point
    path: LineString
    vector: List[Tuple[float, float]]
    segment_diameter_list: List[float]
    segment_length_list: List[float]
    segment_volume_list: List[float]
    length: float
    volume: float

    def __eq__(self, other):
        return (self.start == other.start and self.stop == other.stop
                and self.path == other.path)

    def __hash__(self):
        return hash(
            ('start', tuple(list(self.start.coords)),
             'stop', tuple(list(self.stop.coords)),
             'path', tuple(list(self.path.coords)))
        )
