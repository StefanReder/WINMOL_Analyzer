# DataClass which stores stem parts as start and end points and a list of
# coordinate tuples for the nodes along the path
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Part:
    start: Tuple[int, int]
    stop: Tuple[int, int]
    path: List[Tuple[int, int]]
    l_bound: Tuple[int, int]
    u_bound: Tuple[int, int]

    def __eq__(self, other):
        return (self.start == other.start and self.stop == other.stop and
                self.l_bound == other.l_bound and self.u_bound == other.u_bound)

    def __hash__(self):
        return hash(
            ('start', self.start, 'stop', self.stop,
             'l_bound', self.l_bound, 'u_bound', self.u_bound)
        )
