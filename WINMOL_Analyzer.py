#!/usr/bin/env python

##################################################################################
'''Imports'''

import numpy as np
from typing import List, Tuple
from shapely.geometry import Point, LineString, Polygon
from dataclasses import dataclass
import time



##################################################################################
'''Clases'''

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

#DataClass which stores stem parts as start and end points and a list of coordinate tuples for the nodes along the path
@dataclass
class Part:
    start: Tuple[int, int]
    stop: Tuple[int, int]
    path: List[Tuple[int, int]]     
    l_bound: Tuple[int, int]
    u_bound: Tuple[int, int]    
        
    def __eq__(self, other):
        return self.start==other.start and self.stop==other.stop and self.l_bound==other.l_bound and self.u_bound==other.u_bound

    def __hash__(self):
        return hash(('start', self.start, 'stop', self.stop, 'l_bound', self.l_bound,'u_bound',self.u_bound))

#DataClass representing stem objects
@dataclass        
class Stem:
    start: Point
    stop: Point
    path: LineString
    vector:List[Tuple[float,float]]    
    d:List[float]
    l:List[float]
    v:List[float]
    Length:float    
    Volume:float
        
    def __eq__(self, other):
        return self.start==other.start and self.stop==other.stop and self.path==other.path 

    def __hash__(self):
        return hash(('start', tuple(list(self.start.coords)), 'stop', tuple(list(self.stop.coords)), 'path', tuple(list(self.path.coords))))

import IO as IO   
import Prediction as Pred    
import Skeletonization as Skel
import Vectorization as Vec
import Quantification as Quant



