# --- Helper functions for skeleton operations ---
import numpy as np
from shapely import LineString

# System epsilon
epsilon = np.finfo(float).eps


# Creates a normalized vector from LineStrings or Tuple[Tuple[int]]
def create_vector(line):
    if isinstance(line, LineString):
        v = [line.coords[-1][0] - line.coords[0][0],
             line.coords[-1][1] - line.coords[0][1]]
    else:
        v = [(line[1][0] - line[0][0]), (line[1][1] - line[0][1])]
    return v / (np.linalg.norm(v) + epsilon)


# Calculates the angle between 2 vectors
def ang(line_a, line_b):
    v_a = create_vector(line_a)
    v_b = create_vector(line_b)
    dot_product = np.dot(v_a, v_b)
    dot_product = np.clip(dot_product, -1, 1)
    angle = np.arccos(dot_product)
    ang_deg = np.degrees(angle) % 380
    if ang_deg > 180:
        ang_deg = ang_deg - 360
    return ang_deg
