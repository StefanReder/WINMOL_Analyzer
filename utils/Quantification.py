#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import List

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import LineString, Point

from classes.Stem import Stem
from classes.Timer import Timer
from utils.Geometry import create_vector

# System epsilon
epsilon = np.finfo(float).eps

################################################################################
"""Stem quantification operations"""


# Parallel quantification of stem parameters
def quantify_stems(stems: List[Stem], pred, profile):
    # Quantification of the stem parameters
    t = Timer()
    t.start()

    stems_ = []
    stems__ = []

    print("#######################################################")
    print("Quantifying stems")
    stems = get_diameters(stems, pred, profile)
    pool = mp.Pool(mp.cpu_count() - 1)
    for stem in pool.imap_unordered(clean_diameter, stems):
        stems_.append(stem)

    for stem in pool.imap_unordered(quantify_stem, stems_):
        stems__.append(stem)
    pool.close()

    print("Volume of ", len(stems__), " stems calculated")
    t.stop()
    print("#######################################################")
    print("")
    return stems__


# Parallel version of get_diameters
def get_diameters(stems: List[Stem], pred, profile):
    # Calculates the diameters for all stems in the list
    transform = profile['transform']
    mask = None
    pred[np.where(pred < 0.5)] = 0
    pred[np.where(pred >= 0.5)] = 1
    pred = pred.astype(np.int16)
    pred_shapes_ = ({'properties': {'raster_val': v}, 'geometry': s} for
                    i, (s, v) in enumerate(
        rasterio.features.shapes(pred, mask=mask, transform=transform)))
    pred_shapes = list(pred_shapes_)
    pred_shapes = gpd.GeoDataFrame.from_features(pred_shapes)
    pred_shapes = pred_shapes[pred_shapes['raster_val'] == 1]

    diam_count = 0
    measured_stems = []

    def return_callback(measured_stem):
        measured_stems.append(measured_stem)
        nonlocal diam_count
        diam_count = diam_count + len(measured_stem.segment_diameter_list)

    def error_callback(error):
        print(error, flush=True)

    pool = mp.Pool(mp.cpu_count() - 1)
    r = []
    for stem in stems:
        r.append(pool.apply_async(calc_v_d, args=(stem, pred_shapes),
                                  callback=return_callback,
                                  error_callback=error_callback))

    for r_ in r:
        r_.wait()
    pool.close()

    print(diam_count, " measurements of diameters where conducted")
    return measured_stems


# Calculates the volume and length of a stem
def quantify_stem(stem: Stem):
    stem.segment_length_list = []
    stem.segment_volume_list = []
    for i in range(0, len(stem.path.coords) - 1):
        seg_l, seg_vol = calc_l_v(
            stem.path.coords[i],
            stem.path.coords[i + 1],
            stem.segment_diameter_list[i],
            stem.segment_diameter_list[i + 1]
        )
        stem.segment_length_list.append(seg_l)
        stem.segment_volume_list.append(seg_vol)
    stem.volume = sum(stem.segment_volume_list)
    stem.length = stem.start.distance(stem.stop)
    return stem


# --- Helper functions ---

# Replaces outlier from the diameter list by interpolation or substitution
def clean_diameter(stem):
    q1 = np.quantile(stem.segment_diameter_list, 0.25)
    q3 = np.quantile(stem.segment_diameter_list, 0.75)
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    if len(stem.segment_diameter_list) > 4:
        for i in range(1, len(stem.segment_diameter_list) - 2):
            i_uw = stem.segment_diameter_list[i] > uw
            i_lw = stem.segment_diameter_list[i] < lw
            if i_uw or i_lw:
                wd1 = stem.segment_diameter_list[i - 1] * abs(
                    Point(stem.path.coords[i]).distance(
                        Point(stem.path.coords[i + 1])))
                wd2 = stem.segment_diameter_list[i + 1] * abs(
                    Point(stem.path.coords[i - 1]).distance(
                        Point(stem.path.coords[i])))
                d12 = abs(Point(stem.path.coords[i - 1]).distance(
                    Point(stem.path.coords[i + 1])))
                stem.segment_diameter_list[i] = (wd1 + wd2) / d12
        list_uw = stem.segment_diameter_list[0] > uw
        list_lw = stem.segment_diameter_list[0] < lw
        if list_uw or list_lw:
            stem.segment_diameter_list[0] = stem.segment_diameter_list[1]
        diameter_list_uw = stem.segment_diameter_list[-1] > uw
        diameter_list_lw = stem.segment_diameter_list[-1] < lw
        if diameter_list_uw or diameter_list_lw:
            stem.segment_diameter_list[-1] = stem.segment_diameter_list[-2]
    return stem


# calculate radial vector to measure the diameter
def calc_v_d(stem, contours):
    vector = create_vector((stem.path.coords[0], stem.path.coords[1]))
    vector = [-vector[1], vector[0]]
    p1 = Point(stem.path.coords[0][0] - vector[0] * 1.0,
               stem.path.coords[0][1] - vector[1] * 1.0)
    p2 = Point(stem.path.coords[0][0] + vector[0] * 1.0,
               stem.path.coords[0][1] + vector[1] * 1.0)
    vector = LineString([p1, p2])
    stem.segment_diameter_list.append(
        calc_d(stem.path.coords[0], vector, contours))
    stem.vector.append(vector)

    for i in range(1, len(stem.path.coords) - 1):
        vector = create_vector(
            (stem.path.coords[i - 1], stem.path.coords[i + 1]))
        vector = [-vector[1], vector[0]]
        p1 = Point(stem.path.coords[i][0] - vector[0] * 1.0,
                   stem.path.coords[i][1] - vector[1] * 1.0)
        p2 = Point(stem.path.coords[i][0] + vector[0] * 1.0,
                   stem.path.coords[i][1] + vector[1] * 1.0)
        vector = LineString([p1, p2])
        stem.segment_diameter_list.append(
            calc_d(stem.path.coords[i], vector, contours))
        stem.vector.append(vector)

    vector = create_vector((stem.path.coords[-2], stem.path.coords[-1]))
    vector = [-vector[1], vector[0]]
    p1 = Point(stem.path.coords[-1][0] - vector[0] * 1.0,
               stem.path.coords[-1][1] - vector[1] * 1.0)
    p2 = Point(stem.path.coords[-1][0] + vector[0] * 1.0,
               stem.path.coords[-1][1] + vector[1] * 1.0)
    vector = LineString([p1, p2])
    stem.segment_diameter_list.append(
        calc_d(stem.path.coords[-1], vector, contours))
    stem.vector.append(vector)

    return stem


# Calculate the diameter for a specific node
def calc_d(node, line, contours):
    node = Point(node)
    d = 0
    intersects = contours.geometry.intersection(line)
    intersects = intersects[~intersects.is_empty]

    for i in intersects:
        if node.distance(i) < 0.01:
            if i.geom_type == 'MultiLineString':
                for i_ in i.geoms:
                    if node.distance(i_) < 0.01:
                        d = i_.length
            else:
                d = i.length
    return d


# Calculate the length and volume of a segment described by 2 points and the
# respective diameters
def calc_l_v(p1, p2, d1, d2):
    length = math.dist(p1, p2)
    v = 1 / 3 * math.pi * (
        (d1 / 2) ** 2 + (d1 / 2) * (d2 / 2) + (d2 / 2) ** 2
    ) * length
    return length, v
