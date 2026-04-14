#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio.features
import scipy.ndimage as ndi
from shapely.geometry import LineString, Point

from classes.Stem import Stem
from classes.Timer import Timer
from utils.Geometry import create_vector

# System epsilon
epsilon = np.finfo(float).eps


def _as_binary_mask(pred):
    arr = np.asarray(pred)
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype == np.uint8 and arr.size and arr.max() <= 1:
        return arr.astype(bool, copy=False)
    return arr >= 0.5


def _worker_count(config=None):
    value = getattr(config, 'cpu_workers', None) \
        if config is not None else None
    if value is None:
        value = max(mp.cpu_count() - 1, 1)
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _direction_confidence_threshold(config=None):
    try:
        return float(getattr(config, 'direction_confidence_threshold', 0.15))
    except Exception:
        return 0.15



def ensure_stem_ids(stems: List[Stem], prefix: str = "stem"):
    seen = set()
    counter = 0
    for stem in stems or []:
        sid = getattr(stem, 'stem_id', None)
        sid = None if sid in (None, "", -1) else str(sid)
        if sid is None or sid in seen:
            while True:
                candidate = f"{prefix}_{counter:08d}"
                counter += 1
                if candidate not in seen:
                    sid = candidate
                    break
        stem.stem_id = sid
        seen.add(sid)
    return stems


def refresh_stems_direction_bulk(stems, config=None):
    refreshed = []
    for stem in stems or []:
        stem = refresh_stem_direction(stem, config=config)
        stem = refresh_measurement_vectors(stem, config=config)
        refreshed.append(stem)
    return refreshed


################################################################################
"""Stem quantification operations"""


def quantify_stems(stems: List[Stem], pred, profile, config=None):
    t = Timer()
    t.start()

    print("#######################################################")
    print("Quantifying stems")
    if not stems:
        print("0 measurements of diameters where conducted")
        print("Volume of  0  stems calculated")
        t.stop()
        print("#######################################################")
        print("")
        return []

    stems = determine_stem_diameters(stems, pred, profile, config=config)
    stems = compute_stem_volumes(stems, config=config)

    print("Volume of ", len(stems), " stems calculated")
    t.stop()
    print("#######################################################")
    print("")
    return stems


def determine_stem_diameters(stems: List[Stem], pred, profile, config=None):
    print("Determining diameters")
    if not stems:
        print("0 measurements of diameters where conducted")
        return []

    stems = ensure_stem_ids(list(stems), prefix="part")
    stems = get_diameters(stems, pred, profile, config=config)
    workers = min(_worker_count(config), max(len(stems), 1))

    cleaned = []
    if workers <= 1 or len(stems) <= 1:
        for stem in stems:
            stem = clean_diameter(stem)
            stem = refresh_stem_direction(stem, config=config)
            stem = refresh_measurement_vectors(stem, config=config)
            cleaned.append(stem)
    else:
        with mp.Pool(workers) as pool:
            for stem in pool.imap_unordered(clean_diameter, stems):
                stem = refresh_stem_direction(stem, config=config)
                stem = refresh_measurement_vectors(stem, config=config)
                cleaned.append(stem)
    return cleaned


def compute_stem_volumes(stems: List[Stem], config=None):
    stems = ensure_stem_ids(list(stems), prefix="stem")
    workers = min(_worker_count(config), max(len(stems), 1))
    updated = []
    if workers <= 1 or len(stems) <= 1:
        for stem in stems:
            stem = refresh_stem_direction(stem, config=config)
            stem = refresh_measurement_vectors(stem, config=config)
            updated.append(quantify_stem(stem))
        return updated

    with mp.Pool(workers) as pool:
        stems_refreshed = []
        for stem in stems:
            stems_refreshed.append(refresh_stem_direction(stem, config=config))
        for stem in pool.imap_unordered(quantify_stem, stems_refreshed):
            updated.append(stem)
    return updated


def get_diameters(stems: List[Stem], pred, profile, config=None):
    transform = profile['transform']
    pred_bin = _as_binary_mask(pred).astype(np.int16, copy=False)

    diameter_method = str(getattr(config, 'diameter_method', 'contour'))\
        .lower() if config is not None else 'contour'

    diam_count = 0
    measured_stems = []

    def return_callback(measured_stem):
        measured_stems.append(measured_stem)
        nonlocal diam_count
        diam_count = diam_count + len(measured_stem.segment_diameter_list)

    def error_callback(error):
        print(error, flush=True)

    workers = min(_worker_count(config), max(len(stems), 1))

    if diameter_method == 'edt':
        edt_map = _distance_transform_m(pred_bin, profile)
        if workers <= 1 or len(stems) <= 1:
            for stem in stems:
                try:
                    return_callback(
                        calc_v_d_edt(stem, edt_map, profile, config=config))
                except Exception as error:
                    error_callback(error)
        else:
            for stem in stems:
                try:
                    return_callback(
                        calc_v_d_edt(stem, edt_map, profile, config=config))
                except Exception as error:
                    error_callback(error)
    else:
        mask = None
        pred_shapes_ = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                rasterio.features.shapes(
                    pred_bin, mask=mask, transform=transform))
        )
        pred_shapes = list(pred_shapes_)
        pred_shapes = gpd.GeoDataFrame.from_features(pred_shapes)
        pred_shapes = pred_shapes[pred_shapes['raster_val'] == 1]

        if workers <= 1 or len(stems) <= 1:
            for stem in stems:
                try:
                    return_callback(calc_v_d_contour(
                        stem, pred_shapes, config=config))
                except Exception as error:
                    error_callback(error)
        else:
            with mp.Pool(workers) as pool:
                r = []
                for stem in stems:
                    r.append(pool.apply_async(
                        calc_v_d_contour,
                        args=(stem, pred_shapes, config),
                        callback=return_callback,
                        error_callback=error_callback))
                for r_ in r:
                    r_.wait()

    print(diam_count, " measurements of diameters where conducted")
    return measured_stems


def quantify_stem(stem: Stem):
    stem.segment_length_list = []
    stem.segment_volume_list = []
    for i in range(0, len(stem.path.coords) - 1):
        d1 = stem.segment_diameter_list[i] \
            if i < len(stem.segment_diameter_list) else 0.0
        d2 = stem.segment_diameter_list[i + 1] \
            if i + 1 < len(stem.segment_diameter_list) else d1
        seg_l, seg_vol = calc_l_v(
            stem.path.coords[i],
            stem.path.coords[i + 1],
            d1,
            d2,
        )
        stem.segment_length_list.append(seg_l)
        stem.segment_volume_list.append(seg_vol)
    return stem


# --- Helper functions ---


def _pixel_size(profile) -> Tuple[float, float]:
    return abs(profile['transform'][0]), abs(profile['transform'][4])


def clean_diameter(stem):
    if len(stem.segment_diameter_list) < 2:
        return stem
    if len(stem.segment_diameter_list) == 2:
        return stem
    q1 = np.quantile(stem.segment_diameter_list, 0.25)
    q3 = np.quantile(stem.segment_diameter_list, 0.75)
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    if len(stem.segment_diameter_list) > 4:
        for i in range(1, len(stem.segment_diameter_list) - 1):
            i_uw = stem.segment_diameter_list[i] > uw
            i_lw = stem.segment_diameter_list[i] < lw
            if i_uw or i_lw:
                wd1 = stem.segment_diameter_list[i - 1] * abs(
                    Point(stem.path.coords[i]).distance(Point(
                        stem.path.coords[i + 1])))
                wd2 = stem.segment_diameter_list[i + 1] * abs(
                    Point(stem.path.coords[i - 1]).distance(Point(
                        stem.path.coords[i])))
                d12 = abs(Point(stem.path.coords[i - 1]).distance(
                    Point(stem.path.coords[i + 1])))
                if d12 > epsilon:
                    stem.segment_diameter_list[i] = (wd1 + wd2) / d12
        if (
            stem.segment_diameter_list[0] > uw
            or stem.segment_diameter_list[0] < lw
        ):
            stem.segment_diameter_list[0] = stem.segment_diameter_list[1]

        if (
            stem.segment_diameter_list[-1] > uw
            or stem.segment_diameter_list[-1] < lw
        ):
            stem.segment_diameter_list[-1] = stem.segment_diameter_list[-2]
    return stem


def _local_normal(coords, idx):
    if len(coords) < 2:
        return (0.0, 1.0)
    if idx == 0:
        v = create_vector((coords[0], coords[1]))
    elif idx == len(coords) - 1:
        v = create_vector((coords[-2], coords[-1]))
    else:
        v = create_vector((coords[idx - 1], coords[idx + 1]))
    return (-float(v[1]), float(v[0]))


def _measurement_vector(node_xy, normal_xy, half_len):
    nx, ny = normal_xy
    x, y = node_xy
    p1 = Point(x - nx * half_len, y - ny * half_len)
    p2 = Point(x + nx * half_len, y + ny * half_len)
    return LineString([p1, p2])


def _diameter_sample_indices(coord_count: int):
    if coord_count <= 2:
        return list(range(coord_count))
    return list(range(1, coord_count - 1))


def _restore_endpoint_diameters(diameters, coord_count: int):
    if coord_count <= 0:
        return []
    if len(diameters) == coord_count:
        return list(diameters)
    if coord_count == 1:
        return [float(diameters[0])] if diameters else [0.0]
    if coord_count == 2:
        if len(diameters) >= 2:
            return [float(diameters[0]), float(diameters[1])]
        if len(diameters) == 1:
            value = float(diameters[0])
            return [value, value]
        return [0.0, 0.0]
    if not diameters:
        return [0.0] * coord_count
    restored = [float(diameters[0])]
    restored.extend(float(value) for value in diameters)
    restored.append(float(diameters[-1]))
    if len(restored) < coord_count:
        fill = restored[-1] if restored else 0.0
        restored.extend([fill] * (coord_count - len(restored)))
    return restored[:coord_count]


def _cumulative_distances(coords):
    if not coords:
        return np.array([], dtype=float)
    dist = [0.0]
    for i in range(1, len(coords)):
        dist.append(dist[-1] + math.dist(coords[i - 1], coords[i]))
    return np.asarray(dist, dtype=float)


def _direction_metadata_from_coords(stem: Stem):
    try:
        sx, sy = stem.start.coords[0]
    except Exception:
        sx, sy = (None, None)
    try:
        ex, ey = stem.stop.coords[0]
    except Exception:
        ex, ey = (None, None)
    stem.tree_x = sx
    stem.tree_y = sy
    if sx is None or sy is None or ex is None or ey is None:
        stem.direction_x = 0.0
        stem.direction_y = 0.0
        stem.direction_deg = None
        return stem
    dx = float(ex) - float(sx)
    dy = float(ey) - float(sy)
    norm = math.hypot(dx, dy)
    if norm <= epsilon:
        stem.direction_x = 0.0
        stem.direction_y = 0.0
        stem.direction_deg = None
        return stem
    stem.direction_x = dx / norm
    stem.direction_y = dy / norm
    stem.direction_deg = math.degrees(math.atan2(stem.direction_y,
                                                 stem.direction_x))
    return stem


def reverse_stem_profile(stem: Stem):
    coords = list(stem.path.coords)
    if not coords:
        return stem
    rev_coords = list(reversed(coords))
    rev_d = list(reversed(list(getattr(stem, 'segment_diameter_list', []))))
    rev_l = list(reversed(list(getattr(stem, 'segment_length_list', []))))
    rev_v = list(reversed(list(getattr(stem, 'segment_volume_list', []))))
    rev_vec = list(reversed(list(getattr(stem, 'vector', []))))
    out = Stem(
        start=Point(rev_coords[0]),
        stop=Point(rev_coords[-1]),
        path=LineString(rev_coords),
        vector=rev_vec,
        segment_diameter_list=rev_d,
        segment_length_list=rev_l,
        segment_volume_list=rev_v,
        stem_id=getattr(stem, 'stem_id', None),
        crs=getattr(stem, 'crs', None),
        tree_x=getattr(stem, 'tree_x', None),
        tree_y=getattr(stem, 'tree_y', None),
        direction_x=-float(getattr(stem, 'direction_x', 0.0) or 0.0),
        direction_y=-float(getattr(stem, 'direction_y', 0.0) or 0.0),
        direction_deg=(None if getattr(stem, 'direction_deg', None) is None else
                       float(getattr(stem, 'direction_deg')) + 180.0),
        direction_confidence=float(getattr(stem, 'direction_confidence', 0.0) or 0.0),
        owner_partition_id=getattr(stem, 'owner_partition_id', None),
        source_tile_id=getattr(stem, 'source_tile_id', None),
        is_border_candidate=bool(getattr(stem, 'is_border_candidate', False)),
    )
    for attr in ('_contributors',):
        if hasattr(stem, attr):
            setattr(out, attr, getattr(stem, attr))
    return _direction_metadata_from_coords(out)


def refresh_measurement_vectors(stem: Stem, config=None):
    coords = list(stem.path.coords)
    if not coords:
        stem.vector = []
        return stem
    default_half = float(getattr(config, 'diameter_vector_half_length_m', 1.0)) \
        if config is not None else 1.0
    diams = list(getattr(stem, 'segment_diameter_list', []))
    if len(diams) < len(coords):
        fill = diams[-1] if diams else 0.0
        diams = diams + [fill] * (len(coords) - len(diams))
    elif len(diams) > len(coords):
        diams = diams[:len(coords)]
    stem.vector = []
    for i, xy in enumerate(coords):
        diameter = diams[i] if i < len(diams) else 0.0
        half_len = max(default_half, max(0.0, float(diameter)) / 2.0)
        stem.vector.append(_measurement_vector(xy, _local_normal(coords, i),
                                               half_len))
    return stem


def refresh_stem_direction(stem: Stem, config=None):
    coords = list(stem.path.coords)
    diameters = list(getattr(stem, 'segment_diameter_list', []))
    if not coords:
        stem.direction_confidence = 0.0
        return stem
    if len(diameters) < len(coords):
        fill = diameters[-1] if diameters else 0.0
        diameters = diameters + [fill] * (len(coords) - len(diameters))
    elif len(diameters) > len(coords):
        diameters = diameters[:len(coords)]

    confidence = 0.0
    reverse_needed = False
    if len(coords) >= 2 and len(diameters) >= 2 and any(abs(float(d)) > epsilon for d in diameters):
        dist = _cumulative_distances(coords)
        if len(dist) >= 2 and dist[-1] > epsilon:
            x = dist / max(dist[-1], epsilon)
            y = np.asarray(diameters, dtype=float)
            if len(y) >= 2:
                if len(y) == 2:
                    total_drop = y[-1] - y[0]
                    corr = 1.0
                else:
                    try:
                        slope, _ = np.polyfit(x, y, 1)
                        total_drop = float(slope) * float(x[-1] - x[0])
                    except Exception:
                        total_drop = float(y[-1] - y[0])
                    try:
                        corr = float(np.corrcoef(x, y)[0, 1])
                        if not np.isfinite(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0
                mean_d = float(np.nanmean(np.abs(y)))
                norm_drop = abs(float(total_drop)) / max(mean_d, epsilon)
                confidence = float(min(1.0, max(0.0, norm_drop * max(abs(corr), 0.35))))
                reverse_needed = total_drop > 0.0
    stem.direction_confidence = confidence
    if reverse_needed:
        stem = reverse_stem_profile(stem)
        stem.direction_confidence = confidence
    stem = _direction_metadata_from_coords(stem)
    return stem


def is_direction_ambiguous(stem: Stem, config=None) -> bool:
    return float(getattr(stem, 'direction_confidence', 0.0) or 0.0) \
        < _direction_confidence_threshold(config)


def orient_stem_along_path(stem: Stem, merged_path):
    try:
        s_proj = merged_path.project(Point(stem.start.coords[0]))
        e_proj = merged_path.project(Point(stem.stop.coords[0]))
    except Exception:
        return stem
    if s_proj <= e_proj:
        return stem
    return reverse_stem_profile(stem)


def rebuild_connected_stem_profile(merged_stem: Stem, contributors, config=None,
                                   compute_volume=False):
    contrib_stems = []
    for item in contributors or []:
        if isinstance(item, tuple) and len(item) >= 2:
            contrib_stems.append(item[1])
        else:
            contrib_stems.append(item)
    if not contrib_stems:
        stem = Stem(
            start=Point(merged_stem.start.coords[0]),
            stop=Point(merged_stem.stop.coords[0]),
            path=LineString(list(merged_stem.path.coords)),
            vector=list(getattr(merged_stem, 'vector', [])),
            segment_diameter_list=list(getattr(merged_stem, 'segment_diameter_list', [])),
            segment_length_list=list(getattr(merged_stem, 'segment_length_list', [])),
            segment_volume_list=list(getattr(merged_stem, 'segment_volume_list', [])),
            stem_id=getattr(merged_stem, 'stem_id', None),
            crs=getattr(merged_stem, 'crs', None),
            direction_confidence=float(getattr(merged_stem, 'direction_confidence', 0.0) or 0.0),
            owner_partition_id=getattr(merged_stem, 'owner_partition_id', None),
            source_tile_id=getattr(merged_stem, 'source_tile_id', None),
            is_border_candidate=bool(getattr(merged_stem, 'is_border_candidate', False)),
        )
        stem = refresh_stem_direction(stem, config=config)
        stem = refresh_measurement_vectors(stem, config=config)
        if compute_volume:
            stem = quantify_stem(stem)
        return stem

    oriented = []
    for stem in contrib_stems:
        s = orient_stem_along_path(stem, merged_stem.path)
        try:
            start_proj = merged_stem.path.project(Point(s.start.coords[0]))
            stop_proj = merged_stem.path.project(Point(s.stop.coords[0]))
            pos = min(start_proj, stop_proj)
        except Exception:
            pos = 0.0
        oriented.append((pos, s))
    oriented.sort(key=lambda item: item[0])

    coords = []
    diameters = []
    for _, stem in oriented:
        stem_coords = list(stem.path.coords)
        stem_d = list(getattr(stem, 'segment_diameter_list', []))
        if not stem_coords:
            continue
        if not stem_d:
            stem_d = [0.0] * len(stem_coords)
        if len(stem_d) < len(stem_coords):
            fill = stem_d[-1] if stem_d else 0.0
            stem_d = stem_d + [fill] * (len(stem_coords) - len(stem_d))
        elif len(stem_d) > len(stem_coords):
            stem_d = stem_d[:len(stem_coords)]

        if not coords:
            coords = stem_coords[:]
            diameters = stem_d[:]
            continue

        if tuple(coords[-1]) == tuple(stem_coords[0]):
            coords.extend(stem_coords[1:])
            diameters.extend(stem_d[1:])
        else:
            coords.extend(stem_coords)
            diameters.extend(stem_d)

    if len(coords) < 2:
        coords = list(merged_stem.path.coords)
    if len(diameters) < len(coords):
        fill = diameters[-1] if diameters else 0.0
        diameters = diameters + [fill] * (len(coords) - len(diameters))
    elif len(diameters) > len(coords):
        diameters = diameters[:len(coords)]

    rebuilt = Stem(
        start=Point(coords[0]),
        stop=Point(coords[-1]),
        path=LineString(coords),
        vector=[],
        segment_diameter_list=diameters,
        segment_length_list=[],
        segment_volume_list=[],
        stem_id=getattr(merged_stem, 'stem_id', None),
        crs=getattr(merged_stem, 'crs', None),
        direction_confidence=0.0,
        owner_partition_id=getattr(merged_stem, 'owner_partition_id', None),
        source_tile_id=getattr(merged_stem, 'source_tile_id', None),
        is_border_candidate=bool(getattr(merged_stem, 'is_border_candidate', False)),
    )
    rebuilt = refresh_stem_direction(rebuilt, config=config)
    rebuilt = refresh_measurement_vectors(rebuilt, config=config)
    if compute_volume:
        rebuilt = quantify_stem(rebuilt)
    return rebuilt


def calc_v_d_contour(stem, contours, config=None):
    coords = list(stem.path.coords)
    half_len = (
        float(getattr(config, 'diameter_vector_half_length_m', 1.0))
        if config is not None else 1.0
    )
    sample_indices = _diameter_sample_indices(len(coords))
    diameters = []

    for i in sample_indices:
        xy = coords[i]
        normal = _local_normal(coords, i)
        vector = _measurement_vector(xy, normal, half_len)
        diameters.append(calc_d(xy, vector, contours))

    stem.segment_diameter_list = _restore_endpoint_diameters(
        diameters, len(coords))
    stem.vector = []
    for i, xy in enumerate(coords):
        normal = _local_normal(coords, i)
        vector = _measurement_vector(xy, normal, half_len)
        stem.vector.append(vector)
    return stem

def _distance_transform_m(pred_bin, profile):
    px, py = _pixel_size(profile)
    return ndi.distance_transform_edt(pred_bin.astype(bool),
                                      sampling=(py, px))


def _xy_to_rowcol(x, y, profile) -> Tuple[int, int]:
    transform = profile['transform']
    inv = ~transform
    col, row = inv * (x, y)
    return int(round(row)), int(round(col))


def calc_v_d_edt(stem, edt_map, profile, config=None):
    coords = list(stem.path.coords)
    default_half = (
        float(getattr(config, 'diameter_vector_half_length_m', 1.0))
        if config is not None else 1.0
    )
    clip_max = (
        getattr(config, 'edt_clip_max_m', None)
        if config is not None else None
    )
    sample_indices = _diameter_sample_indices(len(coords))
    diameters = []
    radii_by_index = {}

    h, w = edt_map.shape
    for i in sample_indices:
        xy = coords[i]
        row, col = _xy_to_rowcol(xy[0], xy[1], profile)
        if row < 0 or row >= h or col < 0 or col >= w:
            radius = 0.0
        else:
            radius = float(edt_map[row, col])
            if clip_max is not None:
                radius = min(radius, float(clip_max))
        radii_by_index[i] = radius
        diameters.append(max(0.0, 2.0 * radius))

    stem.segment_diameter_list = _restore_endpoint_diameters(
        diameters, len(coords))
    stem.vector = []
    for i, xy in enumerate(coords):
        radius = radii_by_index.get(i, 0.5 * stem.segment_diameter_list[i])
        normal = _local_normal(coords, i)
        half_len = max(default_half, radius)
        stem.vector.append(_measurement_vector(xy, normal, half_len))
    return stem

# Backward-compatible name
calc_v_d = calc_v_d_contour


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
                        d = max(d, i_.length)
            else:
                d = max(d, i.length)
    return d


def calc_l_v(p1, p2, d1, d2):
    length = math.dist(p1, p2)
    v = 1 / 3 * math.pi * (
        (d1 / 2) ** 2 + (d1 / 2) * (d2 / 2) + (d2 / 2) ** 2
    ) * length
    return length, v
