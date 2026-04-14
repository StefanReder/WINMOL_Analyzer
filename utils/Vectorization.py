#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import List, Sequence, Set

import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from classes.Part import Part
from classes.Stem import Stem
from classes.Timer import Timer
from utils.Geometry import ang
from utils.IO import get_bounds_from_profile
import utils.Quantification as Quant

# System epsilon
epsilon = np.finfo(float).eps


def _worker_count(config=None):
    proc = mp.current_process()
    if proc.name != "MainProcess":
        return 1

    value = getattr(config, 'cpu_workers', None) \
        if config is not None else None
    if value is None:
        value = max(mp.cpu_count() - 1, 1)
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _clone_stem(stem: Stem) -> Stem:
    cloned = Stem(
        start=Point(stem.start.coords[0]),
        stop=Point(stem.stop.coords[0]),
        path=LineString(list(stem.path.coords)),
        vector=list(getattr(stem, 'vector', [])),
        segment_diameter_list=list(getattr(stem, 'segment_diameter_list', [])),
        segment_length_list=list(getattr(stem, 'segment_length_list', [])),
        segment_volume_list=list(getattr(stem, 'segment_volume_list', [])),
        stem_id=getattr(stem, 'stem_id', None),
        crs=getattr(stem, 'crs', None),
        tree_x=getattr(stem, 'tree_x', None),
        tree_y=getattr(stem, 'tree_y', None),
        direction_x=float(getattr(stem, 'direction_x', 0.0) or 0.0),
        direction_y=float(getattr(stem, 'direction_y', 0.0) or 0.0),
        direction_deg=getattr(stem, 'direction_deg', None),
        direction_confidence=float(getattr(stem, 'direction_confidence', 0.0) or 0.0),
        owner_partition_id=getattr(stem, 'owner_partition_id', None),
        source_tile_id=getattr(stem, 'source_tile_id', None),
        is_border_candidate=bool(getattr(stem, 'is_border_candidate', False)),
    )
    if hasattr(stem, '_contributors'):
        cloned._contributors = list(getattr(stem, '_contributors', []))
    return cloned


def _direction_variants(stem: Stem, config=None):
    primary = _clone_stem(stem)
    conf = float(getattr(stem, 'direction_confidence', 0.0) or 0.0)
    try:
        low = float(getattr(config, 'direction_confidence_low', 0.35))
    except Exception:
        low = 0.35
    if conf <= low or Quant.is_direction_ambiguous(stem, config=config):
        return [primary, Quant.reverse_stem_profile(_clone_stem(stem))]
    return [primary]


def _contributors_of(stem: Stem):
    contributors = getattr(stem, '_contributors', None)
    if contributors:
        return list(contributors)
    base = _clone_stem(stem)
    if hasattr(base, '_contributors'):
        delattr(base, '_contributors')
    return [base]


def _set_contributors(stem: Stem, contributors):
    stem._contributors = list(contributors or [])
    return stem


def _merged_candidate_stub(base_stem: Stem, candidate_stem: Stem, new_path: LineString):
    merged = Stem(
        start=Point(list(new_path.coords)[0]),
        stop=Point(list(new_path.coords)[-1]),
        path=LineString(list(new_path.coords)),
        vector=[],
        segment_diameter_list=[],
        segment_length_list=[],
        segment_volume_list=[],
        stem_id=None,
        crs=getattr(base_stem, 'crs', getattr(candidate_stem, 'crs', None)),
        source_tile_id=getattr(base_stem, 'source_tile_id', getattr(candidate_stem, 'source_tile_id', None)),
    )
    contributors = _contributors_of(base_stem) + _contributors_of(candidate_stem)
    merged = Quant.rebuild_connected_stem_profile(
        merged, contributors, config=None, compute_volume=False)
    merged = _set_contributors(merged, contributors)
    return merged


def _build_join_path(base_stem: Stem, candidate_stem: Stem):
    coords_a = list(base_stem.path.coords)
    coords_b = list(candidate_stem.path.coords)
    if len(coords_a) < 2 or len(coords_b) < 2:
        return None
    merged = list(coords_a)
    if tuple(coords_a[-1]) == tuple(coords_b[0]):
        merged.extend(coords_b[1:])
    else:
        merged.extend(coords_b)
    if len(merged) < 2:
        return None
    return LineString(merged)


def _stem_end_lines(stem: Stem):
    if len(stem.path.coords) < 4:
        line_start = LineString([stem.path.coords[0], stem.path.coords[-1]])
        line_stop = LineString([stem.path.coords[0], stem.path.coords[-1]])
    else:
        i = len(stem.path.coords) - 2 if len(stem.path.coords) < 8 else 6
        line_start = LineString([stem.path.coords[1], stem.path.coords[i]])
        line_stop = LineString([stem.path.coords[-(i + 1)],
                                stem.path.coords[-2]])
    return line_start, line_stop


def _query_tree_indices(tree: STRtree, geom, fallback_geoms=None):
    try:
        matches = tree.query(geom)
    except Exception:
        return []
    if len(matches) == 0:
        return []
    first = matches[0]
    if isinstance(first, (int, np.integer)):
        return [int(i) for i in matches]
    if fallback_geoms is None:
        return []
    geom_to_idx = {id(g): i for i, g in enumerate(fallback_geoms)}
    return [geom_to_idx[id(g)] for g in matches if id(g) in geom_to_idx]


def _remove_duplicates_against_base(
    cycle_stems: Sequence[Stem], remaining: Set[int], base_idx: int) \
        -> tuple[Set[int], int]:
    if base_idx not in remaining:
        return remaining, 0
    base = cycle_stems[base_idx]
    buffer_geom = base.path.buffer(0.3)
    to_remove = set()
    for idx in remaining:
        if idx == base_idx:
            continue
        try:
            if buffer_geom.contains(cycle_stems[idx].path):
                to_remove.add(idx)
        except Exception:
            continue
    if to_remove:
        remaining = set(remaining)
        remaining.difference_update(to_remove)
    return remaining, len(to_remove)


################################################################################
"""Vector operations"""


# Spatial-index accelerated version of connect_stems
def connect_stems(stems: List[Stem], config) -> List[Stem]:
    max_distance = config.max_distance
    max_tree_height = config.max_tree_height
    tolerance_angle = config.tolerance_angle

    t = Timer()
    t.start()
    print("#######################################################")
    print("Gathering stem segments")

    cycle_nbr = 1
    c_count = 0
    out_count = 0
    duplicates_count = 0
    count_stem_parts = len(stems)
    global_change = True

    stems = [Quant.refresh_stem_direction(_clone_stem(stem), config=config)
             for stem in stems]

    while global_change:
        global_change = False
        print("Cycle ", cycle_nbr)
        cycle_stems = list(stems)
        if not cycle_stems:
            break

        start_points = [s.start for s in cycle_stems]
        stop_points = [s.stop for s in cycle_stems]
        start_tree = STRtree(start_points)
        stop_tree = STRtree(stop_points)
        remaining = set(range(len(cycle_stems)))
        connected_stems = []

        while remaining:
            base_idx = next(iter(remaining))
            base_stem = cycle_stems[base_idx]

            while True:
                line_start, line_stop = _stem_end_lines(base_stem)
                start_buffer = base_stem.start.buffer(max_distance, resolution=32)
                end_buffer = base_stem.stop.buffer(max_distance, resolution=32)

                candidate_indices = set(_query_tree_indices(stop_tree, start_buffer, stop_points))
                candidate_indices.update(_query_tree_indices(start_tree, start_buffer, start_points))
                candidate_indices.update(_query_tree_indices(stop_tree, end_buffer, stop_points))
                candidate_indices.update(_query_tree_indices(start_tree, end_buffer, start_points))
                candidate_indices.intersection_update(remaining)
                candidate_indices.discard(base_idx)

                filtered_indices = []
                for idx in candidate_indices:
                    candidate = cycle_stems[idx]
                    if (
                        start_buffer.contains(candidate.start)
                        or start_buffer.contains(candidate.stop)
                        or end_buffer.contains(candidate.start)
                        or end_buffer.contains(candidate.stop)
                    ):
                        filtered_indices.append(idx)

                best_vote = math.inf
                best_candidate = None
                best_slave_idx = None

                for idx in filtered_indices:
                    raw_candidate = cycle_stems[idx]
                    for base_variant in _direction_variants(base_stem, config=config):
                        lv_start, lv_stop = _stem_end_lines(base_variant)
                        sv_start_buffer = base_variant.start.buffer(max_distance, resolution=32)
                        sv_end_buffer = base_variant.stop.buffer(max_distance, resolution=32)
                        for cand_variant in _direction_variants(raw_candidate, config=config):
                            changed, vote, candidate_stem, _ = calc_connectivity_votes(
                                base_variant,
                                lv_start,
                                lv_stop,
                                sv_start_buffer,
                                sv_end_buffer,
                                max_distance,
                                max_tree_height,
                                tolerance_angle,
                                cand_variant,
                                config=config,
                            )
                            if changed and vote < best_vote:
                                best_vote = vote
                                best_candidate = candidate_stem
                                best_slave_idx = idx

                if best_candidate is not None and best_slave_idx is not None:
                    base_stem = Quant.refresh_stem_direction(best_candidate, config=config)
                    base_stem = Quant.refresh_measurement_vectors(base_stem, config=config)
                    cycle_stems[base_idx] = base_stem
                    remaining.discard(best_slave_idx)
                    global_change = True
                    c_count += 1
                    remaining, dup_removed = _remove_duplicates_against_base(
                        cycle_stems, remaining, base_idx)
                    duplicates_count += dup_removed
                    continue

                connected_stems.append(base_stem)
                remaining.discard(base_idx)
                break

        stems = connected_stems
        cycle_nbr += 1

    connected_stems = []
    for stem in stems:
        stem = Quant.refresh_stem_direction(stem, config=config)
        stem = Quant.refresh_measurement_vectors(stem, config=config)
        if stem.length > config.min_length:
            connected_stems.append(stem)
        else:
            out_count += 1
    connected_stems, dup_count_2 = remove_duplicates_spatial(connected_stems, config=config)
    duplicates_count += dup_count_2

    print("")
    print(count_stem_parts, "stem segments analyzed")
    print(c_count, "stem segments appended to other stems")
    print(duplicates_count, "duplicates are removed")
    print(out_count, "stem fragments with a length less than ",
          config.min_length, "m are filtered out")
    connected_stems = Quant.ensure_stem_ids(connected_stems, prefix="stem")
    print("final number of stems", len(connected_stems))
    t.stop()
    print("#######################################################")
    print("")
    return connected_stems


def calc_connectivity_votes(
        stems0: Stem,
        line_start: LineString,
        line_stop: LineString,
        start_buffer,
        end_buffer,
        max_distance,
        max_tree_height,
        tolerance_angle,
        stem: Stem,
        config=None,
) -> (bool, List[float], List[Stem], List[Stem]):
    if stem == stems0:
        return False, math.inf, None, None

    change = False
    votes = []
    candidates = []
    slaves = []

    if len(stem.path.coords) < 4:
        e_line_start = LineString([stem.path.coords[0], stem.path.coords[-1]])
        e_line_stop = LineString([stem.path.coords[0], stem.path.coords[-1]])
    else:
        k = len(stem.path.coords) - 2 if len(stem.path.coords) < 8 else 6
        e_line_start = LineString([stem.path.coords[1], stem.path.coords[k]])
        e_line_stop = LineString([stem.path.coords[-(k + 1)], stem.path.coords[-2]])

    ang_l_sp_el_st = abs(ang(line_stop.coords, e_line_start.coords))
    ang_el_sp_l_st = abs(ang(e_line_stop.coords, line_start.coords))

    gap_forward = stems0.stop.distance(stem.start)
    if end_buffer.contains(stem.start) and ang_l_sp_el_st <= tolerance_angle:
        bridge = LineString([stems0.path.coords[-2], stem.path.coords[1]]) if (
            len(stems0.path.coords) > 1 and len(stem.path.coords) > 1
        ) else LineString([stems0.stop.coords[0], stem.start.coords[0]])
        ang_l_sp_mp = abs(ang(line_stop.coords, bridge.coords))
        ang_mp_el_st = abs(ang(bridge.coords, e_line_start.coords))
        new_path = _build_join_path(stems0, stem)
        if new_path is not None:
            merged_length = new_path.length
            merged_span = Point(new_path.coords[0]).distance(Point(new_path.coords[-1]))
            if (
                gap_forward <= max_distance
                and ang_l_sp_el_st <= tolerance_angle
                and ang_l_sp_mp <= tolerance_angle
                and ang_mp_el_st <= tolerance_angle
                and merged_length <= max_tree_height
                and merged_span <= max_tree_height
            ):
                change = True
                candidate = _merged_candidate_stub(stems0, stem, new_path)
                vote = calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st,
                                 candidate, stem, stems0, tolerance_angle)
                candidates.append(candidate)
                votes.append(vote)
                slaves.append(stem)

    gap_reverse = stem.stop.distance(stems0.start)
    if start_buffer.contains(stem.stop) and ang_el_sp_l_st <= tolerance_angle:
        bridge = LineString([stem.path.coords[-2], stems0.path.coords[1]]) if (
            len(stem.path.coords) > 1 and len(stems0.path.coords) > 1
        ) else LineString([stem.stop.coords[0], stems0.start.coords[0]])
        ang_el_sp_mp = abs(ang(e_line_stop.coords, bridge.coords))
        ang_mp_l_st = abs(ang(bridge.coords, line_start.coords))
        new_path = _build_join_path(stem, stems0)
        if new_path is not None:
            merged_length = new_path.length
            merged_span = Point(new_path.coords[0]).distance(Point(new_path.coords[-1]))
            if (
                gap_reverse <= max_distance
                and ang_el_sp_l_st <= tolerance_angle
                and ang_el_sp_mp <= tolerance_angle
                and ang_mp_l_st <= tolerance_angle
                and merged_length <= max_tree_height
                and merged_span <= max_tree_height
            ):
                change = True
                candidate = _merged_candidate_stub(stem, stems0, new_path)
                vote = calc_vote(ang_el_sp_l_st, ang_el_sp_mp, ang_mp_l_st,
                                 candidate, stems0, stem, tolerance_angle)
                candidates.append(candidate)
                votes.append(vote)
                slaves.append(stem)

    if change:
        index_min = min(range(len(votes)), key=votes.__getitem__)
        return True, votes[index_min], candidates[index_min], slaves[index_min]
    return False, math.inf, None, None


# calculate vote
def calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st, candidate, stem,
              stems0, tolerance_angle):
    return (
        ((1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / max(tolerance_angle, epsilon)) *
        candidate.start.distance(candidate.stop) ** 2
        + stems0.stop.distance(stem.start) ** 2 *
        (1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / max(tolerance_angle, epsilon)
    )


# - Helper functions vector operations -
# Converts the List of [Part] containing Tuples[int] into List of [Stem]
# consisting of shapely geometries
def build_stem_parts(segments: List[Part]):

    t = Timer()
    t.start()
    print("#######################################################")
    print("Build stem segments")
    stems = []
    segments = set(segments)
    for seg in segments:
        path_coords = list(seg.path)
        if len(path_coords) < 2:
            continue
        stem = Stem(
            start=Point(path_coords[0]),
            stop=Point(path_coords[-1]),
            path=LineString(path_coords),
            vector=[],
            segment_diameter_list=[],
            segment_length_list=[],
            segment_volume_list=[],
            stem_id=f"part_{len(stems):08d}",
        )
        stems.append(stem)

    print(len(stems), "stems segments build")

    t.stop()
    print("#######################################################")
    print("")
    return stems


def rebuild_endnodes_from_stems(stems: List[Stem]) -> List[Point]:
    t = Timer()
    t.start()
    print("#######################################################")
    print("Rebuild endnodes from stems")
    nodes = []
    for s in stems:
        nodes.append(s.start.coords)
        nodes.append(s.stop.coords)
    t.stop()
    print("#######################################################")
    print("")
    return nodes


def remove_duplicates_spatial(stems: List[Stem], config=None, stems0=None):
    stems = list(stems)
    stems.sort(key=lambda x: x.length, reverse=True)
    if not stems:
        return [], 0
    try:
        buffer_m = float(getattr(config, 'partition_dedup_buffer_m', 0.02))
    except Exception:
        buffer_m = 0.02
    buffer_m = max(buffer_m, 1e-6)

    if type(stems0) is Stem:
        try:
            base_geom = stems0.path.buffer(buffer_m)
        except Exception:
            return remove_duplicates(stems, stems0=stems0)
        kept = []
        count = 0
        for s in stems:
            try:
                if base_geom.contains(s.path):
                    count += 1
                    continue
            except Exception:
                pass
            kept.append(s)
        kept.append(stems0)
        return kept, count

    buffers = []
    for stem in stems:
        try:
            buffers.append(stem.path.buffer(buffer_m))
        except Exception:
            buffers.append(stem.path)
    tree = STRtree(buffers)
    remaining = set(range(len(stems)))
    kept = []
    count = 0
    while remaining:
        idx = min(remaining)
        remaining.remove(idx)
        base = stems[idx]
        base_buf = buffers[idx]
        kept.append(base)
        try:
            cand_idx = set(_query_tree_indices(tree, base_buf, buffers))
        except Exception:
            cand_idx = set()
        for j in cand_idx:
            if j not in remaining or j == idx:
                continue
            try:
                if base_buf.contains(stems[j].path):
                    remaining.remove(j)
                    count += 1
            except Exception:
                continue
    return kept, count


# Removes duplicates from stem list
def remove_duplicates(stems: List[Stem], stems0=None) -> List[Stem]:
    stems = list(stems)
    stems.sort(key=lambda x: x.length, reverse=True)
    count = 0

    if type(stems0) is Stem:
        kept = []
        buffer_geom = stems0.path.buffer(0.3)
        for s in stems:
            try:
                if buffer_geom.contains(s.path):
                    count += 1
                    continue
            except Exception:
                pass
            kept.append(s)
        kept.append(stems0)
        return kept, count

    kept = []
    while stems:
        base = stems.pop(0)
        buffer_geom = base.path.buffer(0.3)
        survivors = []
        for s in stems:
            try:
                if buffer_geom.contains(s.path):
                    count += 1
                    continue
            except Exception:
                pass
            survivors.append(s)
        kept.append(base)
        stems = survivors
    return kept, count


# remove padding and restore geoinformation of the stems
def restore_geoinformation(stems: List[Stem], config, profile):
    t = Timer()
    t.start()

    print("#######################################################")
    print("Restoring geoinformation")

    px_size_x = abs(profile['transform'][0])
    px_size_y = abs(profile['transform'][4])
    bounds = get_bounds_from_profile(profile)
    padding = int(config.max_tree_height / max(px_size_x, px_size_y)) + 1

    for j in range(len(stems)):
        stems[j].start = (
            bounds.left + (stems[j].start[1] - padding) * px_size_x,
            bounds.top - (stems[j].start[0] - padding) * px_size_y)
        stems[j].stop = (
            bounds.left + (stems[j].stop[1] - padding) * px_size_x,
            bounds.top - (stems[j].stop[0] - padding) * px_size_y)

        for k in range(len(stems[j].path)):
            stems[j].path[k] = (
                bounds.left + (stems[j].path[k][1] - padding) * px_size_x,
                bounds.top - (stems[j].path[k][0] - padding) * px_size_y
            )

    t.stop()
    print("#######################################################")
    print("")
    return stems
