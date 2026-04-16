#!/usr/bin/env python

################################################################################
"""Clean vector operations for stem connection.

This version implements:
- explicit endpoint-in-buffer candidate generation only
- all 4 endpoint cases:
    start->start, start->end, end->start, end->end
- one hard local antiparallel filter using:
    v_stem               [base endpoint -> base interior]
    v_stem_bridge        [base endpoint -> candidate endpoint]
    v_candidate_bridge   [candidate endpoint -> base endpoint]
    v_candidate          [candidate endpoint -> candidate interior]
  with all three angles tested near 180 degrees using:
    tolerance_angle * dist_f
    dist_f = 1 - (1 / (3 + max_distance - gap)) ** 0.5
- unchanged vote function
- best non-conflicting matches accepted per cycle
- no self-connections
"""

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

# System epsilon
epsilon = np.finfo(float).eps


def _worker_count(config=None):
    proc = mp.current_process()
    if proc.name != "MainProcess":
        return 1

    value = getattr(config, 'cpu_workers', None) if config is not None else None
    if value is None:
        value = max(mp.cpu_count() - 1, 1)
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _clone_stem(stem: Stem) -> Stem:
    return Stem(
        start=Point(stem.start.coords[0]),
        stop=Point(stem.stop.coords[0]),
        path=LineString(list(stem.path.coords)),
        vector=list(getattr(stem, 'vector', [])),
        segment_diameter_list=list(getattr(stem, 'segment_diameter_list', [])),
        segment_length_list=list(getattr(stem, 'segment_length_list', [])),
        segment_volume_list=list(getattr(stem, 'segment_volume_list', [])),
        crs=getattr(stem, 'crs', None),
    )


def _same_stem_identity(a: Stem, b: Stem) -> bool:
    try:
        if a is b:
            return True
        a_coords = list(a.path.coords)
        b_coords = list(b.path.coords)
        if a_coords == b_coords:
            return True
        if a_coords == list(reversed(b_coords)):
            return True
        if a.start.equals(b.start) and a.stop.equals(b.stop) and a.path.equals(b.path):
            return True
        if a.start.equals(b.stop) and a.stop.equals(b.start):
            try:
                if a.path.equals(LineString(list(reversed(b.path.coords)))):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _endpoint_in_buffer(stem: Stem, endpoint_name: str, buffer_geom) -> bool:
    point = stem.start if endpoint_name == 'start' else stem.stop
    try:
        return buffer_geom.covers(point)
    except Exception:
        return buffer_geom.contains(point)


def _distance_factor(gap: float, max_distance: float) -> float:
    denom = max(epsilon, 3.0 + max_distance - gap)
    return 1.0 - (1.0 / denom) ** 0.5


def _safe_linestring(coords):
    coords = list(coords)
    if len(coords) >= 2:
        for i in range(1, len(coords)):
            if coords[i] != coords[0]:
                return LineString(coords)
    if not coords:
        coords = [(0.0, 0.0), (0.0, 0.0)]
    elif len(coords) == 1:
        coords = [coords[0], coords[0]]
    else:
        coords = [coords[0], coords[0]]
    return LineString(coords)


def _first_distinct_from_start(coords):
    p0 = coords[0]
    for i in range(1, len(coords)):
        if coords[i] != p0:
            return coords[i]
    return coords[-1]


def _first_distinct_from_end(coords):
    p0 = coords[-1]
    for i in range(len(coords) - 2, -1, -1):
        if coords[i] != p0:
            return coords[i]
    return coords[0]


def _stem_endpoint_vector(stem: Stem, endpoint_name: str) -> LineString:
    coords = list(stem.path.coords)
    if len(coords) < 2:
        p0 = stem.start.coords[0]
        p1 = stem.stop.coords[0]
        return _safe_linestring([p0, p1])

    if endpoint_name == 'start':
        p0 = coords[0]
        p1 = _first_distinct_from_start(coords)
    else:
        p0 = coords[-1]
        p1 = _first_distinct_from_end(coords)
    return _safe_linestring([p0, p1])


def _orient_stem_for_base(stem: Stem, endpoint_name: str) -> Stem:
    """Orient base stem so the tested endpoint becomes stop."""
    coords = list(stem.path.coords)
    if endpoint_name == 'start':
        coords = list(reversed(coords))
    oriented = _clone_stem(stem)
    oriented.path = _safe_linestring(coords)
    oriented.start = Point(coords[0])
    oriented.stop = Point(coords[-1])
    return oriented


def _orient_stem_for_candidate(stem: Stem, endpoint_name: str) -> Stem:
    """Orient candidate stem so the tested endpoint becomes start."""
    coords = list(stem.path.coords)
    if endpoint_name in ('end', 'stop'):
        coords = list(reversed(coords))
    oriented = _clone_stem(stem)
    oriented.path = _safe_linestring(coords)
    oriented.start = Point(coords[0])
    oriented.stop = Point(coords[-1])
    return oriented


def _merged_candidate_from_oriented(base_stem: Stem, candidate_stem: Stem) -> Stem:
    base_coords = list(base_stem.path.coords)
    cand_coords = list(candidate_stem.path.coords)

    if not base_coords:
        coords = cand_coords
    elif not cand_coords:
        coords = base_coords
    elif base_coords[-1] == cand_coords[0]:
        coords = base_coords + cand_coords[1:]
    else:
        coords = base_coords + cand_coords

    new_path = _safe_linestring(coords)
    merged = _clone_stem(base_stem)
    merged.path = new_path
    merged.start = Point(new_path.coords[0])
    merged.stop = Point(new_path.coords[-1])
    return merged


def _passes_endpoint_gate(
    base_oriented: Stem,
    candidate_oriented: Stem,
    max_distance: float,
    tolerance_angle: float,
):
    """Hard local antiparallel gate for one oriented endpoint pair.

    Assumes:
    - tested base endpoint == base_oriented.stop
    - tested candidate endpoint == candidate_oriented.start
    """
    if _same_stem_identity(base_oriented, candidate_oriented):
        return False, math.inf, None, None

    gap = base_oriented.stop.distance(candidate_oriented.start)
    if gap > max_distance:
        return False, math.inf, None, None

    dist_f = _distance_factor(gap, max_distance)
    tolerance_gate = tolerance_angle * dist_f

    v_stem = _stem_endpoint_vector(base_oriented, 'end')
    v_candidate = _stem_endpoint_vector(candidate_oriented, 'start')

    base_endpoint = base_oriented.stop.coords[0]
    cand_endpoint = candidate_oriented.start.coords[0]
    v_stem_bridge = _safe_linestring([base_endpoint, cand_endpoint])
    v_candidate_bridge = _safe_linestring([cand_endpoint, base_endpoint])

    ang_stem_bridge = abs(180.0 - ang(v_stem.coords, v_stem_bridge.coords))
    ang_candidate_bridge = abs(180.0 - ang(v_candidate_bridge.coords, v_candidate.coords))
    ang_stem_candidate = abs(180.0 - ang(v_stem.coords, v_candidate.coords))

    if not (
        ang_stem_bridge < tolerance_gate
        and ang_candidate_bridge < tolerance_gate
        and ang_stem_candidate < tolerance_gate
    ):
        return False, math.inf, None, None

    merged = _merged_candidate_from_oriented(base_oriented, candidate_oriented)
    vote = calc_vote(
        ang_stem_candidate,
        ang_stem_bridge,
        ang_candidate_bridge,
        merged,
        candidate_oriented,
        base_oriented,
        tolerance_angle,
    )
    return True, vote, merged, gap


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


################################################################################
"""Vector operations"""


def connect_stems(stems: List[Stem], config) -> List[Stem]:
    max_distance = config.max_distance
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

        proposals = []

        for base_idx in list(remaining):
            base_stem = cycle_stems[base_idx]
            start_buffer = base_stem.start.buffer(max_distance, resolution=32)
            end_buffer = base_stem.stop.buffer(max_distance, resolution=32)

            explicit_cases = []

            # Base start buffer: only explicit endpoints inside this buffer.
            for idx in _query_tree_indices(start_tree, start_buffer, start_points):
                if idx == base_idx or idx not in remaining:
                    continue
                candidate = cycle_stems[idx]
                if _same_stem_identity(base_stem, candidate):
                    continue
                if _endpoint_in_buffer(candidate, 'start', start_buffer):
                    explicit_cases.append((idx, 'start', 'start'))

            for idx in _query_tree_indices(stop_tree, start_buffer, stop_points):
                if idx == base_idx or idx not in remaining:
                    continue
                candidate = cycle_stems[idx]
                if _same_stem_identity(base_stem, candidate):
                    continue
                if _endpoint_in_buffer(candidate, 'end', start_buffer):
                    explicit_cases.append((idx, 'start', 'end'))

            # Base end buffer: only explicit endpoints inside this buffer.
            for idx in _query_tree_indices(start_tree, end_buffer, start_points):
                if idx == base_idx or idx not in remaining:
                    continue
                candidate = cycle_stems[idx]
                if _same_stem_identity(base_stem, candidate):
                    continue
                if _endpoint_in_buffer(candidate, 'start', end_buffer):
                    explicit_cases.append((idx, 'end', 'start'))

            for idx in _query_tree_indices(stop_tree, end_buffer, stop_points):
                if idx == base_idx or idx not in remaining:
                    continue
                candidate = cycle_stems[idx]
                if _same_stem_identity(base_stem, candidate):
                    continue
                if _endpoint_in_buffer(candidate, 'end', end_buffer):
                    explicit_cases.append((idx, 'end', 'end'))

            # De-duplicate exact endpoint proposals while preserving order.
            seen = set()
            filtered_cases = []
            for item in explicit_cases:
                if item in seen:
                    continue
                seen.add(item)
                filtered_cases.append(item)

            for idx, base_ep, cand_ep in filtered_cases:
                candidate = cycle_stems[idx]
                base_oriented = _orient_stem_for_base(base_stem, base_ep)
                candidate_oriented = _orient_stem_for_candidate(candidate, cand_ep)

                if _same_stem_identity(base_oriented, candidate_oriented):
                    continue

                changed, vote, merged, _ = calc_connectivity_votes(
                    base_oriented,
                    None,
                    None,
                    None,
                    None,
                    max_distance,
                    getattr(config, 'max_tree_height', math.inf),
                    tolerance_angle,
                    candidate_oriented,
                )
                if changed:
                    proposals.append({
                        'base_idx': base_idx,
                        'cand_idx': idx,
                        'base_ep': base_ep,
                        'cand_ep': cand_ep,
                        'vote': vote,
                        'merged': merged,
                    })

        proposals.sort(key=lambda x: x['vote'])

        used_stems = set()
        used_endpoints = set()
        connected_stems = []

        for proposal in proposals:
            base_idx = proposal['base_idx']
            cand_idx = proposal['cand_idx']
            base_ep = proposal['base_ep']
            cand_ep = proposal['cand_ep']

            if base_idx in used_stems or cand_idx in used_stems:
                continue
            if (base_idx, base_ep) in used_endpoints:
                continue
            if (cand_idx, cand_ep) in used_endpoints:
                continue

            connected_stems.append(proposal['merged'])
            used_stems.add(base_idx)
            used_stems.add(cand_idx)
            used_endpoints.add((base_idx, base_ep))
            used_endpoints.add((cand_idx, cand_ep))
            global_change = True
            c_count += 1

        for idx, stem in enumerate(cycle_stems):
            if idx not in used_stems:
                connected_stems.append(stem)

        stems = connected_stems
        cycle_nbr += 1

    connected_stems = []
    for stem in stems:
        if stem.length > config.min_length:
            connected_stems.append(stem)
        else:
            out_count += 1
    connected_stems, dup_count_2 = remove_duplicates(connected_stems)
    duplicates_count += dup_count_2

    print("")
    print(count_stem_parts, "stem segments analyzed")
    print(c_count, "stem segments appended to other stems")
    print(duplicates_count, "duplicates are removed")
    print(out_count, "stem fragments with a length less than ",
          config.min_length, "m are filtered out")
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
        stem: Stem
) -> (bool, List[float], List[Stem], List[Stem]):
    if stem == stems0:
        return False, math.inf, None, None

    changed, vote, merged, _ = _passes_endpoint_gate(
        stems0,
        stem,
        max_distance,
        tolerance_angle,
    )
    if changed:
        return True, vote, merged, stem
    return False, math.inf, None, None


# calculate vote (unchanged)
def calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st, candidate, stem,
              stems0, tolerance_angle):
    return (
        ((1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / tolerance_angle) *
        candidate.start.distance(candidate.stop) ** 2
        + stems0.stop.distance(stem.start) ** 2 *
        (1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / tolerance_angle
    )


# - Helper functions vector operations -
def build_stem_parts(segments: List[Part]):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Build stem segments")
    stems = []
    for i in range(len(segments)):
        if segments[i].start[1] >= segments[i].stop[1]:
            h = segments[i].start
            segments[i].start = segments[i].stop
            segments[i].stop = h
            segments[i].path.reverse()
    segments = set(segments)
    for seg in segments:
        stem = Stem(Point(seg.start), Point(seg.stop), LineString(seg.path), [],
                    [], [], [])
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
