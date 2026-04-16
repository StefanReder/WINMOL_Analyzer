#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import List, Sequence, Set

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
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

    value = getattr(config, 'cpu_workers', None) \
        if config is not None else None
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
        if list(a.path.coords) == list(b.path.coords):
            return True
        if a.start.equals(b.start) and a.stop.equals(b.stop) and a.path.equals(b.path):
            return True
        if a.start.equals(b.stop) and a.stop.equals(b.start) and a.path.equals(LineString(list(reversed(b.path.coords)))):
            return True
    except Exception:
        pass
    return False


def _stem_end_lines(stem: Stem):
    coords = list(stem.path.coords)
    n = len(coords)
    if n < 2:
        line = LineString([stem.start.coords[0], stem.stop.coords[0]])
        return line, line
    if n == 2:
        line_start = LineString([coords[0], coords[1]])
        line_stop = LineString([coords[-1], coords[0]])
        return line_start, line_stop

    start_idx = 2 if n > 2 else 1
    stop_idx = n - 3 if n > 2 else 0
    # Both endpoint vectors point away from the local junction point:
    # start -> interior, stop -> interior
    line_start = LineString([coords[0], coords[start_idx]])
    line_stop = LineString([coords[-1], coords[stop_idx]])
    return line_start, line_stop


def _distance_factor(gap: float, max_distance: float) -> float:
    denom = max(epsilon, 3.0 + max_distance - gap)
    return 1.0 - (1.0 / denom) ** 0.5


def _endpoint_in_buffer(stem: Stem, endpoint_name: str, buffer_geom) -> bool:
    point = stem.start if endpoint_name == 'start' else stem.stop
    return buffer_geom.contains(point)


def _orient_stem_for_base(stem: Stem, endpoint_name: str) -> Stem:
    coords = list(stem.path.coords)
    if endpoint_name == 'start':
        coords = list(reversed(coords))
    oriented = _clone_stem(stem)
    oriented.path = LineString(coords)
    oriented.start = Point(coords[0])
    oriented.stop = Point(coords[-1])
    return oriented


def _orient_stem_for_candidate(stem: Stem, endpoint_name: str) -> Stem:
    coords = list(stem.path.coords)
    if endpoint_name == 'end':
        coords = list(reversed(coords))
    oriented = _clone_stem(stem)
    oriented.path = LineString(coords)
    oriented.start = Point(coords[0])
    oriented.stop = Point(coords[-1])
    return oriented


def _safe_linestring(coords):
    coords = list(coords)
    if len(coords) < 2:
        p = coords[0]
        return LineString([p, p])
    return LineString(coords)


def _merged_candidate_from_oriented(base_stem: Stem, candidate_stem: Stem) -> Stem:
    base_coords = list(base_stem.path.coords)
    cand_coords = list(candidate_stem.path.coords)
    if not base_coords or not cand_coords:
        coords = base_coords or cand_coords
    elif base_coords[-1] == cand_coords[0]:
        coords = base_coords + cand_coords[1:]
    else:
        coords = base_coords + cand_coords
    if len(coords) < 2:
        coords = base_coords if len(base_coords) >= 2 else cand_coords
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
    max_tree_height: float,
    tolerance_angle: float,
):
    if _same_stem_identity(base_oriented, candidate_oriented):
        return False, math.inf, None, None
    line_start_base, line_stop_base = _stem_end_lines(base_oriented)
    line_start_cand, _ = _stem_end_lines(candidate_oriented)

    gap = base_oriented.stop.distance(candidate_oriented.start)
    if gap > max_distance:
        return False, math.inf, None, None

    dist_f = _distance_factor(gap, max_distance)
    tolerance_gate = tolerance_angle * dist_f

    bridge_at_base = LineString([
        base_oriented.stop.coords[0],
        candidate_oriented.start.coords[0],
    ])
    bridge_at_candidate = LineString([
        candidate_oriented.start.coords[0],
        base_oriented.stop.coords[0],
    ])

    ang_stem_stem = abs(ang(line_stop_base.coords, line_start_cand.coords))
    ang_base_bridge = abs(180.0 - ang(line_stop_base.coords, bridge_at_base.coords))
    ang_candidate_bridge = abs(
        180.0 - ang(line_start_cand.coords, bridge_at_candidate.coords)
    )

    if not (
        ang_stem_stem < tolerance_gate
        and ang_base_bridge < tolerance_gate
        and ang_candidate_bridge < tolerance_gate
    ):
        return False, math.inf, None, None

    merged = _merged_candidate_from_oriented(base_oriented, candidate_oriented)
    merged_length = merged.path.length
    merged_span = merged.start.distance(merged.stop)

    if merged_length > max_tree_height or merged_span > max_tree_height:
        return False, math.inf, None, None

    vote = calc_vote(
        ang_stem_stem,
        ang_base_bridge,
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


def _heading_to_north_deg(p0, p1) -> float:
    dr = float(p1[0] - p0[0])
    dc = float(p1[1] - p0[1])
    return math.degrees(math.atan2(dc, -dr)) % 360.0


def _circular_diff_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _is_antiparallel(a: float, b: float, tolerance_deg: float) -> bool:
    return abs(_circular_diff_deg(a, b) - 180.0) <= tolerance_deg


def _attachment_heading(coords, attachment_idx: int, retained_side: str) -> float:
    n = len(coords)
    if n < 2:
        return 0.0

    if retained_side == 'prefix':
        target_idx = attachment_idx - 1 if attachment_idx > 0 else 1
    elif retained_side == 'suffix':
        target_idx = attachment_idx + 1 if attachment_idx < n - 1 else n - 2
    else:
        raise ValueError(f"Unsupported retained_side: {retained_side}")

    target_idx = min(max(target_idx, 0), n - 1)
    if target_idx == attachment_idx:
        if attachment_idx > 0:
            target_idx = attachment_idx - 1
        elif attachment_idx < n - 1:
            target_idx = attachment_idx + 1
        else:
            return 0.0

    return _heading_to_north_deg(coords[attachment_idx], coords[target_idx])


def _attachment_heading_gate(
        base_stem: Stem,
        base_attachment_idx: int,
        base_retained_side: str,
        candidate_stem: Stem,
        candidate_attachment_idx: int,
        candidate_retained_side: str,
        tolerance_deg: float,
) -> bool:
    base_coords = list(base_stem.path.coords)
    candidate_coords = list(candidate_stem.path.coords)

    base_heading = _attachment_heading(
        base_coords,
        base_attachment_idx,
        base_retained_side,
    )
    candidate_heading = _attachment_heading(
        candidate_coords,
        candidate_attachment_idx,
        candidate_retained_side,
    )

    base_point = base_coords[base_attachment_idx]
    candidate_point = candidate_coords[candidate_attachment_idx]

    if base_point == candidate_point:
        return _is_antiparallel(base_heading, candidate_heading, tolerance_deg)

    bridge_heading_at_base = _heading_to_north_deg(base_point, candidate_point)
    bridge_heading_at_candidate = _heading_to_north_deg(
        candidate_point,
        base_point,
    )

    return (
        _is_antiparallel(base_heading, candidate_heading, tolerance_deg)
        and _is_antiparallel(base_heading, bridge_heading_at_base,
                             tolerance_deg)
        and _is_antiparallel(candidate_heading, bridge_heading_at_candidate,
                             tolerance_deg)
    )


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
        # Build all valid proposals in this cycle across all 4 endpoint cases.
        for base_idx in list(remaining):
            base_stem = cycle_stems[base_idx]
            start_buffer = base_stem.start.buffer(max_distance, resolution=32)
            end_buffer = base_stem.stop.buffer(max_distance, resolution=32)

            candidate_indices = set(
                _query_tree_indices(start_tree, start_buffer, start_points)
            )
            candidate_indices.update(
                _query_tree_indices(stop_tree, start_buffer, stop_points)
            )
            candidate_indices.update(
                _query_tree_indices(start_tree, end_buffer, start_points)
            )
            candidate_indices.update(
                _query_tree_indices(stop_tree, end_buffer, stop_points)
            )
            candidate_indices.discard(base_idx)

            for idx in candidate_indices:
                if idx == base_idx:
                    continue
                candidate = cycle_stems[idx]
                if _same_stem_identity(base_stem, candidate):
                    continue

                cases = []
                if _endpoint_in_buffer(candidate, 'start', end_buffer):
                    cases.append(('end', 'start'))
                if _endpoint_in_buffer(candidate, 'end', end_buffer):
                    cases.append(('end', 'end'))
                if _endpoint_in_buffer(candidate, 'start', start_buffer):
                    cases.append(('start', 'start'))
                if _endpoint_in_buffer(candidate, 'end', start_buffer):
                    cases.append(('start', 'end'))

                for base_ep, cand_ep in cases:
                    base_oriented = _orient_stem_for_base(base_stem, base_ep)
                    cand_oriented = _orient_stem_for_candidate(candidate, cand_ep)
                    if _same_stem_identity(base_oriented, cand_oriented):
                        continue
                    changed, vote, merged, _ = calc_connectivity_votes(
                        base_oriented,
                        None,
                        None,
                        None,
                        None,
                        max_distance,
                        max_tree_height,
                        tolerance_angle,
                        cand_oriented,
                    )
                    if changed:
                        proposals.append({
                            'base_idx': base_idx,
                            'cand_idx': idx,
                            'vote': vote,
                            'merged': merged,
                            'base_ep': base_ep,
                            'cand_ep': cand_ep,
                        })

        proposals.sort(key=lambda x: x['vote'])

        used_stems = set()
        used_endpoints = set()
        connected_stems = []

        for proposal in proposals:
            base_idx = proposal['base_idx']
            cand_idx = proposal['cand_idx']
            if base_idx in used_stems or cand_idx in used_stems:
                continue
            if (base_idx, proposal['base_ep']) in used_endpoints:
                continue
            if (cand_idx, proposal['cand_ep']) in used_endpoints:
                continue

            connected_stems.append(proposal['merged'])
            used_stems.add(base_idx)
            used_stems.add(cand_idx)
            used_endpoints.add((base_idx, proposal['base_ep']))
            used_endpoints.add((cand_idx, proposal['cand_ep']))
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
        max_tree_height,
        tolerance_angle,
    )
    if changed:
        return True, vote, merged, stem
    return False, math.inf, None, None


# calculate vote
def calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st, candidate, stem,
              stems0, tolerance_angle):
    return (
        ((1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / tolerance_angle) *
        candidate.start.distance(candidate.stop) ** 2
        + stems0.stop.distance(stem.start) ** 2 *
        (1 + ang_l_sp_el_st + ang_l_sp_mp + ang_mp_el_st) / tolerance_angle
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
    for i in range(len(segments)):
        if segments[i].start[1] >= segments[i].stop[1]:
            h = segments[i].start
            segments[i].start = segments[i].stop
            segments[i].stop = h
            segments[i].path.reverse()
     #   else:
     #       h = segments[i].start
     #       segments[i].start = segments[i].stop
     #       segments[i].stop = h
     #       segments[i].path.reverse()
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
