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
    # start  -> interior, stop -> interior
    line_start = LineString([coords[0], coords[start_idx]])
    line_stop = LineString([coords[-1], coords[stop_idx]])
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
        connected_stems = []

        while remaining:
            base_idx = next(iter(remaining))
            base_stem = cycle_stems[base_idx]

            while True:
                line_start, line_stop = _stem_end_lines(base_stem)
                start_buffer = base_stem.start.buffer(
                    max_distance, resolution=32)
                end_buffer = base_stem.stop.buffer(
                    max_distance, resolution=32)

                candidate_indices = set(
                    _query_tree_indices(stop_tree, start_buffer, stop_points))
                candidate_indices.update(
                    _query_tree_indices(start_tree, end_buffer, start_points))
                candidate_indices.intersection_update(remaining)
                candidate_indices.discard(base_idx)

                # exact endpoint filter
                filtered_indices = []
                for idx in candidate_indices:
                    candidate = cycle_stems[idx]
                    if (
                        start_buffer.contains(candidate.stop)
                        or end_buffer.contains(candidate.start)
                    ):
                        filtered_indices.append(idx)

                best_vote = math.inf
                best_candidate = None
                best_slave_idx = None

                for idx in filtered_indices:
                    changed, vote, candidate_stem, _ = calc_connectivity_votes(
                        base_stem,
                        line_start,
                        line_stop,
                        start_buffer,
                        end_buffer,
                        max_distance,
                        max_tree_height,
                        tolerance_angle,
                        cycle_stems[idx],
                    )
                    if changed and vote < best_vote:
                        best_vote = vote
                        best_candidate = candidate_stem
                        best_slave_idx = idx

                if best_candidate is not None and best_slave_idx is not None:
                    base_stem = best_candidate
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
    # Calculate votes for the aggregation of stem parts to stems
    if stem == stems0:
        # if the stems are identical return no change and infinite vote
        return False, math.inf, None, None
    change = False
    votes = []
    candidates = []
    slaves = []

    if len(stem.path.coords) < 4:
        e_line_start = LineString([stem.path.coords[0], stem.path.coords[-1]])
        e_line_stop = LineString([stem.path.coords[0], stem.path.coords[-1]])
    else:
        if len(stem.path.coords) < 8:
            k = len(stem.path.coords) - 2
        else:
            k = 6
        e_line_start = LineString([stem.path.coords[1], stem.path.coords[k]])
        e_line_stop = LineString(
            [stem.path.coords[-(k + 1)], stem.path.coords[-2]])

    ang_l_sp_el_st = abs(ang(line_stop.coords, e_line_start.coords))
    ang_el_sp_l_st = abs(ang(e_line_stop.coords, line_start.coords))

    has_length_2 = len(stem.path.coords) == 2

    if end_buffer.contains(stem.start):
        # True endpoint connection: stems0.stop -> stem.start
        base_attach_idx = max(len(stems0.path.coords) - 1, 0)
        candidate_attach_idx = 0
        missing_part_ = LineString(
            [stems0.path.coords[base_attach_idx],
             stem.path.coords[candidate_attach_idx]]
        )
        gap = stems0.stop.distance(stem.start)
        dist_f = 1 - (
            1 / (3 + max_distance - gap)
            ** 0.5
        )
        tolerance_gate = tolerance_angle * dist_f
        forward_heading_ok = _attachment_heading_gate(
            stems0,
            base_attach_idx,
            'prefix',
            stem,
            candidate_attach_idx,
            'suffix',
            tolerance_gate,
        )

        # Endpoint-local bridge orientations
        bridge_from_base = missing_part_
        bridge_from_candidate = LineString([
            stem.path.coords[candidate_attach_idx],
            stems0.path.coords[base_attach_idx],
        ])

        # Both stem endpoint vectors point away from the junction.
        # Therefore, each must be almost antiparallel to its local bridge
        # direction, while the two stem endpoint vectors must be nearly parallel.
        ang_stem_stem = abs(ang(line_stop.coords, e_line_start.coords))
        ang_base_bridge = abs(180.0 - ang(line_stop.coords, bridge_from_base.coords))
        ang_candidate_bridge = abs(
            180.0 - ang(e_line_start.coords, bridge_from_candidate.coords)
        )

        if (
                ang_stem_stem < tolerance_gate
                and ang_base_bridge < tolerance_gate
                and ang_candidate_bridge < tolerance_gate
                and stems0.start.distance(stem.stop) < max_tree_height
                and forward_heading_ok):

            new_coords = list(stems0.path.coords) + list(stem.path.coords)
            try:
                new_path = LineString(new_coords)
            except Exception:
                new_path = linemerge([
                    LineString(stems0.path.coords),
                    missing_part_,
                    LineString(stem.path.coords),
                ])

            merged_length = new_path.length
            merged_span = stems0.start.distance(stem.stop)

            if (
                    gap <= max_distance
                    and merged_length <= max_tree_height
                    and merged_span <= max_tree_height):
                change = True
                candidate = _clone_stem(stems0)
                candidate.path = new_path
                candidate.stop = stem.stop
                slave = stem
                vote = calc_vote(
                    ang_stem_stem,
                    ang_base_bridge,
                    ang_candidate_bridge,
                    candidate,
                    stem,
                    stems0,
                    tolerance_angle,
                )
                candidates.append(candidate)
                votes.append(vote)
                slaves.append(slave)

    if start_buffer.contains(stem.stop):
        # True endpoint connection: stem.stop -> stems0.start
        candidate_attach_idx = max(len(stem.path.coords) - 1, 0)
        base_attach_idx = 0
        missing_part_ = LineString(
            [stem.path.coords[candidate_attach_idx],
             stems0.path.coords[base_attach_idx]]
        )
        gap = stem.stop.distance(stems0.start)
        dist_f = 1 - (
            1 / (3 + max_distance - gap)
            ** 0.5
        )
        tolerance_gate = tolerance_angle * dist_f
        reverse_heading_ok = _attachment_heading_gate(
            stems0,
            base_attach_idx,
            'suffix',
            stem,
            candidate_attach_idx,
            'prefix',
            tolerance_gate,
        )

        bridge_from_candidate = missing_part_
        bridge_from_base = LineString([
            stems0.path.coords[base_attach_idx],
            stem.path.coords[candidate_attach_idx],
        ])

        ang_stem_stem = abs(ang(e_line_stop.coords, line_start.coords))
        ang_candidate_bridge = abs(
            180.0 - ang(e_line_stop.coords, bridge_from_candidate.coords)
        )
        ang_base_bridge = abs(180.0 - ang(line_start.coords, bridge_from_base.coords))

        if (
                ang_stem_stem < tolerance_gate
                and ang_candidate_bridge < tolerance_gate
                and ang_base_bridge < tolerance_gate
                and stem.start.distance(stems0.stop) < max_tree_height
                and reverse_heading_ok):
            new_coords = list(stem.path.coords) + list(stems0.path.coords)
            try:
                new_path = LineString(new_coords)
            except Exception:
                new_path = linemerge([
                    LineString(stem.path.coords),
                    missing_part_,
                    LineString(stems0.path.coords),
                ])

            merged_length = new_path.length
            merged_span = stem.start.distance(stems0.stop)

            if (
                    gap <= max_distance
                    and merged_length <= max_tree_height
                    and merged_span <= max_tree_height):
                change = True
                candidate = _clone_stem(stems0)
                candidate.path = new_path
                candidate.start = stem.start
                slave = stem
                vote = calc_vote(
                    ang_stem_stem,
                    ang_candidate_bridge,
                    ang_base_bridge,
                    candidate,
                    stems0,
                    stem,
                    tolerance_angle,
                )
                candidates.append(candidate)
                votes.append(vote)
                slaves.append(slave)

    if change:
        index_min = min(range(len(votes)), key=votes.__getitem__)
        return True, votes[index_min], candidates[index_min], slaves[index_min]
    else:
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
