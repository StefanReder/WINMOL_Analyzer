#!/usr/bin/env python

################################################################################
"""Clean vector operations with aggressive endpoint prefilter.

This version implements:
- explicit endpoint-in-buffer candidate generation only
- both base endpoints are checked (start and end)
- both candidate endpoint types are checked (start and end)
- aggressive prefilter before the hard gate:
    * explicit endpoint must be inside the relevant buffer
    * candidate must not belong to the same stem
    * stem.length + gap + candidate.length < max_tree_height
- hard gate exactly on four local vectors:
    * v_stem            [base endpoint -> base interior]
    * v_stem_bridge     [base endpoint -> candidate endpoint]
    * v_candidate_bridge[candidate endpoint -> base endpoint]
    * v_candidate       [candidate endpoint -> candidate interior]
  with all three angles tested near 180 degrees using:
    tolerance_angle * dist_f
    dist_f = 1 - (1 / (3 + max_distance - gap)) ** 0.5
- unchanged vote function
- greedy best-candidate-per-base per cycle
- no heading gate
- no self-connections
"""

import math
import multiprocessing as mp
from typing import List, Sequence, Set, Tuple

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


def _endpoint_point(stem: Stem, endpoint_name: str) -> Point:
    return stem.start if endpoint_name in ('start',) else stem.stop


def _endpoint_in_buffer(stem: Stem, endpoint_name: str, buffer_geom) -> bool:
    point = _endpoint_point(stem, endpoint_name)
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


def _endpoint_vector_from_oriented(stem: Stem, endpoint_name: str) -> LineString:
    coords = list(stem.path.coords)
    if len(coords) < 3:
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
    """Orient base so the tested endpoint becomes stop."""
    coords = list(stem.path.coords)
    if endpoint_name == 'start':
        coords = list(reversed(coords))

    # Drop the tested endpoint (= stop) and keep only the remaining stem
    if len(coords) >= 3:
        coords_trimmed = coords[:-1]
    else:
        # fallback: keep at least 2 points
        coords_trimmed = coords[:]

    oriented = _clone_stem(stem)
    oriented.path = _safe_linestring(coords_trimmed)
    oriented.start = Point(coords_trimmed[0])
    oriented.stop = Point(coords_trimmed[-1])
    return oriented


def _orient_stem_for_candidate(stem: Stem, endpoint_name: str) -> Stem:
    """Orient candidate so the tested endpoint becomes start."""
    coords = list(stem.path.coords)
    if endpoint_name in ('end', 'stop'):
        coords = list(reversed(coords))

    # Drop the tested endpoint (= stop) and keep only the remaining stem
    if len(coords) >= 4:
        coords_trimmed = coords[1:]
    else:
        # fallback: keep at least 2 points
        coords_trimmed = coords[:]

    oriented = _clone_stem(stem)
    oriented.path = _safe_linestring(coords_trimmed)
    oriented.start = Point(coords_trimmed[0])
    oriented.stop = Point(coords_trimmed[-1])
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
    """Local antiparallel hard gate for one oriented endpoint pair.

    Assumes:
    - tested base endpoint == base_oriented.stop
    - tested candidate endpoint == candidate_oriented.start
    """
    #if _same_stem_identity(base_oriented, candidate_oriented):
    #    return False, math.inf, None, None

    gap = base_oriented.stop.distance(candidate_oriented.start)
    #if gap > max_distance:
    #    return False, math.inf, None, None

    dist_f = _distance_factor(gap, max_distance)
    tolerance_gate = tolerance_angle * dist_f

    v_stem = _endpoint_vector_from_oriented(base_oriented, 'end')
    v_candidate = _endpoint_vector_from_oriented(candidate_oriented, 'start')

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


def _candidate_prefilter_ok(base_stem: Stem, candidate_stem: Stem, base_endpoint_name: str,
                            cand_endpoint_name: str, max_distance: float,
                            max_tree_height: float, buffer_geom) -> Tuple[bool, float]:
    if _same_stem_identity(base_stem, candidate_stem):
        return False, math.inf
    if not _endpoint_in_buffer(candidate_stem, cand_endpoint_name, buffer_geom):
        return False, math.inf
    base_point = _endpoint_point(base_stem, base_endpoint_name)
    cand_point = _endpoint_point(candidate_stem, cand_endpoint_name)
    gap = base_point.distance(cand_point)
    if gap > max_distance:
        return False, math.inf
    if (getattr(base_stem, 'length', base_stem.path.length) + gap +
            getattr(candidate_stem, 'length', candidate_stem.path.length)) >= max_tree_height:
        return False, math.inf
    return True, gap


################################################################################
"""Vector operations"""


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
        used_endpoints = set()  # (stem_idx, 'start'|'end')
        connected_stems = []

        while remaining:
            base_idx = next(iter(remaining))
            base_stem = cycle_stems[base_idx]

            best_vote = math.inf
            best_candidate = None
            best_slave_idx = None
            best_base_ep = None
            best_cand_ep = None

            # Check both base endpoints independently.
            for base_ep, buffer_geom in (
                ('start', base_stem.start.buffer(max_distance, resolution=32)),
                ('end', base_stem.stop.buffer(max_distance, resolution=32)),
            ):
                if (base_idx, base_ep) in used_endpoints:
                    continue

                # In each buffer, test both candidate endpoint types explicitly.
                endpoint_queries = [
                    ('start', _query_tree_indices(start_tree, buffer_geom, start_points)),
                    ('end', _query_tree_indices(stop_tree, buffer_geom, stop_points)),
                ]

                seen_pairs = set()
                for cand_ep, idxs in endpoint_queries:
                    for idx in idxs:
                        pair_key = (idx, cand_ep)
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        if idx == base_idx or idx not in remaining:
                            continue
                        if (idx, cand_ep) in used_endpoints:
                            continue

                        candidate = cycle_stems[idx]
                        ok_prefilter, _ = _candidate_prefilter_ok(
                            base_stem, candidate, base_ep, cand_ep,
                            max_distance, max_tree_height, buffer_geom,
                        )
                        if not ok_prefilter:
                            continue

                        base_oriented = _orient_stem_for_base(base_stem, base_ep)
                        candidate_oriented = _orient_stem_for_candidate(candidate, cand_ep)
                        #if _same_stem_identity(base_oriented, candidate_oriented):
                        #    continue

                        changed, vote, merged, _ = calc_connectivity_votes(
                            base_oriented,
                            None,
                            None,
                            None,
                            None,
                            max_distance,
                            max_tree_height,
                            tolerance_angle,
                            candidate_oriented,
                        )
                        if changed and vote < best_vote:
                            best_vote = vote
                            best_candidate = merged
                            best_slave_idx = idx
                            best_base_ep = base_ep
                            best_cand_ep = cand_ep

            if best_candidate is not None and best_slave_idx is not None:
                connected_stems.append(best_candidate)
                remaining.discard(base_idx)
                remaining.discard(best_slave_idx)
                used_endpoints.add((base_idx, best_base_ep))
                used_endpoints.add((best_slave_idx, best_cand_ep))
                global_change = True
                c_count += 1
            else:
                connected_stems.append(base_stem)
                remaining.discard(base_idx)

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
