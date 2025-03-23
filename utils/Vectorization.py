#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import List

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

from classes.Part import Part
from classes.Stem import Stem
from classes.Timer import Timer
from utils.Geometry import ang
from utils.IO import get_bounds_from_profile

# System epsilon
epsilon = np.finfo(float).eps

################################################################################
"""Vector operations"""


# Parallel version of connect_stems
def connect_stems(stems: List[Stem], config) -> List[Stem]:
    max_distance = config.max_distance
    tolerance_angle = config.tolerance_angle

    # Aggregate aligned stem segments to stems. Reconstructs occluded stems
    # parts up die max_distance

    def return_callback(result):
        if result is not None:
            results.append(result)

    def error_callback(error):
        print(error, flush=True)

    t = Timer()
    t.start()
    pool = mp.Pool(mp.cpu_count() - 1)
    print("#######################################################")
    print("Gathering stem segments ")
    cycle_nbr = 1
    c_count = 0
    out_count = 0
    duplicates_count = 0
    count_stem_parts = len(stems)
    global_change = True

    while global_change:
        # loop as long as stem parts can be attached to each other
        global_change = False
        # sort by x coordinate
        #   stems.sort(key=lambda x: x.start.x)#, reverse=True)
        print("Cycle ", cycle_nbr)
        connected_stems = []

        stem_count = 0
        can_count = 0
        merged_count = 0
        vote_count = 0
        dub_count = 0

        while stems:
            # loop while there are stems to extend

            # create a line of max 3 segments at both ends of the stem which
            # represents the direction of the stem end
            if len(stems[0].path.coords) < 4:
                line_start = LineString(
                    [stems[0].path.coords[0], stems[0].path.coords[-1]]
                )
                line_stop = LineString(
                    [stems[0].path.coords[0], stems[0].path.coords[-1]]
                )
            else:
                if len(stems[0].path.coords) < 8:
                    i = len(stems[0].path.coords) - 2
                else:
                    i = 6
                line_start = LineString(
                    [stems[0].path.coords[1], stems[0].path.coords[i]]
                )
                line_stop = LineString(
                    [stems[0].path.coords[-(i + 1)], stems[0].path.coords[-2]]
                )
            start_buffer = stems[0].start.buffer(max_distance, resolution=32)
            end_buffer = stems[0].stop.buffer(max_distance, resolution=32)

            # look at both ends for stems and search for potentially candidates
            # to append on stem[0]
            stems_ = [s for s in stems[1:] if
                      start_buffer.contains(s.stop) or end_buffer.contains(
                          s.start)]
            can_count = can_count + (len(stems_))

            candidates = []
            votes = []
            slaves = []
            results = []

            r = []
            # Parallel computation of connectivity votes for stem parts
            # potentially appended to stems[0]
            for stem in stems_:
                r.append(
                    pool.apply_async(
                        calc_connectivity_votes, args=(
                            stems[0],
                            line_start,
                            line_stop,
                            start_buffer,
                            end_buffer,
                            max_distance,
                            tolerance_angle,
                            stem
                        ), callback=return_callback,
                        error_callback=error_callback))

            for r_ in r:
                r_.wait()

            # prepare for evaluation
            change = [result[0] for result in results]
            votes = [result[1] for result in results]
            candidates = [result[2] for result in results]
            slaves = [result[3] for result in results]
            vote_count = vote_count + len(votes)

            if any(change):
                index_min = min(range(len(votes)), key=votes.__getitem__)
                # stem is updated and the merged part is removed
                stems[0] = candidates[index_min]
                stems.remove(slaves[index_min])
                global_change = True
                c_count = c_count + 1
                stems, dub_count_ = remove_duplicates(stems, stems[0])
                dub_count = dub_count + dub_count_
                merged_count = merged_count + 1

            else:
                # if no other stem part could be attached to stems[0], it is
                # added to the export container and removed from the stem list
                connected_stems.append(stems[0])
                stems.remove(stems[0])
                stem_count = stem_count + 1
                duplicates_count = duplicates_count + dub_count

                can_count = 0
                merged_count = 0
                dub_count = 0

        # all stems have veen observed and passed to connected stems. s
        stems = connected_stems
        cycle_nbr = cycle_nbr + 1

    pool.close()

    connected_stems = []
    for stem in stems:
        if stem.length > config.min_length:
            connected_stems.append(stem)
        else:
            out_count = out_count + 1
    connected_stems, dup_count_2 = remove_duplicates(connected_stems)
    dub_count_ = duplicates_count + dup_count_2
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
        tolerance_angle,
        stem: Stem
) -> (bool, List[float], List[Stem], List[Stem]):
    # Calculate votes for the aggregation of stem parts to stems
    if stem == stems0:
        if stem.start == stems0.start:
            if stem.stop == stems0.stop:
                print("Alerta!!!!", flush=True)
                print("stem0: ", list(stems0.path.coords), flush=True)
                print("stem: ", list(stem.path.coords), flush=True)
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
    if end_buffer.contains(stem.start) and ang_l_sp_el_st < tolerance_angle:
        # missing_part = LineString([stems0.stop, stem.start])
        missing_part_ = LineString(
            [stems0.path.coords[-2],
             stem.path.coords[1]]
        )
        dist_f = 1 - (
            1 / (3 + max_distance - stems0.stop.distance(stem.start))
            ** 0.5
        )
        ang_l_sp_mp = abs(ang(line_stop.coords, missing_part_.coords))
        ang_mp_el_st = abs(ang(missing_part_.coords, e_line_start.coords))

        if (ang_l_sp_el_st < (tolerance_angle * dist_f) and ang_l_sp_mp < (
                tolerance_angle * dist_f) and ang_mp_el_st < (
                tolerance_angle * dist_f) and stems0.start.distance(
                stem.stop) < 35):

            if len(stems0.path.coords) > 2 and len(stem.path.coords) > 2:
                start = LineString(stems0.path.coords[:-1])
                end = LineString(stem.path.coords[1:])
                new_path = linemerge([start, missing_part_, end])
            else:
                if len(stems0.path.coords) > 2 and has_length_2:
                    start = LineString(stems0.path.coords[:-1])
                    new_path = linemerge([start, missing_part_])
                else:
                    if (len(stems0.path.coords) == 2 and len(
                            stem.path.coords) > 2):
                        end = LineString(stem.path.coords[1:])
                        new_path = linemerge([missing_part_, end])
                    else:
                        if (len(stems0.path.coords) == 2 and has_length_2):
                            new_path = missing_part_

            change = True
            candidate = stems0
            candidate.path = new_path
            candidate.stop = stem.stop
            slave = stem
            vote = calc_vote(ang_l_sp_el_st, ang_l_sp_mp, ang_mp_el_st,
                             candidate, stem, stems0, tolerance_angle)
            candidates.append(candidate)
            votes.append(vote)
            slaves.append(slave)

    if start_buffer.contains(stem.stop) and ang_el_sp_l_st < tolerance_angle:
        # missing_part = LineString([stem.stop, stems0.start])
        missing_part_ = LineString(
            [stem.path.coords[-2], stems0.path.coords[1]])
        dist_f = 1 - (
            1 / (3 + max_distance - stem.stop.distance(stems0.start))
            ** 0.5
        )
        ang_el_sp_mp = abs(ang(e_line_stop.coords, missing_part_.coords))
        ang_mp_l_st = abs(ang(missing_part_.coords, line_start.coords))

        if (ang_el_sp_l_st < (tolerance_angle * dist_f) and ang_el_sp_mp < (
                tolerance_angle * dist_f) and abs(
                ang(missing_part_.coords, line_start.coords)) < (
                tolerance_angle * dist_f) and stem.start.distance(
                stems0.stop) < 35):
            if len(stem.path.coords) > 2 and len(stems0.path.coords) > 2:
                start = LineString(stem.path.coords[:-1])
                end = LineString(stems0.path.coords[1:])
                new_path = linemerge([start, missing_part_, end])
            else:
                if len(stem.path.coords) > 2 and len(stems0.path.coords) == 2:
                    start = LineString(stem.path.coords[:-1])
                    new_path = linemerge([start, missing_part_])
                else:
                    if has_length_2 and len(stems0.path.coords) > 2:
                        end = LineString(stems0.path.coords[1:])
                        new_path = linemerge([missing_part_, end])
                    else:
                        if (has_length_2 and len(
                                stems0.path.coords) == 2):
                            new_path = missing_part_

            change = True
            candidate = stems0
            candidate.path = new_path
            candidate.start = stem.start
            slave = stem
            vote = calc_vote(ang_el_sp_l_st, ang_el_sp_mp, ang_mp_l_st,
                             candidate, stems0, stem, tolerance_angle)
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
        else:
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


# Removes duplicates from stem list
def remove_duplicates(stems: List[Stem], stems0=None) -> List[Stem]:
    stems.sort(key=lambda x: x.length, reverse=True)
    stems_ = []
    count = 0
    if type(stems0) is Stem:
        for s in stems:
            if stems0.path.buffer(0.3).contains(s.path):
                stems.remove(s)
                count = count + 1
        stems_ = stems
        stems_.append(stems0)
    else:
        while stems:
            for s in stems[1:]:
                if stems[0].path.buffer(0.3).contains(s.path):
                    stems.remove(s)
                    count = count + 1
            stems_.append(stems[0])
            stems.remove(stems[0])
    return stems_, count


# remove padding and restore geoinformation of the stems
def restore_geoinformation(stems: List[Stem], config, profile):
    t = Timer()
    t.start()

    print("#######################################################")
    print("Restoring geoinformation")

    px_size = profile['transform'][0]
    bounds = get_bounds_from_profile(profile)
    padding = int(config.max_tree_height / px_size) + 1

    for j in range(len(stems)):
        stems[j].start = (bounds.left + (stems[j].start[1] - padding) * px_size,
                          bounds.top - (stems[j].start[0] - padding) * px_size)
        stems[j].stop = (bounds.left + (stems[j].stop[1] - padding) * px_size,
                         bounds.top - (stems[j].stop[0] - padding) * px_size)

        for k in range(len(stems[j].path)):
            stems[j].path[k] = (
                bounds.left + (stems[j].path[k][1] - padding) * px_size,
                bounds.top - (stems[j].path[k][0] - padding) * px_size
            )
        
    t.stop()
    print("#######################################################")
    print("")
    return stems
