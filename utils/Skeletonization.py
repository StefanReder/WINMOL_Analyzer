#!/usr/bin/env python

################################################################################
"""Imports"""

import math
import multiprocessing as mp
from typing import Any, List, Tuple

import numpy as np
import scipy.ndimage.measurements
from numpy import ndarray
from skimage import morphology

from classes.Part import Part
from classes.Timer import Timer
from utils.Geometry import ang


# System epsilon
epsilon = np.finfo(float).eps

################################################################################
"""Skeleton operations"""


# Find Nodes and connected segments in the Skeleton
def find_segments(pred, config, profile) -> (List[Part], List[Tuple[int]]):
    t = Timer()
    t.start()

    print("#######################################################")
    print("Skeletonize Image")

    px_size = profile['transform'][0]
    min_length = config.min_length / 4
    padding = int(config.max_tree_height / px_size) + 1
    pred = np.pad(
        pred,
        ((padding, padding), (padding, padding)),
        'constant',
        constant_values=False
    )

    # binarize image
    pred[np.where(pred < 0.5)] = 0
    pred[np.where(pred >= 0.5)] = 1

    skel = morphology.skeletonize(pred)

    t.stop()
    print("#######################################################")
    print("")

    end_nodes, skel = get_nodes(skel)
    segments, skel = find_skeleton_segments(skel, end_nodes,
                                            math.floor(min_length / px_size),
                                            padding)
    segments = refine_skeleton_segments(segments, skel,
                                        math.floor(min_length / px_size))

    return segments


# get nodes
def get_nodes(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], Any]:
    t = Timer()
    t.start()
    print("#######################################################")
    print("Splitting the skeleton into segments and detecting endnodes")

    skel, dn_count = remove_dense_skeleton_nodes(skel)

    print("Dense nodes removed: ", dn_count)
    t.stop()
    t.start()
    end_nodes, branch_points = find_skeleton_nodes(skel)
    bp_count = len(branch_points)
    while len(branch_points) > 0:
        skel = remove_branchpoints_from_skel(skel, branch_points)
        end_nodes, branch_points = find_skeleton_nodes(skel)
        bp_count = bp_count + len(branch_points)
    skel = morphology.skeletonize(skel)
    print("Branch points removed: ", bp_count)
    print("Detected end nodes: ", len(end_nodes))
    t.stop()
    print("#######################################################")
    print("")
    return end_nodes, skel


# Remove "dense" (2x2 or larger) regions in the skeleton.
def remove_dense_skeleton_nodes(skel: np.ndarray) -> Tuple[ndarray, int]:
    dense_nodes = morphology.binary_erosion(
        np.pad(skel, 1),
        np.ones((2, 2))
    )[1:-1, 1:-1]
    labeled_array, num_features = scipy.ndimage.measurements.label(dense_nodes)
    centers = scipy.ndimage.measurements.center_of_mass(
        dense_nodes,
        labeled_array, [*range(1, num_features + 1)]
    )
    count = len(centers)

    skel[np.where(dense_nodes.__eq__(True))] = False
    return skel, count


def find_skeleton_nodes(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    print("Find skeletion nodes")

    print("# Pad the skeleton array (same as in the numpy version)")
    skel = np.pad(skel, 1, mode='constant', constant_values=0)

    print("# Extract 8-neighbors using slicing")
    p2 = skel[:-2, 1:-1]
    p3 = skel[:-2, 2:]
    p4 = skel[1:-1, 2:]
    p5 = skel[2:, 2:]
    p6 = skel[2:, 1:-1]
    p7 = skel[2:, :-2]
    p8 = skel[1:-1, :-2]
    p9 = skel[:-2, :-2] 
    p1 = skel[1:-1, 1:-1]  # This is the center pixel (without padding)

    print("# Binary skeleton mask")
    mask = p1 == 1

    print("# A(p1) calculation (transition count)")
    transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8) + \
                  ((p3 == 0) & (p4 == 1)) + \
                  ((p4 == 0) & (p5 == 1)) + \
                  ((p5 == 0) & (p6 == 1)) + \
                  ((p6 == 0) & (p7 == 1)) + \
                  ((p7 == 0) & (p8 == 1)) + \
                  ((p8 == 0) & (p9 == 1)) + \
                  ((p9 == 0) & (p2 == 1))

    print("# Endpoint: A(p1) == 1, Branchpoint: A(p1) >= 3")
    endpoint_mask = (transitions == 1) & mask
    branchpoint_mask = (transitions >= 3) & mask

    print("# Get coordinates (remove padding offset)")
    endpoints = np.argwhere(endpoint_mask)  # shape (N, 2)
    branchpoints = np.argwhere(branchpoint_mask)

    print("# Convert to CPU tuples")
    endpoints = [tuple(map(int, p)) for p in endpoints]
    branchpoints = [tuple(map(int, p)) for p in branchpoints]

    return endpoints, branchpoints


def remove_branchpoints_from_skel(skel, branchpoints):
    print("Remove branch points")
    skel_arr = np.asarray(skel, dtype=bool)
    branchpoints_arr = np.asarray(branchpoints)

    mask = np.zeros_like(skel_arr, dtype=bool)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            xs = branchpoints_arr[:, 0] + dx
            ys = branchpoints_arr[:, 1] + dy

            # Clip to valid range to avoid index errors
            xs = np.clip(xs, 0, skel_arr.shape[0] - 1)
            ys = np.clip(ys, 0, skel_arr.shape[1] - 1)

            mask[xs, ys] = True

    skel_arr[mask] = False
    return skel_arr


def find_skeleton_segments(
        skel: np.ndarray,
        end_nodes: List[Tuple[int]],
        min_length: int,
        padding: int
) -> (List[Part], np.ndarray):
    t = Timer()
    t.start()
    out_skel = np.full(skel.shape, False)
    skeleton_parts = []

    def return_callback(result):
        nonlocal skeleton_parts  # noqa: F824
        nonlocal out_skel  # noqa: F824
        if result is not None:
            (
                skel_part_inner, sub_skel_inner,
                low_bounds_inner, up_bounds_inner
            ) = result
            skeleton_parts.append(skel_part_inner)
            out_skel[
                low_bounds_inner[0]:up_bounds_inner[0] + 1,
                low_bounds_inner[1]:up_bounds_inner[1] + 1
            ] = (
                sub_skel_inner | out_skel[
                    low_bounds_inner[0]: up_bounds_inner[0] + 1,
                    low_bounds_inner[1]:
                    up_bounds_inner[1] + 1
                ]
            )

    def error_callback(error):
        print(error, flush=True)

    print("#######################################################")
    print("Find connected segments in the skeleton")
    print("Initial length of skeleton: ", np.count_nonzero(skel))
    print("Number of end nodes", len(end_nodes))
    print("Minimum length in pixel: ", min_length)

    pool = mp.Pool(mp.cpu_count() - 1)
    r = []
    for end_node in end_nodes:
        low_bounds = (end_node[0] - padding, end_node[1] - padding)
        up_bounds = (end_node[0] + padding, end_node[1] + padding)
        sub_skel = skel[
            low_bounds[0]:up_bounds[0] + 1,
            low_bounds[1]:up_bounds[1] + 1
        ]
        r.append(pool.apply_async(get_segment, args=(
            end_node,
            end_nodes,
            sub_skel,
            low_bounds,
            up_bounds,
            min_length
        ), callback=return_callback, error_callback=error_callback))
    for r_ in r:
        r_.wait()
    pool.close()
    skeleton_parts = set(skeleton_parts)
    print("Detected skeleton segments: ", len(skeleton_parts))
    t.stop()
    print("#######################################################")
    print("")

    return skeleton_parts, out_skel


def get_segment(end_node, end_nodes, skel, low_bounds, up_bounds, min_length):
    end_node = (end_node[0] - low_bounds[0], end_node[1] - low_bounds[1])
    for i in range(len(end_nodes)):
        end_nodes[i] = (
            end_nodes[i][0] - low_bounds[0], end_nodes[i][1] - low_bounds[1])

    temp_skel = np.full(skel.shape, False)
    x, y = end_node
    end_nodes.remove(end_node)
    skel[(x, y)] = False
    temp_skel[(x, y)] = True
    node = False
    l_bound_x = end_node[0]
    l_bound_y = end_node[1]
    u_bound_x = end_node[0]
    u_bound_y = end_node[1]
    length = 0

    if len(get_neighbors(x, y, skel)) == 0:
        return None
    while not node:
        frontier = get_neighbors(x, y, skel)
        if frontier:
            length = length + 1
            x, y = frontier[0]
            if x < l_bound_x:
                l_bound_x = x
            if y < l_bound_y:
                l_bound_y = y
            if x > u_bound_x:
                u_bound_x = x
            if y > u_bound_y:
                u_bound_y = y
            skel[(x, y)] = False
            temp_skel[(x, y)] = True
            if frontier[0] in end_nodes:
                node = True
                l_bound = (l_bound_x, l_bound_y)
                u_bound = (u_bound_x, u_bound_y)
                new_part = Part(end_node, frontier[0], [end_node, frontier[0]],
                                l_bound, u_bound)
        else:
            l_bound = (l_bound_x, l_bound_y)
            u_bound = (u_bound_x, u_bound_y)
            new_part = Part(end_node, (x, y), [end_node, (x, y)], l_bound,
                            u_bound)
            node = True
    if length < min_length:
        return None

    if new_part.start[0] > new_part.stop[0]:
        new_part = Part(new_part.stop, new_part.start,
                        [new_part.stop, new_part.start], l_bound, u_bound)

    new_part.start = (
        new_part.start[0] + low_bounds[0], new_part.start[1] + low_bounds[1])
    new_part.stop = (
        new_part.stop[0] + low_bounds[0], new_part.stop[1] + low_bounds[1])
    for i in range(len(new_part.path)):
        new_part.path[i] = (new_part.path[i][0] + low_bounds[0],
                            new_part.path[i][1] + low_bounds[1])
    new_part.l_bound = (
        new_part.l_bound[0] + low_bounds[0],
        new_part.l_bound[1] + low_bounds[1])
    new_part.u_bound = (
        new_part.u_bound[0] + low_bounds[0],
        new_part.u_bound[1] + low_bounds[1])

    return new_part, temp_skel, low_bounds, up_bounds


# Parallel version of refine_skeleton_segments
# Find stem parts between nodes using the connectivity in the skeleton.
# Returns a list of parts (pairs of nodes) with a minimum distance and a
# cleaned skeleton.

def refine_skeleton_segments(parts: List[Part], skel: np.ndarray,
                             distance: int) -> (List[Part], np.ndarray):
    split = 0
    out = 0
    refined_parts = []

    def return_callback(result):
        refined_part, s, o = result
        nonlocal split
        nonlocal out
        nonlocal refined_parts  # noqa: F824
        split = split + s
        out = out + o
        if refined_part is not None:
            for refined in refined_part:
                refined_parts.append(refined)

    def error_callback(error):
        print(error, flush=True)

    t = Timer()
    t.start()
    refined_parts = []

    print("#######################################################")
    print("#Refining and sorting out skeleton segments")
    print("Initial length of skeleton: ", np.count_nonzero(skel))
    print("Number of initial skeleton segments", len(parts))

    pool = mp.Pool(mp.cpu_count() - 1)
    r = []
    for part in parts:
        low_bounds = (part.l_bound[0] - 5, part.l_bound[1] - 5)
        up_bounds = (part.u_bound[0] + 5, part.u_bound[1] + 5)
        sub_skel = skel[
            low_bounds[0]:up_bounds[0] + 1,
            low_bounds[1]:up_bounds[1] + 1
        ]

        r.append(pool.apply_async(refine_skeleton_segment, args=(
            part,
            low_bounds,
            up_bounds,
            sub_skel,
            distance
        ), callback=return_callback, error_callback=error_callback))

    for r_ in r:
        r_.wait()
    pool.close()

    print("Number of split segments:", split)
    print("Number of removed segments:", out)
    print("Number of refined segments:", len(refined_parts))

    t.stop()
    print("#######################################################")
    print("")
    return refined_parts


def refine_skeleton_segment(part: Part, low_bounds: Tuple[int, int],
                            up_bounds: Tuple[int, int],
                            skel: np.ndarray, distance: int) -> List[Part]:
    part.start = (part.start[0] - low_bounds[0], part.start[1] - low_bounds[1])
    part.stop = (part.stop[0] - low_bounds[0], part.stop[1] - low_bounds[1])
    part.path = [part.start, part.stop]
    refined_parts_ = []
    parts = []
    parts.append(part)
    out_ = 0
    split_ = 0
    while len(parts) > 0:

        w = parts[0].start
        n = parts[0].start
        z = parts[0].stop

        # p_last=part.path
        p_last = [parts[0].start, parts[0].stop]
        parts[0].path = []
        parts[0].path.extend([w])
        temp = np.full(skel.shape, False)
        x_last, x_last = w
        while w != z:
            x, y = w
            skel[(x, y)] = False
            temp[(x, y)] = True
            ww = get_neighbors(x, y, skel)
            if ww:
                # Step forward
                w = ww[0]
                p_recent = [n, w]
                angle = ang(p_recent, p_last)

                if w == z:
                    if angle > 10:
                        new_part = Part(n, parts[0].stop, [n, parts[0].stop],
                                        low_bounds, up_bounds)
                        parts.append(new_part)
                        parts[0].stop = n
                        # restore points since last node
                        skel[np.where(temp.__eq__(True))] = True
                        temp = np.full(skel.shape, False)
                        split_ = split_ + 1
                    else:
                        parts[0].path.extend([w])
                        temp = np.full(skel.shape, False)
                else:
                    if math.dist(n, w) > distance:
                        if n == parts[0].start:
                            if angle > 10:
                                new_part = Part(w, parts[0].stop,
                                                [w, parts[0].stop], low_bounds,
                                                up_bounds)
                                parts.append(new_part)
                                parts[0].stop = w
                                parts[0].path.extend([w])
                                split_ = split_ + 1
                                z = w
                            else:  # Add node for diameter measurement
                                parts[0].path.extend([w])
                                p_last = p_recent
                                n = w
                                temp = np.full(skel.shape, False)

                        else:
                            if angle > 30:
                                new_part = Part(n, parts[0].stop,
                                                [n, parts[0].stop], low_bounds,
                                                up_bounds)
                                parts.append(new_part)
                                parts[0].stop = n
                                # restore points since last node
                                skel[np.where(temp.__eq__(True))] = True
                                z = w
                                split_ = split_ + 1

                            else:
                                # Add node for diameter measurement
                                parts[0].path.extend([w])
                                p_last = p_recent
                                n = w
                                temp = np.full(skel.shape, False)

            else:
                parts[0].path.extend([(x, y)])
                parts[0].stop = (x, y)
                z = (x, y)
                w = z

        refined_part_ = Part(parts[0].start, parts[0].stop, parts[0].path,
                             low_bounds, up_bounds)
        parts.pop(0)

        if math.dist(refined_part_.start, refined_part_.stop) >= distance:
            refined_part_.start = (refined_part_.start[0] + low_bounds[0],
                                   refined_part_.start[1] + low_bounds[1])
            refined_part_.stop = (refined_part_.stop[0] + low_bounds[0],
                                  refined_part_.stop[1] + low_bounds[1])
            for i in range(len(refined_part_.path)):
                refined_part_.path[i] = (
                    refined_part_.path[i][0] + low_bounds[0],
                    refined_part_.path[i][1] + low_bounds[1])

            if refined_part_.start[0] > refined_part_.stop[0]:
                refined_part_ = Part(refined_part_.stop, refined_part_.start,
                                     refined_part_.path, low_bounds, up_bounds)
                refined_part_.path.reverse()

            refined_parts_.append(refined_part_)
        else:
            out_ = out_ + 1

    if len(refined_parts_) == 0:
        return None, split_, out_

    return refined_parts_, split_, out_


def get_neighbors(x: int, y: int, skel: np.ndarray) -> List[Tuple[int, int]]:
    # Define the 8 neighbor coordinate offsets (excluding the center (0, 0))
    offsets = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1],
    ])

    # Calculate absolute coordinates of neighbors
    coords = offsets + [x, y]

    # Filter out-of-bounds coordinates
    h, w = skel.shape
    mask = (
        (coords[:, 0] >= 0) & (coords[:, 0] < h) &
        (coords[:, 1] >= 0) & (coords[:, 1] < w)
    )
    valid_coords = coords[mask]

    # Select only coordinates where skeleton is non-zero
    is_skeleton = skel[valid_coords[:, 0], valid_coords[:, 1]]
    result = valid_coords[is_skeleton != 0]

    # Convert to list of tuples
    return [tuple(pt) for pt in result]