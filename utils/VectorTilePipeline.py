from __future__ import annotations

import multiprocessing as mp
import os
from functools import partial

from utils.IO import load_stem_map, write_all_layers_to_gpkg, write_stems_to_gpkg
import utils.Skeletonization as Skel
import utils.Vectorization as Vec
import utils.Quantification as Quant


def export_tile_results(stems, profile, process_type: str, output_prefix: str):
    if process_type == 'Trees':
        return write_stems_to_gpkg(stems, profile, output_prefix)
    return write_all_layers_to_gpkg(stems, profile, output_prefix)


def process_prediction_tile(
    pred_tile_path: str,
    config,
    process_type: str,
    output_prefix: str,
):
    pred, profile = load_stem_map(pred_tile_path)
    print(f"Processing prediction tile: {pred_tile_path}")
    segments = Skel.find_segments(pred, config, profile)
    segments = Vec.restore_geoinformation(segments, config, profile)
    stems = Vec.build_stem_parts(segments)
    stems = Vec.connect_stems(stems, config)
    Vec.rebuild_endnodes_from_stems(stems)
    stems = Quant.quantify_stems(stems, pred, profile)
    return export_tile_results(stems, profile, process_type, output_prefix)


def _process_prediction_tile_star(args):
    return process_prediction_tile(*args)


def process_prediction_tiles(
    pred_tile_paths: list[str],
    config,
    process_type: str,
    output_dir: str,
    cpu_workers: int,
):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for pred_tile_path in pred_tile_paths:
        name = os.path.splitext(os.path.basename(pred_tile_path))[0].replace('_roi_stem_map', '')
        output_prefix = os.path.join(output_dir, name)
        tasks.append((pred_tile_path, config, process_type, output_prefix))

    if cpu_workers and cpu_workers > 1 and len(tasks) > 1:
        with mp.Pool(cpu_workers) as pool:
            return pool.map(_process_prediction_tile_star, tasks)
    return [_process_prediction_tile_star(t) for t in tasks]
