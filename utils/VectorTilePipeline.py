from __future__ import annotations

import multiprocessing as mp
import os
import time

from classes.Config import Config
from utils.IO \
    import load_stem_map, write_all_layers_to_gpkg, write_stems_to_gpkg
import utils.Skeletonization as Skel
import utils.Vectorization as Vec
import utils.Quantification as Quant


def _clone_config(config, **updates):
    cfg = Config()
    for key, value in getattr(config, 'to_dict', lambda: {})().items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


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
    print(f"Processing prediction tile: {pred_tile_path}", flush=True)
    segments = Skel.find_segments(pred, config, profile)
    segments = Vec.restore_geoinformation(segments, config, profile)
    stems = Vec.build_stem_parts(segments)
    stems = Vec.connect_stems(stems, config)
    Vec.rebuild_endnodes_from_stems(stems)
    stems = Quant.quantify_stems(stems, pred, profile, config=config)
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
    total_workers = \
        max(1, int(cpu_workers or getattr(config, 'cpu_workers', 1) or 1))
    requested_tile_workers = \
        max(1, int(getattr(config, 'vector_tile_workers', 1) or 1))
    tile_workers = \
        max(1, min(requested_tile_workers,
                   len(pred_tile_paths), total_workers))
    inner_workers = max(1, total_workers // tile_workers)
    progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))

    tasks = []
    for pred_tile_path in pred_tile_paths:
        name = os.path.splitext(os.path.basename(
            pred_tile_path))[0].replace('_roi_stem_map', '')
        output_prefix = os.path.join(output_dir, name)
        tile_cfg = _clone_config(
            config, cpu_workers=inner_workers, vector_tile_workers=1)
        tasks.append((pred_tile_path, tile_cfg, process_type, output_prefix))

    start = time.monotonic()
    last_report = start
    results = []

    if tile_workers <= 1 or len(tasks) <= 1:
        for idx, task in enumerate(tasks, start=1):
            results.append(_process_prediction_tile_star(task))
            now = time.monotonic()
            if idx == len(tasks) or (now - last_report) >= progress_interval_s:
                rate = idx / max(now - start, 1e-9)
                eta = (len(tasks) - idx) / rate if rate > 0 else float('inf')
                print(f"Vector tiles {idx}/{len(tasks)} | "
                      f"{idx / len(tasks):.1%} | {rate * 60:.2f} tiles/min "
                      f"| ETA {eta / 60:.1f} min", flush=True)
                last_report = now
        return results

    with mp.Pool(tile_workers) as pool:
        for idx, result in enumerate(
            pool.imap_unordered(_process_prediction_tile_star, tasks),
            start=1,
        ):
            results.append(result)
            now = time.monotonic()
            if idx == len(tasks) or (now - last_report) >= progress_interval_s:
                rate = idx / max(now - start, 1e-9)
                eta = (len(tasks) - idx) / rate if rate > 0 else float('inf')
                print(f"Vector tiles {idx}/{len(tasks)} | "
                      f"{idx / len(tasks):.1%} | {rate * 60:.2f}"
                      f" tiles/min | ETA {eta / 60:.1f} min", flush=True)
                last_report = now
    return results
