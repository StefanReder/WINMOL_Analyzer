from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import time

import numpy as np

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


def _vector_summary(tile_label, fg_count, segment_count, stem_count, timings, output_path):
    print(
        f'VECTOR TILE {tile_label} | fg {fg_count} | segments {segment_count} '
        f'| stems {stem_count} | '
        f'skel {timings["skel_s"]:.3f}s restore {timings["restore_s"]:.3f}s '
        f'build {timings["build_s"]:.3f}s connect {timings["connect_s"]:.3f}s '
        f'quant {timings["quant_s"]:.3f}s write {timings["write_s"]:.3f}s '
        f'| total {timings["total_s"]:.3f}s | output {os.path.basename(output_path) if output_path else "none"}',
        flush=True,
    )


def _run_vector_pipeline(pred, profile, config, process_type: str, output_prefix: str, tile_label: str):
    fg_count = int(np.count_nonzero(pred))
    if fg_count <= 0:
        return None

    timings = {
        'skel_s': 0.0,
        'restore_s': 0.0,
        'build_s': 0.0,
        'connect_s': 0.0,
        'quant_s': 0.0,
        'write_s': 0.0,
        'total_s': 0.0,
    }
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    segments = Skel.find_segments(pred, config, profile)
    timings['skel_s'] = time.perf_counter() - t0
    if not segments:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(tile_label, fg_count, 0, 0, timings, None)
        return None
    segment_count = len(segments)

    t0 = time.perf_counter()
    segments = Vec.restore_geoinformation(segments, config, profile)
    timings['restore_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stems = Vec.build_stem_parts(segments)
    timings['build_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(tile_label, fg_count, segment_count, 0, timings, None)
        return None

    t0 = time.perf_counter()
    stems = Vec.connect_stems(stems, config)
    timings['connect_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(tile_label, fg_count, segment_count, 0, timings, None)
        return None

    Vec.rebuild_endnodes_from_stems(stems)

    t0 = time.perf_counter()
    stems = Quant.quantify_stems(stems, pred, profile, config=config)
    timings['quant_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(tile_label, fg_count, segment_count, 0, timings, None)
        return None
    stem_count = len(stems)

    t0 = time.perf_counter()
    output_path = export_tile_results(stems, profile, process_type, output_prefix)
    timings['write_s'] = time.perf_counter() - t0
    timings['total_s'] = time.perf_counter() - total_t0

    if bool(getattr(config, 'vector_summary_log', True)):
        _vector_summary(tile_label, fg_count, segment_count, stem_count, timings, output_path)
    return output_path


def _run_with_debug_control(fn, config, *args, **kwargs):
    if bool(getattr(config, 'vector_debug', False)):
        return fn(*args, **kwargs)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn(*args, **kwargs)


def process_prediction_array_to_gpkg(
    pred_arr,
    profile,
    config_dict,
    process_type: str,
    output_prefix: str,
):
    config = Config()
    for key, value in (config_dict or {}).items():
        try:
            setattr(config, key, value)
        except Exception:
            pass
    config.cpu_workers = max(1, int(getattr(config, 'cpu_workers', 1) or 1))
    config.vector_tile_workers = 1

    pred = np.asarray(pred_arr)
    if pred.size == 0 or not np.any(pred >= 1):
        return None

    tile_label = os.path.basename(output_prefix)
    return _run_with_debug_control(
        _run_vector_pipeline,
        config,
        pred,
        profile,
        config,
        process_type,
        output_prefix,
        tile_label,
    )


def process_prediction_tile(
    pred_tile_path: str,
    config,
    process_type: str,
    output_prefix: str,
):
    pred, profile = load_stem_map(pred_tile_path)
    pred_arr = np.asarray(pred)
    if pred_arr.size == 0 or not np.any(pred_arr >= 1):
        return None
    tile_label = os.path.splitext(os.path.basename(pred_tile_path))[0]
    return _run_with_debug_control(
        _run_vector_pipeline,
        config,
        pred,
        profile,
        config,
        process_type,
        output_prefix,
        tile_label,
    )


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
