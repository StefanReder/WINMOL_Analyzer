from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from classes.Config import Config
from utils.IO \
    import load_stem_map, write_stems_to_gpkg
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



def _vector_result(tile_label, output_path, fg_count, segment_count, stem_count, timings):
    return {
        'tile_label': tile_label,
        'gpkg_path': output_path,
        'fg_count': int(fg_count or 0),
        'segment_count': int(segment_count or 0),
        'stem_count': int(stem_count or 0),
        'timings': dict(timings),
    }


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
        return _vector_result(tile_label, None, fg_count, 0, 0, timings)
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
        return _vector_result(tile_label, None, fg_count, segment_count, 0, timings)

    t0 = time.perf_counter()
    stems = Vec.connect_stems(stems, config)
    Vec.rebuild_endnodes_from_stems(stems)
    timings['connect_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(tile_label, fg_count, segment_count, 0, timings, None)
        return _vector_result(tile_label, None, fg_count, segment_count, 0, timings)

    t0 = time.perf_counter()
    stems = Quant.quantify_stems(stems, pred, profile, config=config)
    timings['quant_s'] = time.perf_counter() - t0
    if not stems:
        timings['total_s'] = time.perf_counter() - total_t0
        if bool(getattr(config, 'vector_summary_log', True)):
            _vector_summary(tile_label, fg_count, segment_count, 0, timings, None)
        return _vector_result(tile_label, None, fg_count, segment_count, 0, timings)
    stem_count = len(stems)

    t0 = time.perf_counter()
    output_path = write_stems_to_gpkg(stems, profile, output_prefix)
    timings['write_s'] = time.perf_counter() - t0
    timings['total_s'] = time.perf_counter() - total_t0
    output_exists = bool(output_path) and os.path.exists(output_path)
    print(
        f'VECTOR TILE {tile_label} | output_exists {output_exists} | path {output_path}',
        flush=True,
    )

    if bool(getattr(config, 'vector_summary_log', True)):
        _vector_summary(tile_label, fg_count, segment_count, stem_count, timings, output_path)
    return _vector_result(
        tile_label,
        output_path,
        fg_count,
        segment_count,
        stem_count,
        timings,
    )


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
    config.cpu_workers = 1
    config.vector_tile_workers = 1

    pred = np.asarray(pred_arr)
    if pred.size == 0 or not np.any(pred >= 1):
        return None

    tile_label = os.path.basename(output_prefix)
    return _run_vector_pipeline(
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
    return _run_vector_pipeline(
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
    total_workers = max(1, int(cpu_workers or getattr(config, 'cpu_workers', 1) or 1))
    requested_outer = int(getattr(config, 'vector_tile_workers', 1) or 1)
    if requested_outer <= 1:
        if total_workers >= 36:
            requested_outer = 6
        elif total_workers >= 24:
            requested_outer = 4
        elif total_workers >= 12:
            requested_outer = 2
        else:
            requested_outer = 1
    outer_workers = max(1, min(len(pred_tile_paths), requested_outer))
    inner_workers = max(1, total_workers // max(outer_workers, 1))
    progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))

    tasks = []
    for pred_tile_path in pred_tile_paths:
        name = os.path.splitext(os.path.basename(pred_tile_path))[0].replace('_roi_stem_map', '')
        output_prefix = os.path.join(output_dir, name)
        tile_cfg = _clone_config(
            config,
            cpu_workers=inner_workers,
            vector_tile_workers=1,
        )
        tasks.append((pred_tile_path, tile_cfg, process_type, output_prefix))

    print(
        f'VECTOR SCHEDULER | tiles {len(tasks)} | outer_workers {outer_workers} | '
        f'inner_workers {inner_workers} | cpu_budget {total_workers}',
        flush=True,
    )

    start = time.monotonic()
    last_report = start
    completed = 0
    results = [None] * len(tasks)

    def _emit_progress(now_ts):
        rate = completed / max(now_ts - start, 1e-9)
        eta = (len(tasks) - completed) / rate if rate > 0 else float('inf')
        print(
            f"Vector tiles {completed}/{len(tasks)} | {completed / max(len(tasks), 1):.1%} | "
            f"{rate * 60:.2f} tiles/min | ETA {eta / 60:.1f} min",
            flush=True,
        )

    if outer_workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            results[idx - 1] = _process_prediction_tile_star(task)
            completed = idx
            now = time.monotonic()
            if idx == len(tasks) or (now - last_report) >= progress_interval_s:
                _emit_progress(now)
                last_report = now
    else:
        with ThreadPoolExecutor(max_workers=outer_workers) as executor:
            future_to_idx = {
                executor.submit(_process_prediction_tile_star, task): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                now = time.monotonic()
                if completed == len(tasks) or (now - last_report) >= progress_interval_s:
                    _emit_progress(now)
                    last_report = now

    wall_s = time.monotonic() - start
    valid_results = [r for r in results if r]
    tiles_with_output = sum(1 for r in valid_results if r.get('gpkg_path'))
    total_fg = sum(int(r.get('fg_count', 0) or 0) for r in valid_results)
    total_segments = sum(int(r.get('segment_count', 0) or 0) for r in valid_results)
    total_stems = sum(int(r.get('stem_count', 0) or 0) for r in valid_results)
    timings = {
        'skel_s': 0.0,
        'restore_s': 0.0,
        'build_s': 0.0,
        'connect_s': 0.0,
        'quant_s': 0.0,
        'write_s': 0.0,
        'total_s': 0.0,
    }
    for result in valid_results:
        for key in timings:
            timings[key] += float(result.get('timings', {}).get(key, 0.0) or 0.0)

    denom = max(len(valid_results), 1)
    print(
        f'VECTOR SUMMARY | tiles_total {len(tasks)} | completed {completed} | '
        f'tiles_with_output {tiles_with_output} | fg_total {total_fg} | '
        f'segments_total {total_segments} | stems_total {total_stems}',
        flush=True,
    )
    print(
        f'VECTOR STEP TOTALS | skel_s {timings["skel_s"]:.3f} | restore_s {timings["restore_s"]:.3f} | '
        f'build_s {timings["build_s"]:.3f} | connect_s {timings["connect_s"]:.3f} | '
        f'quant_s {timings["quant_s"]:.3f} | write_s {timings["write_s"]:.3f} | '
        f'total_s {timings["total_s"]:.3f}',
        flush=True,
    )
    print(
        f'VECTOR STEP AVGS | skel_s {timings["skel_s"] / denom:.3f} | restore_s {timings["restore_s"] / denom:.3f} | '
        f'build_s {timings["build_s"] / denom:.3f} | connect_s {timings["connect_s"] / denom:.3f} | '
        f'quant_s {timings["quant_s"] / denom:.3f} | write_s {timings["write_s"] / denom:.3f} | '
        f'total_s {timings["total_s"] / denom:.3f}',
        flush=True,
    )
    print(
        f'VECTOR WALL | outer_workers {outer_workers} | inner_workers {inner_workers} | wall_s {wall_s:.3f}',
        flush=True,
    )
    return results
