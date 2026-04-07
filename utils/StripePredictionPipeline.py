from __future__ import annotations

import concurrent.futures as cf
import heapq
import math
import multiprocessing as mp
import os
import tempfile
import time
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.windows import Window

from utils import IO
from utils import Prediction as Pred
from utils.GridVectorPipeline import (
    _build_vector_items,
    _current_queue_limit,
    _current_worker_target,
    _dense_threshold,
    _ema,
    _pop_completed,
    _prediction_core_step,
    _submit_available,
    build_aligned_grid,
)
from utils.Tiling import tile_profile_from_parent
from utils.VectorTilePipeline import process_prediction_array_to_gpkg


def _aligned_stripe_inner_rows_px(grid_inner_px: int, step_px: int, config) -> int:
    target_steps = max(1, int(getattr(config, 'stripe_inner_steps', 16) or 16))
    target_rows = target_steps * step_px
    rows = max(target_rows, int(grid_inner_px))
    rows = int(math.ceil(rows / max(int(grid_inner_px), 1))) * int(grid_inner_px)
    return max(int(grid_inner_px), rows)


def build_prediction_stripes(out_profile, grid_inner_px: int, halo_px: int, config):
    step_px = _prediction_core_step(config)
    width = int(out_profile['width'])
    height = int(out_profile['height'])
    inner_rows = _aligned_stripe_inner_rows_px(grid_inner_px, step_px, config)
    stripes = []
    row_idx = 0
    for y0 in range(0, height, inner_rows):
        y1 = min(y0 + inner_rows, height)
        hy0 = max(0, y0 - halo_px)
        hy1 = min(height, y1 + halo_px)
        stripe = {
            'stripe_id': f'stripe_r{row_idx:05d}',
            'index': row_idx,
            'y0': int(y0),
            'y1': int(y1),
            'hy0': int(hy0),
            'hy1': int(hy1),
            'inner_window': Window(0, y0, width, y1 - y0),
            'halo_window': Window(0, hy0, width, hy1 - hy0),
            'arr': None,
            'inner_written': False,
            'release_index': 0,
            'vector_jobs': [],
        }
        stripes.append(stripe)
        row_idx += 1
    return stripes, inner_rows, step_px


def assign_vector_tiles_to_stripes(grid_jobs, stripes):
    out: Dict[int, List] = {s['index']: [] for s in stripes}
    stripe_idx = 0
    for job in grid_jobs:
        while stripe_idx + 1 < len(stripes) and job.y0 >= stripes[stripe_idx]['y1']:
            stripe_idx += 1
        assigned = stripe_idx
        if not (stripes[assigned]['y0'] <= job.y0 < stripes[assigned]['y1']):
            for s in stripes:
                if s['y0'] <= job.y0 < s['y1']:
                    assigned = s['index']
                    break
        out[assigned].append(job)
    for idx, jobs in out.items():
        jobs.sort(key=lambda j: (j.hy1, j.hx0))
    return out


def _activate_stripe(stripe, out_profile):
    if stripe is None or stripe.get('arr') is not None:
        return
    stripe['arr'] = np.zeros(
        (int(stripe['halo_window'].height), int(stripe['halo_window'].width)),
        dtype=np.uint8,
    )
    stripe['profile'] = tile_profile_from_parent(out_profile, stripe['halo_window'])



def _row_groups(pred_jobs):
    row = None
    buf = []
    for job in pred_jobs:
        dst_row = int(job['dst_row'])
        if row is None:
            row = dst_row
        if dst_row != row:
            yield row, buf
            row = dst_row
            buf = []
        buf.append(job)
    if buf:
        yield row, buf



def _job_row_range(job):
    core_h = int(job.get('core_h', 0) or 0)
    return int(job['dst_row']), int(job['dst_row']) + core_h



def _paste_core_to_stripe(stripe, pred_job, pred_core, out_width):
    job_r0 = int(pred_job['dst_row'])
    job_c0 = int(pred_job['dst_col'])
    job_r1 = job_r0 + int(pred_core.shape[0])
    job_c1 = job_c0 + int(pred_core.shape[1])

    r0 = max(job_r0, int(stripe['hy0']))
    r1 = min(job_r1, int(stripe['hy1']))
    c0 = max(job_c0, 0)
    c1 = min(job_c1, int(out_width))
    if r0 >= r1 or c0 >= c1:
        return False

    src_r0 = r0 - job_r0
    src_r1 = src_r0 + (r1 - r0)
    src_c0 = c0 - job_c0
    src_c1 = src_c0 + (c1 - c0)

    dst_r0 = r0 - int(stripe['hy0'])
    dst_r1 = dst_r0 + (r1 - r0)
    dst_c0 = c0
    dst_c1 = dst_c0 + (c1 - c0)

    stripe['arr'][dst_r0:dst_r1, dst_c0:dst_c1] = np.maximum(
        stripe['arr'][dst_r0:dst_r1, dst_c0:dst_c1],
        pred_core[src_r0:src_r1, src_c0:src_c1],
    )
    return True



def _extract_tile_from_stripe(stripe, tile_job):
    local_r0 = int(tile_job.hy0 - stripe['hy0'])
    local_r1 = local_r0 + int(tile_job.halo_window.height)
    local_c0 = int(tile_job.hx0)
    local_c1 = local_c0 + int(tile_job.halo_window.width)
    return np.ascontiguousarray(stripe['arr'][local_r0:local_r1, local_c0:local_c1], dtype=np.uint8)



def _release_ready_tiles(
    stripe,
    row_ready,
    out_profile,
    halo_px,
    config,
    work_dir,
    nonzero_history,
    waiting_heap,
    sequence_box,
    stats,
    alpha,
):
    released_tiles = 0
    split_tiles = 0
    while stripe['release_index'] < len(stripe['vector_jobs']):
        tile_job = stripe['vector_jobs'][stripe['release_index']]
        if int(tile_job.hy1) > int(row_ready):
            break
        tile_arr = _extract_tile_from_stripe(stripe, tile_job)
        tile_profile = tile_profile_from_parent(out_profile, tile_job.halo_window)
        fg_count = int(np.count_nonzero(tile_arr))
        if fg_count > 0:
            nonzero_history.append(fg_count)
            vector_items, split_count = _build_vector_items(
                tile_job,
                tile_arr,
                tile_profile,
                halo_px,
                config,
                work_dir,
                nonzero_history,
                depth=0,
            )
            if vector_items:
                released_tiles += 1
                split_tiles += split_count
                stats['jobs_per_tile_ema'] = _ema(
                    stats.get('jobs_per_tile_ema'), len(vector_items), alpha)
                for item in vector_items:
                    score = int(item['score']) if bool(getattr(config, 'grid_priority_dense_first', True)) else 0
                    heapq.heappush(waiting_heap, (-score, sequence_box[0], item))
                    sequence_box[0] += 1
        stripe['release_index'] += 1
    return released_tiles, split_tiles



def _write_inner_rows(dst, stripe):
    row0 = int(stripe['y0'] - stripe['hy0'])
    row1 = row0 + int(stripe['inner_window'].height)
    arr = np.ascontiguousarray(stripe['arr'][row0:row1, :], dtype=np.uint8)
    dst.write(arr, 1, window=stripe['inner_window'])
    stripe['inner_written'] = True



def run_stripe_binary_pipeline(model, uav_path, stem_path, trees_path, process_type, config):
    if process_type == 'Stems':
        raise ValueError('Stripe vector pipeline is only for Trees/Nodes')

    os.makedirs(os.path.dirname(stem_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(trees_path) or '.', exist_ok=True)

    with rasterio.open(uav_path) as src:
        src_profile = src.profile.copy()
        layout = Pred._resampling_layout((src.height, src.width), src_profile, config)

    out_profile = IO.build_safe_prediction_profile(
        src_profile=src_profile,
        width=layout['out_width'],
        height=layout['out_height'],
        transform=layout['out_transform'],
        compress='DEFLATE' if getattr(config, 'compress_output', True) else None,
        dtype='uint8',
    )
    pred_jobs = list(Pred._iter_tile_jobs(layout, config))
    core_h = int(config.img_height - config.overlap_pred)
    core_w = int(config.img_width - config.overlap_pred)
    for job in pred_jobs:
        job['core_h'] = core_h
        job['core_w'] = core_w

    grid_jobs, grid_inner_px, aligned_m, halo_px, step_px, _, _ = build_aligned_grid(out_profile, config)
    stripes, stripe_inner_rows, _, = build_prediction_stripes(out_profile, grid_inner_px, halo_px, config)
    tiles_by_stripe = assign_vector_tiles_to_stripes(grid_jobs, stripes)
    for stripe in stripes:
        stripe['vector_jobs'] = tiles_by_stripe.get(stripe['index'], [])

    stripe_bytes = int(out_profile['width']) * int(max(s['halo_window'].height for s in stripes))
    stripe_mb = stripe_bytes / (1024.0 * 1024.0)
    print('')
    print('STRIPE BINARY PIPELINE')
    print(f'Prediction core step: {step_px} px')
    print(f'Aligned grid inner size: {grid_inner_px} px (~{aligned_m:.2f} m)')
    print(f'Grid halo: {halo_px} px (~{float(getattr(config, "grid_halo_m", 12.0)):.2f} m)')
    print(f'Stripe inner rows: {stripe_inner_rows} px')
    print(f'Stripes: {len(stripes)}')
    print(f'Estimated active stripe buffer: ~{stripe_mb:.1f} MiB each')
    print(f'Grid tiles: {len(grid_jobs)}')

    work_dir = tempfile.mkdtemp(prefix='winmol_stripe_', dir=os.path.dirname(trees_path) or None)
    tmp_stem_path = IO.atomic_tmp_path(stem_path)
    keep_temp = bool(getattr(config, 'keep_temp_tiles', False))
    min_workers = max(1, int(getattr(config, 'grid_vector_workers_min', 1) or 1))
    max_workers = max(min_workers, int(getattr(config, 'grid_vector_workers_max', getattr(config, 'grid_vector_workers', 2)) or 2))
    progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))
    cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
    spawn_ctx = mp.get_context('spawn')
    alpha = float(getattr(config, 'grid_adaptive_ema', 0.2) or 0.2)
    batch_size = max(1, int(getattr(config, 'prediction_batch_size', None) or getattr(config, 'prediction_batch_gpu', 1)))

    pending = []
    waiting_heap = []
    tile_outputs = []
    nonzero_history: List[int] = []
    target_crs_box = [None]
    start = time.monotonic()
    last_report = start
    total_read_s = 0.0
    total_infer_s = 0.0
    rows_done = 0
    released_tiles_total = 0
    split_tiles = 0
    sequence_box = [0]
    stats = {
        'pred_tile_ema': None,
        'vec_job_ema': None,
        'vec_dense_ema': None,
        'jobs_per_tile_ema': None,
        'completed_jobs': 0,
    }
    pending_pred_s = 0.0
    active_idx = 0

    if stripes:
        _activate_stripe(stripes[0], out_profile)
    if len(stripes) > 1:
        _activate_stripe(stripes[1], out_profile)

    with rasterio.open(uav_path) as src, \
            rasterio.open(tmp_stem_path, 'w', **out_profile) as dst, \
            cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as pool:
        indexes = list(range(1, min(config.n_channels, src.count) + 1))
        for dst_row, row_jobs in _row_groups(pred_jobs):
            row_t0 = time.monotonic()
            for start_idx in range(0, len(row_jobs), batch_size):
                batch_jobs = row_jobs[start_idx:start_idx + batch_size]
                raw_tiles = []
                raw_masks = []
                read_s = 0.0
                for job in batch_jobs:
                    t0 = time.perf_counter()
                    window = Window(job['src_col'], job['src_row'], job['src_width'], job['src_height'])
                    tile = src.read(indexes, window=window, boundless=True, fill_value=0).transpose(1, 2, 0)
                    gdal_mask = src.read_masks(1, window=window, boundless=True) > 0
                    pixel_mask = np.any(tile != 0, axis=2)
                    valid_mask = pixel_mask if np.all(gdal_mask) else (gdal_mask & pixel_mask)
                    read_s += time.perf_counter() - t0
                    raw_tiles.append(tile)
                    raw_masks.append(valid_mask)
                infer0 = time.perf_counter()
                pred_cores = Pred._predict_batch_core(raw_tiles, raw_masks, model, config)
                infer_s = time.perf_counter() - infer0
                total_read_s += read_s
                total_infer_s += infer_s
                for job, pred_core in zip(batch_jobs, pred_cores):
                    # keep at most current and next stripe active; activate one look-ahead when needed
                    for idx in range(active_idx, min(active_idx + 2, len(stripes))):
                        stripe = stripes[idx]
                        if stripe['arr'] is None:
                            _activate_stripe(stripe, out_profile)
                        _paste_core_to_stripe(stripe, job, pred_core, int(out_profile['width']))
            row_elapsed = time.monotonic() - row_t0
            pending_pred_s += row_elapsed
            rows_done += 1
            row_ready = int(dst_row) + core_h

            released_now = 0
            while active_idx < len(stripes):
                stripe = stripes[active_idx]
                if stripe['arr'] is None:
                    break
                rel_tiles, rel_splits = _release_ready_tiles(
                    stripe,
                    row_ready,
                    out_profile,
                    halo_px,
                    config,
                    work_dir,
                    nonzero_history,
                    waiting_heap,
                    sequence_box,
                    stats,
                    alpha,
                )
                released_now += rel_tiles
                split_tiles += rel_splits
                if not stripe['inner_written'] and row_ready >= int(stripe['y1']):
                    _write_inner_rows(dst, stripe)
                if (stripe['inner_written']
                        and row_ready >= int(stripe['hy1'])
                        and stripe['release_index'] >= len(stripe['vector_jobs'])):
                    stripe['arr'] = None
                    active_idx += 1
                    next_idx = active_idx + 1
                    if next_idx < len(stripes) and stripes[next_idx]['arr'] is None:
                        _activate_stripe(stripes[next_idx], out_profile)
                    continue
                break

            if released_now > 0:
                sample = pending_pred_s / max(released_now, 1)
                stats['pred_tile_ema'] = _ema(stats.get('pred_tile_ema'), sample, alpha)
                pending_pred_s = 0.0
            released_tiles_total += released_now

            worker_target = _current_worker_target(config, stats)
            queue_limit = _current_queue_limit(config, worker_target)
            _submit_available(waiting_heap, pending, pool, worker_target, cfg_dict, process_type)
            while len(pending) + len(waiting_heap) > queue_limit and pending:
                dense_cutoff = _dense_threshold(nonzero_history, config)
                _pop_completed(
                    pending,
                    tile_outputs,
                    keep_temp,
                    out_profile,
                    target_crs_box,
                    stats,
                    alpha,
                    dense_cutoff,
                )
                worker_target = _current_worker_target(config, stats)
                _submit_available(waiting_heap, pending, pool, worker_target, cfg_dict, process_type)

            now = time.monotonic()
            if rows_done == 1 or rows_done == layout['y_tiles'] or (now - last_report) >= progress_interval_s:
                elapsed = max(now - start, 1e-9)
                rate = rows_done / elapsed
                eta_s = (layout['y_tiles'] - rows_done) / rate if rate > 0 else float('inf')
                dense_cutoff = _dense_threshold(nonzero_history, config)
                print(
                    f'Prediction rows completed {rows_done}/{layout["y_tiles"]} | {rows_done / layout["y_tiles"]:.1%} | '
                    f'{rate * 60:.2f} rows/min | ETA {Pred._format_eta(eta_s)} | '
                    f'avg read {total_read_s / max(rows_done, 1):.3f}s infer {total_infer_s / max(rows_done, 1):.3f}s | '
                    f'vector tiles released {released_tiles_total}/{len(grid_jobs)} | '
                    f'workers {worker_target}/{max_workers} | pending {len(pending)} | queued {len(waiting_heap)} | '
                    f'dense cutoff {0 if math.isinf(dense_cutoff) else int(dense_cutoff)} | splits {split_tiles}',
                    flush=True,
                )
                last_report = now

        final_ready = int(out_profile['height'])
        for stripe in stripes:
            if stripe.get('arr') is None:
                continue
            rel_tiles, rel_splits = _release_ready_tiles(
                stripe,
                final_ready,
                out_profile,
                halo_px,
                config,
                work_dir,
                nonzero_history,
                waiting_heap,
                sequence_box,
                stats,
                alpha,
            )
            released_tiles_total += rel_tiles
            split_tiles += rel_splits
            if not stripe['inner_written']:
                _write_inner_rows(dst, stripe)
        if pending_pred_s > 0 and released_tiles_total > 0:
            stats['pred_tile_ema'] = _ema(stats.get('pred_tile_ema'), pending_pred_s / max(1, released_tiles_total), alpha)
        while waiting_heap or pending:
            worker_target = _current_worker_target(config, stats)
            _submit_available(waiting_heap, pending, pool, worker_target, cfg_dict, process_type)
            if pending:
                dense_cutoff = _dense_threshold(nonzero_history, config)
                _pop_completed(
                    pending,
                    tile_outputs,
                    keep_temp,
                    out_profile,
                    target_crs_box,
                    stats,
                    alpha,
                    dense_cutoff,
                )
            elif waiting_heap:
                _submit_available(waiting_heap, pending, pool, worker_target, cfg_dict, process_type)

    IO.finalize_raster(tmp_stem_path, stem_path)

    out_path = trees_path if trees_path.lower().endswith('.gpkg') else f'{trees_path}.gpkg'
    if tile_outputs:
        IO.merge_selected_tile_results(
            tile_outputs,
            out_path,
            out_profile,
            keep_temp=keep_temp,
        )
    else:
        if os.path.exists(out_path):
            os.remove(out_path)
        print('')
        print('MERGE SUMMARY')
        print('Tiles processed:       0')
        print('Total stems written:   0')
        print('Total nodes written:   0')
        print('Total vectors written: 0')
        print(f'No output GPKG created: {out_path} (0 features written)')

    print('')
    print('STRIPE PIPELINE SUMMARY')
    print(f'Prediction row groups: {layout["y_tiles"]}')
    print(f'Grid tiles released:   {released_tiles_total}')
    print(f'Vector jobs completed: {stats["completed_jobs"]}')
    print(f'Dense tiles split:     {split_tiles}')
    print(f'Stripe temp directory: {work_dir}')

    if not keep_temp:
        try:
            for name in os.listdir(work_dir):
                os.remove(os.path.join(work_dir, name))
            os.rmdir(work_dir)
        except Exception:
            pass
    return out_path
