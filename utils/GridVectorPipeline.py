from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import tempfile
import time
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from classes.Config import Config
from utils import IO
from utils import Prediction as Pred
import utils.Quantification as Quant
import utils.Skeletonization as Skel
import utils.Vectorization as Vec
from utils.Tiling import build_tile_grid, meters_to_pixels, tile_profile_from_parent
from utils.VectorTilePipeline import process_prediction_array_to_gpkg


def _config_from_dict(config_dict: dict) -> Config:
    cfg = Config()
    for key, value in config_dict.items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    return cfg


def _prediction_core_step(config) -> int:
    return int(config.img_width - config.overlap_pred)


def aligned_grid_inner_px(profile, config) -> Tuple[int, float, int]:
    px_size = max(abs(profile['transform'][0]), abs(profile['transform'][4]))
    if px_size <= 0:
        px_size = abs(profile['transform'][0]) or 1.0
    target_m = float(getattr(config, 'grid_inner_m', 250.0))
    target_px = max(1, int(round(target_m / px_size)))
    step_px = _prediction_core_step(config)
    steps = max(1, int(round(target_px / max(step_px, 1))))
    inner_px = steps * step_px
    return inner_px, inner_px * px_size, step_px


def build_aligned_grid(profile, config):
    inner_px, aligned_m, step_px = aligned_grid_inner_px(profile, config)
    halo_px = meters_to_pixels(
        float(getattr(config, 'grid_halo_m', 12.0)),
        float(profile['transform'][0]),
        float(profile['transform'][4]),
    )
    jobs = build_tile_grid(
        int(profile['width']),
        int(profile['height']),
        inner_px,
        halo_px,
    )
    n_cols = int(math.ceil(int(profile['width']) / max(inner_px, 1)))
    n_rows = int(math.ceil(int(profile['height']) / max(inner_px, 1)))
    return jobs, inner_px, aligned_m, halo_px, step_px, n_rows, n_cols


def _job_intersection(job: dict, tile_job) -> Tuple[int, int, int, int] | None:
    core_h = int(job.get('core_h', 0) or 0)
    core_w = int(job.get('core_w', 0) or 0)
    if core_h <= 0 or core_w <= 0:
        return None
    job_r0 = int(job['dst_row'])
    job_c0 = int(job['dst_col'])
    job_r1 = job_r0 + core_h
    job_c1 = job_c0 + core_w

    tile_r0 = int(tile_job.hy0)
    tile_c0 = int(tile_job.hx0)
    tile_r1 = int(tile_job.hy1)
    tile_c1 = int(tile_job.hx1)

    r0 = max(job_r0, tile_r0)
    c0 = max(job_c0, tile_c0)
    r1 = min(job_r1, tile_r1)
    c1 = min(job_c1, tile_c1)
    if r0 >= r1 or c0 >= c1:
        return None
    return r0, c0, r1, c1


def assign_prediction_jobs_to_grid(pred_jobs, grid_jobs, n_rows, n_cols, inner_px, halo_px):
    out: Dict[str, List[dict]] = {job.tile_id: [] for job in grid_jobs}
    grid_index = {}
    for idx, job in enumerate(grid_jobs):
        grid_index[(idx // n_cols, idx % n_cols)] = job.tile_id

    for job in pred_jobs:
        core_h = int(job.get('core_h', 0) or 0)
        core_w = int(job.get('core_w', 0) or 0)
        r0 = int(job['dst_row'])
        c0 = int(job['dst_col'])
        r1 = r0 + core_h
        c1 = c0 + core_w

        gr0 = max(0, int(math.floor((r0 - halo_px) / max(inner_px, 1))))
        gc0 = max(0, int(math.floor((c0 - halo_px) / max(inner_px, 1))))
        gr1 = min(n_rows - 1, int(math.floor((r1 + halo_px - 1) / max(inner_px, 1))))
        gc1 = min(n_cols - 1, int(math.floor((c1 + halo_px - 1) / max(inner_px, 1))))
        for gr in range(gr0, gr1 + 1):
            for gc in range(gc0, gc1 + 1):
                tile_id = grid_index.get((gr, gc))
                if tile_id is not None:
                    out[tile_id].append(job)
    return out


def _read_raw_prediction_inputs(src, indexes, batch_jobs):
    raw_tiles = []
    raw_masks = []
    read_s = 0.0
    for job in batch_jobs:
        t0 = time.perf_counter()
        window = Window(job['src_col'], job['src_row'], job['src_width'], job['src_height'])
        tile = src.read(
            indexes,
            window=window,
            boundless=True,
            fill_value=0,
        ).transpose(1, 2, 0)
        gdal_mask = src.read_masks(
            1,
            window=window,
            boundless=True,
        ) > 0
        pixel_mask = np.any(tile != 0, axis=2)
        valid_mask = pixel_mask if np.all(gdal_mask) else (gdal_mask & pixel_mask)
        read_s += time.perf_counter() - t0
        raw_tiles.append(tile)
        raw_masks.append(valid_mask)
    return raw_tiles, raw_masks, read_s


def _paste_prediction_core(tile_arr, tile_job, pred_job, pred_core):
    inter = _job_intersection(pred_job, tile_job)
    if inter is None:
        return
    r0, c0, r1, c1 = inter
    src_r0 = r0 - int(pred_job['dst_row'])
    src_c0 = c0 - int(pred_job['dst_col'])
    src_r1 = src_r0 + (r1 - r0)
    src_c1 = src_c0 + (c1 - c0)
    dst_r0 = r0 - int(tile_job.hy0)
    dst_c0 = c0 - int(tile_job.hx0)
    dst_r1 = dst_r0 + (r1 - r0)
    dst_c1 = dst_c0 + (c1 - c0)
    tile_arr[dst_r0:dst_r1, dst_c0:dst_c1] = np.maximum(
        tile_arr[dst_r0:dst_r1, dst_c0:dst_c1],
        pred_core[src_r0:src_r1, src_c0:src_c1],
    )


def _predict_binary_grid_tile(uav_path, model, grid_job, pred_jobs, config):
    tile_arr = np.zeros(
        (int(grid_job.halo_window.height), int(grid_job.halo_window.width)),
        dtype=np.uint8,
    )
    tile_profile = None
    stats = {'read_s': 0.0, 'infer_s': 0.0, 'jobs': len(pred_jobs)}
    if not pred_jobs:
        with rasterio.open(uav_path) as src:
            layout = Pred._resampling_layout((src.height, src.width), src.profile.copy(), config)
            out_profile = IO.build_safe_prediction_profile(
                src_profile=src.profile.copy(),
                width=layout['out_width'],
                height=layout['out_height'],
                transform=layout['out_transform'],
                compress='DEFLATE' if getattr(config, 'compress_output', True) else None,
                dtype='uint8',
            )
        return tile_arr, tile_profile_from_parent(out_profile, grid_job.halo_window), stats

    batch_size = max(1, int(getattr(config, 'prediction_batch_size', None) or getattr(config, 'prediction_batch_gpu', 1)))
    with rasterio.open(uav_path) as src:
        indexes = list(range(1, min(config.n_channels, src.count) + 1))
        base_profile = src.profile.copy()
        layout = Pred._resampling_layout((src.height, src.width), base_profile, config)
        out_profile = IO.build_safe_prediction_profile(
            src_profile=base_profile,
            width=layout['out_width'],
            height=layout['out_height'],
            transform=layout['out_transform'],
            compress='DEFLATE' if getattr(config, 'compress_output', True) else None,
            dtype='uint8',
        )
        tile_profile = tile_profile_from_parent(out_profile, grid_job.halo_window)

        for start in range(0, len(pred_jobs), batch_size):
            batch_jobs = pred_jobs[start:start + batch_size]
            raw_tiles, raw_masks, read_s = _read_raw_prediction_inputs(src, indexes, batch_jobs)
            infer0 = time.perf_counter()
            pred_cores = Pred._predict_batch_core(raw_tiles, raw_masks, model, config)
            infer_s = time.perf_counter() - infer0
            stats['read_s'] += read_s
            stats['infer_s'] += infer_s
            for job, pred_core in zip(batch_jobs, pred_cores):
                _paste_prediction_core(tile_arr, grid_job, job, pred_core)
    return tile_arr, tile_profile, stats




def _wait_one(pending):
    done, _ = cf.wait([f for _, f in pending], return_when=cf.FIRST_COMPLETED)
    completed = []
    remaining = []
    for item in pending:
        if item[1] in done:
            completed.append(item)
        else:
            remaining.append(item)
    return completed, remaining


def _inner_slice(tile_job):
    row0 = int(tile_job.y0 - tile_job.hy0)
    col0 = int(tile_job.x0 - tile_job.hx0)
    row1 = row0 + int(tile_job.inner_window.height)
    col1 = col0 + int(tile_job.inner_window.width)
    return row0, row1, col0, col1


def run_binary_grid_pipeline(model, uav_path, stem_path, trees_path, process_type, config):
    if process_type == 'Stems':
        raise ValueError('Grid vector pipeline is only for Trees/Nodes')

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

    grid_jobs, inner_px, aligned_m, halo_px, step_px, n_rows, n_cols = build_aligned_grid(out_profile, config)
    jobs_by_grid = assign_prediction_jobs_to_grid(pred_jobs, grid_jobs, n_rows, n_cols, inner_px, halo_px)

    print('')
    print('GRID BINARY PIPELINE')
    print(f'Prediction core step: {step_px} px')
    print(f'Aligned grid inner size: {inner_px} px (~{aligned_m:.2f} m)')
    print(f'Grid halo: {halo_px} px (~{float(getattr(config, "grid_halo_m", 12.0)):.2f} m)')
    print(f'Grid tiles: {len(grid_jobs)}')

    work_dir = tempfile.mkdtemp(prefix='winmol_grid_', dir=os.path.dirname(trees_path) or None)
    tmp_stem_path = IO.atomic_tmp_path(stem_path)
    keep_temp = bool(getattr(config, 'keep_temp_tiles', False))

    vector_workers = max(1, int(getattr(config, 'grid_vector_workers', 1) or 1))
    inflight_limit = max(vector_workers, int(getattr(config, 'grid_inflight_tiles', 2) or 2))
    progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))
    cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
    spawn_ctx = mp.get_context('spawn')

    pending = []
    tile_outputs = []
    start = time.monotonic()
    last_report = start
    total_read_s = 0.0
    total_infer_s = 0.0
    predicted_tiles = 0
    submitted_tiles = 0

    with rasterio.open(tmp_stem_path, 'w', **out_profile) as dst, \
            cf.ProcessPoolExecutor(max_workers=vector_workers,
                                   mp_context=spawn_ctx) as pool:
        for idx, grid_job in enumerate(grid_jobs, start=1):
            tile_arr, tile_profile, stats = _predict_binary_grid_tile(
                uav_path, model, grid_job, jobs_by_grid.get(grid_job.tile_id, []), config)
            total_read_s += float(stats.get('read_s', 0.0))
            total_infer_s += float(stats.get('infer_s', 0.0))

            r0, r1, c0, c1 = _inner_slice(grid_job)
            inner_arr = np.ascontiguousarray(tile_arr[r0:r1, c0:c1], dtype=np.uint8)
            dst.write(inner_arr, 1, window=grid_job.inner_window)
            predicted_tiles += 1

            if np.any(tile_arr):
                output_prefix = os.path.join(work_dir, grid_job.tile_id)
                future = pool.submit(
                    process_prediction_array_to_gpkg,
                    tile_arr,
                    tile_profile,
                    cfg_dict,
                    process_type,
                    output_prefix,
                )
                pending.append((grid_job, future))
                submitted_tiles += 1

            while len(pending) >= inflight_limit:
                completed, pending = _wait_one(pending)
                for done_job, fut in completed:
                    gpkg_path = fut.result()
                    if gpkg_path is not None:
                        tile_outputs.append((done_job, gpkg_path))

            now = time.monotonic()
            if idx == 1 or idx == len(grid_jobs) or (now - last_report) >= progress_interval_s:
                elapsed = max(now - start, 1e-9)
                rate = idx / elapsed
                eta_s = (len(grid_jobs) - idx) / rate if rate > 0 else float('inf')
                print(
                    f'Grid tiles predicted {idx}/{len(grid_jobs)} | {idx / len(grid_jobs):.1%} | '
                    f'{rate * 60:.2f} tiles/min | ETA {Pred._format_eta(eta_s)} | '
                    f'avg read {total_read_s / max(idx, 1):.3f}s infer {total_infer_s / max(idx, 1):.3f}s | '
                    f'pending vector {len(pending)}',
                    flush=True,
                )
                last_report = now

        while pending:
            completed, pending = _wait_one(pending)
            for done_job, fut in completed:
                gpkg_path = fut.result()
                if gpkg_path is not None:
                    tile_outputs.append((done_job, gpkg_path))

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
    print('GRID PIPELINE SUMMARY')
    print(f'Grid tiles predicted:  {predicted_tiles}')
    print(f'Grid tiles submitted:  {submitted_tiles}')
    print(f'Tile temp directory:   {work_dir}')

    if not keep_temp:
        try:
            for name in os.listdir(work_dir):
                if name.lower().endswith('.gpkg'):
                    os.remove(os.path.join(work_dir, name))
            os.rmdir(work_dir)
        except Exception:
            pass
    return out_profile
