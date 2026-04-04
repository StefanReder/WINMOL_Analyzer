from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import List

import numpy as np
import rasterio
from rasterio.windows import Window

from classes.Config import Config
from utils import IO
from utils.Prediction import _iter_tile_jobs, _prepare_inference_batch, \
    _resampling_layout, _format_eta


def _config_from_dict(config_dict: dict) -> Config:
    cfg = Config()
    for key, value in config_dict.items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    return cfg


def _predict_batch(raw_tiles, raw_masks, model, config):
    tile_tensor, mask_resized = _prepare_inference_batch(
        raw_tiles, raw_masks, config)
    pred = model.predict_on_batch(tile_tensor)
    crop = config.overlap_pred // 2
    cores = []
    for idx in range(pred.shape[0]):
        pred_core = pred[idx, crop:(config.img_width - crop),
                         crop:(config.img_width - crop), 0]
        mask_core = mask_resized[idx, crop:(config.img_width - crop),
                                 crop:(config.img_width - crop), 0] > 0.5
        cores.append(np.ascontiguousarray((pred_core * mask_core)
                                          .astype(np.float32)))
    return cores


def _group_jobs(jobs, batch_size):
    batch = []
    for job in jobs:
        batch.append(job)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def prediction_worker(
    gpu_id: int,
    model_path: str,
    input_raster: str,
    jobs: List[dict],
    results,
    config_dict: dict,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import tensorflow as tf
    from utils.IO import load_model_from_path

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    cfg = _config_from_dict(config_dict)
    model = load_model_from_path(model_path)
    batch_size = max(1, int(getattr(cfg, 'prediction_batch_size', None)
                            or getattr(cfg, 'prediction_batch_gpu', 4)))

    with rasterio.open(input_raster) as src:
        indexes = list(range(1, min(cfg.n_channels, src.count) + 1))
        for batch_jobs in _group_jobs(jobs, batch_size):
            raw_tiles = []
            raw_masks = []
            read_s = 0.0
            for job in batch_jobs:
                t0 = time.perf_counter()
                window = Window(job['src_col'], job['src_row'],
                                job['src_width'], job['src_height'])
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

                if np.all(gdal_mask):
                    valid_mask = pixel_mask
                else:
                    valid_mask = gdal_mask & pixel_mask

                read_s += time.perf_counter() - t0
                raw_tiles.append(tile)
                raw_masks.append(valid_mask)
            infer0 = time.perf_counter()
            pred_cores = _predict_batch(raw_tiles, raw_masks, model, cfg)
            infer_s = time.perf_counter() - infer0
            for job, pred_core in zip(batch_jobs, pred_cores):
                results.put({
                    'row_off': job['dst_row'],
                    'col_off': job['dst_col'],
                    'array': pred_core,
                    'read_s': read_s / max(len(batch_jobs), 1),
                    'infer_s': infer_s / max(len(batch_jobs), 1),
                })
    results.put({'done': True, 'gpu_id': gpu_id})


def run_multi_gpu_prediction(
    model_path: str,
    input_raster: str,
    output_raster: str,
    tile_jobs,
    gpu_ids: list[int],
    config,
):
    if not gpu_ids:
        raise ValueError('multi_gpu_prediction requires at least one GPU id')

    ctx = mp.get_context('spawn')
    result_q = ctx.Queue(maxsize=max(8, len(gpu_ids) * 8))

    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        layout = _resampling_layout((src.height, src.width), profile, config)
        all_jobs = list(_iter_tile_jobs(layout, config)) \
            if tile_jobs is None else list(tile_jobs)

    shards = [[] for _ in gpu_ids]
    for idx, job in enumerate(all_jobs):
        shards[idx % len(gpu_ids)].append(job)

    out_profile = IO.build_safe_prediction_profile(
        src_profile=profile,
        width=layout['out_width'],
        height=layout['out_height'],
        transform=layout['out_transform'],
        compress='DEFLATE' if getattr(config, 'compress_output', True)
        else None,
    )
    tmp_path = IO.atomic_tmp_path(output_raster)

    cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
    workers = []
    for shard, gpu_id in zip(shards, gpu_ids):
        p = ctx.Process(
            target=prediction_worker,
            args=(gpu_id, model_path, input_raster, shard, result_q, cfg_dict),
        )
        p.start()
        workers.append(p)

    total_tiles = len(all_jobs)
    done = 0
    finished = 0
    start = time.monotonic()
    last_report = start
    total_read_s = 0.0
    total_infer_s = 0.0
    total_write_s = 0.0
    progress_interval_s = float(getattr(config, 'progress_interval_s', 20.0))

    with rasterio.open(tmp_path, 'w', **out_profile) as dst:
        while finished < len(workers):
            result = result_q.get()
            if result.get('done'):
                finished += 1
                continue
            arr = result['array']
            row_off = int(result['row_off'])
            col_off = int(result['col_off'])
            write_h = min(arr.shape[0], layout['out_height'] - row_off)
            write_w = min(arr.shape[1], layout['out_width'] - col_off)
            t0 = time.perf_counter()
            dst.write(
                np.ascontiguousarray(arr[:write_h, :write_w],
                                     dtype=np.float32), 1,
               window=Window(col_off, row_off, write_w, write_h))
            total_write_s += time.perf_counter() - t0
            total_read_s += float(result.get('read_s', 0.0))
            total_infer_s += float(result.get('infer_s', 0.0))
            done += 1
            now = time.monotonic()
            if(done == 1 or done == total_tiles
               or (now - last_report) >= progress_interval_s
            ):
                    elapsed = max(now - start, 1e-9)
                    rate = done / elapsed
                    eta_s = (total_tiles - done) / rate if rate > 0 \
                        else float('inf')
                    print(
                        f"Multi-GPU prediction {done}/{total_tiles} | "
                        f"{done / total_tiles:.1%} | {rate * 60:.1f} tiles/min"
                        f" | ETA {_format_eta(eta_s)} | avg read "
                        f"{total_read_s / max(done, 1):.3f}s infer "
                        f"{total_infer_s / max(done, 1):.3f}s write "
                        f"{total_write_s / max(done, 1):.3f}s",
                        flush=True,
                    )
                    last_report = now

    for p in workers:
        p.join()
    IO.finalize_raster(tmp_path, output_raster)
    return out_profile
