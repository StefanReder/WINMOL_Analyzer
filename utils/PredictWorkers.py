from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from typing import Any, List

import numpy as np
import rasterio
from rasterio.windows import Window

from classes.Config import Config
from utils import IO
from utils.Prediction import _iter_tile_jobs, _prepare_inference_batch, \
    _resampling_layout, _format_eta


def _config_from_dict(config_dict: dict) -> Config:
    cfg = Config()
    for key, value in (config_dict or {}).items():
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
    threshold = float(getattr(config, 'stem_binary_threshold', 0.5))
    cores = []
    for idx in range(pred.shape[0]):
        pred_core = pred[idx, crop:(config.img_width - crop),
                         crop:(config.img_width - crop), 0]
        mask_core = mask_resized[idx, crop:(config.img_width - crop),
                                 crop:(config.img_width - crop), 0] > 0.5
        cores.append(np.ascontiguousarray(
            ((pred_core >= threshold) & mask_core).astype(np.uint8)))
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


def _batch_predict_worker(gpu_id: int, model_path: str, input_q, output_q, config_dict: dict):
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

    while True:
        task = input_q.get()
        if task is None:
            break
        call_id = int(task['call_id'])
        start_idx = int(task['start_idx'])
        batch = np.asarray(task['batch'])
        if batch.size == 0:
            output_q.put({
                'call_id': call_id,
                'start_idx': start_idx,
                'pred': np.empty((0,), dtype=np.float32),
            })
            continue
        try:
            pred = model.predict_on_batch(batch)
            output_q.put({
                'call_id': call_id,
                'start_idx': start_idx,
                'pred': np.asarray(pred),
            })
        except Exception as exc:
            output_q.put({
                'call_id': call_id,
                'start_idx': start_idx,
                'error': f'{type(exc).__name__}: {exc}',
            })


class MultiGPUBatchPredictor:
    def __init__(self, model_path: str, gpu_ids: list[int], config):
        if not gpu_ids:
            raise ValueError('MultiGPUBatchPredictor requires at least one GPU id')
        self.model_path = model_path
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.config = _config_from_dict(dict(getattr(config, 'to_dict', lambda: {})()))
        self.ctx = mp.get_context('spawn')
        queue_depth = max(8, len(self.gpu_ids) * 4)
        self.input_q = self.ctx.Queue(maxsize=queue_depth)
        self.output_q = self.ctx.Queue(maxsize=queue_depth)
        self.workers = []
        self._call_id = 0
        for gpu_id in self.gpu_ids:
            proc = self.ctx.Process(
                target=_batch_predict_worker,
                args=(gpu_id, model_path, self.input_q, self.output_q, self.config.to_dict()),
            )
            proc.start()
            self.workers.append(proc)

    def predict_on_batch(self, tile_tensor: Any):
        batch = np.asarray(tile_tensor)
        if batch.ndim == 3:
            batch = batch[None, ...]
        n = int(batch.shape[0]) if batch.ndim >= 1 else 0
        if n <= 0:
            return np.empty((0,), dtype=np.float32)
        num_parts = min(len(self.gpu_ids), n)
        self._call_id += 1
        call_id = self._call_id
        starts = [int(round(i * n / num_parts)) for i in range(num_parts + 1)]
        pending = 0
        for i in range(num_parts):
            s, e = starts[i], starts[i + 1]
            if s >= e:
                continue
            self.input_q.put({
                'call_id': call_id,
                'start_idx': s,
                'batch': np.ascontiguousarray(batch[s:e]),
            })
            pending += 1
        pieces = []
        while len(pieces) < pending:
            result = self.output_q.get()
            if int(result.get('call_id', -1)) != call_id:
                continue
            if 'error' in result:
                raise RuntimeError(result['error'])
            pieces.append((int(result['start_idx']), np.asarray(result['pred'])))
        pieces.sort(key=lambda x: x[0])
        preds = [pred for _, pred in pieces if pred.size > 0]
        if not preds:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(preds, axis=0)

    def close(self):
        for _ in self.workers:
            try:
                self.input_q.put(None)
            except Exception:
                pass
        for proc in self.workers:
            proc.join(timeout=10.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        try:
            self.input_q.close()
        except Exception:
            pass
        try:
            self.output_q.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def build_batch_predictor(model_path: str, gpu_ids: list[int], config):
    return MultiGPUBatchPredictor(model_path, gpu_ids, config)


def run_multi_gpu_prediction(
    model_path: str,
    input_raster: str,
    output_raster: str,
    tile_jobs,
    gpu_ids: list[int],
    config,
):
    # Compatibility wrapper: preserve the public function name, but execute
    # a single outer prediction loop and shard only each inference batch.
    from utils import Prediction as Pred

    if not gpu_ids:
        raise ValueError('multi_gpu_prediction requires at least one GPU id')

    with build_batch_predictor(model_path, gpu_ids, config) as predictor:
        return Pred.predict_stream_to_raster(
            input_raster,
            output_raster,
            predictor,
            config,
        )
