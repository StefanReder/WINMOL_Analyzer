from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
from queue import Empty

import rasterio
from rasterio.windows import Window

from classes.Config import Config
from utils.Prediction import _resampling_layout


def prediction_worker(
    gpu_id: int,
    model_path: str,
    input_raster: str,
    jobs,
    results,
    config_dict: dict,
):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import tensorflow as tf
    from utils.IO import load_model_from_path
    from utils.Prediction import predict_tile_array

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    model = load_model_from_path(model_path)
    cfg = Config()
    for key, value in config_dict.items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass

    with rasterio.open(input_raster) as src:
        while True:
            try:
                job = jobs.get(timeout=1)
            except Empty:
                continue
            if job is None:
                break
            window = Window(job['src_col'], job['src_row'], job['src_width'], job['src_height'])
            tile = src.read(
                list(range(1, min(cfg.n_channels, src.count) + 1)),
                window=window,
                boundless=True,
                fill_value=0,
            ).transpose(1, 2, 0)
            pred_core = predict_tile_array(tile, model, cfg)
            results.put({
                'row_off': job['dst_row'],
                'col_off': job['dst_col'],
                'array': pred_core,
            })
    results.put({'done': True})



def run_multi_gpu_prediction(
    model_path: str,
    input_raster: str,
    output_raster: str,
    tile_jobs,
    gpu_ids: list[int],
    config,
):
    from utils import IO

    ctx = mp.get_context('spawn')
    job_q = ctx.Queue(maxsize=max(2, len(gpu_ids) * 2))
    result_q = ctx.Queue(maxsize=max(2, len(gpu_ids) * 2))

    with rasterio.open(input_raster) as src:
        src_profile = src.profile.copy()
        layout = _resampling_layout((src.height, src.width), src_profile, config)
        out_profile = IO.build_safe_prediction_profile(
            src_profile=src_profile,
            width=layout['out_width'],
            height=layout['out_height'],
            transform=layout['out_transform'],
            compress='DEFLATE' if getattr(config, 'compress_output', True) else None,
        )
        tmp_path = IO.atomic_tmp_path(output_raster)
        with rasterio.open(tmp_path, 'w', **out_profile) as dst:
            for i in range(layout['y_tiles']):
                src_row = int((i * (layout['px_per_tile_y'] - layout['overlap_img_y'])))
                for j in range(layout['x_tiles']):
                    src_col = int((j * (layout['px_per_tile_x'] - layout['overlap_img_x'])))
                    job_q.put({
                        'src_row': src_row,
                        'src_col': src_col,
                        'src_width': max(1, layout['px_per_tile_x'] - 1),
                        'src_height': max(1, layout['px_per_tile_y'] - 1),
                        'dst_row': config.overlap_pred // 2 + i * layout['img_width_inner'],
                        'dst_col': config.overlap_pred // 2 + j * layout['img_width_inner'],
                    })

            workers = []
            cfg_dict = dict(getattr(config, 'to_dict', lambda: {})())
            for gpu_id in gpu_ids:
                p = ctx.Process(
                    target=prediction_worker,
                    args=(gpu_id, model_path, input_raster, job_q, result_q, cfg_dict),
                )
                p.start()
                workers.append(p)
            for _ in workers:
                job_q.put(None)

            finished = 0
            while finished < len(workers):
                result = result_q.get()
                if result.get('done'):
                    finished += 1
                    continue
                arr = np.ascontiguousarray(result['array'], dtype='float32')
                write_h = min(arr.shape[0], layout['out_height'] - int(result['row_off']))
                write_w = min(arr.shape[1], layout['out_width'] - int(result['col_off']))
                if write_h <= 0 or write_w <= 0:
                    continue
                dst.write(
                    arr[:write_h, :write_w],
                    1,
                    window=Window(result['col_off'], result['row_off'], write_w, write_h),
                )

            for p in workers:
                p.join()
        IO.finalize_raster(tmp_path, output_raster)
    return out_profile
