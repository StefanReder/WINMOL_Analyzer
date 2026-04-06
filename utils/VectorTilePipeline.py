from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import time

import numpy as np

from classes.Config import Config
from utils.IO import append_layers_to_gpkg, load_raster_window_with_profile, \
    process_tile_gpkg, write_all_layers_to_gpkg, write_stems_to_gpkg
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


def _config_from_dict(config_dict: dict) -> Config:
    cfg = Config()
    for key, value in config_dict.items():
        try:
            setattr(cfg, key, value)
        except Exception:
            pass
    return cfg


def export_tile_results(stems, profile, process_type: str, output_prefix: str):
    if process_type == 'Trees':
        return write_stems_to_gpkg(stems, profile, output_prefix)
    return write_all_layers_to_gpkg(stems, profile, output_prefix)


def process_prediction_tile(
    pred_raster_path: str,
    tile_job,
    config_dict: dict,
    process_type: str,
    output_prefix: str,
):
    config = _config_from_dict(config_dict)
    pred, profile = load_raster_window_with_profile(
        pred_raster_path, tile_job.halo_window)
    print(f"Processing prediction tile: {tile_job.tile_id}", flush=True)
<<<<<<< Updated upstream
    segments = Skel.find_segments(pred, config, profile)
=======

    pred_arr = np.asarray(pred)
    if pred_arr.size == 0 or not np.any(pred_arr >= 0.5):
        print(f"Skipping empty prediction tile: {tile_job.tile_id}", flush=True)
        return tile_job, None

    segments = Skel.find_segments(pred_arr, config, profile)
    if not segments:
        print(f"No segments found in tile: {tile_job.tile_id}", flush=True)
        return tile_job, None

>>>>>>> Stashed changes
    segments = Vec.restore_geoinformation(segments, config, profile)
    stems = Vec.build_stem_parts(segments)
    stems = Vec.connect_stems(stems, config)
    if not stems:
        print(f"No stems found in tile: {tile_job.tile_id}", flush=True)
        return tile_job, None

    Vec.rebuild_endnodes_from_stems(stems)
<<<<<<< Updated upstream
    stems = Quant.quantify_stems(stems, pred, profile, config=config)
=======
    stems = Quant.quantify_stems(stems, pred_arr, profile, config=config)
    if not stems:
        print(f"No quantified stems in tile: {tile_job.tile_id}", flush=True)
        return tile_job, None

>>>>>>> Stashed changes
    gpkg_path = export_tile_results(stems, profile, process_type, output_prefix)
    return tile_job, gpkg_path


class TileVectorExecutor:
    def __init__(
        self,
        pred_raster_path: str,
        raster_profile: dict,
        config,
        process_type: str,
        output_dir: str,
        output_gpkg: str,
        tile_workers: int,
        keep_temp: bool = False,
        merge_batch_tiles: int = 8,
    ):
        self.pred_raster_path = pred_raster_path
        self.raster_profile = raster_profile
        self.process_type = process_type
        self.output_dir = output_dir
        self.output_gpkg = output_gpkg
        self.keep_temp = bool(keep_temp)
        self.progress_interval_s = float(getattr(config, 'progress_interval_s', 60.0))
        self.merge_batch_tiles = max(1, int(merge_batch_tiles))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_gpkg) or '.', exist_ok=True)
        if os.path.exists(self.output_gpkg):
            os.remove(self.output_gpkg)

        base_cfg = _clone_config(config, cpu_workers=1, vector_tile_workers=1)
        self.config_dict = dict(getattr(base_cfg, 'to_dict', lambda: {})())
        self.tile_workers = max(1, int(tile_workers or getattr(config, 'vector_tile_workers', 1) or 1))
        self.executor = cf.ProcessPoolExecutor(
            max_workers=self.tile_workers,
            mp_context=mp.get_context('spawn'),
        )
        self.pending = {}
        self.start = time.monotonic()
        self.last_report = self.start
        self.submitted = 0
        self.completed = 0
        self.target_crs = raster_profile.get('crs')
        self.total_stems = 0
        self.total_nodes = 0
        self.total_vectors = 0
        self._buf_stems = []
        self._buf_nodes = []
        self._buf_vectors = []
        self._buf_tiles = 0

    def submit(self, tile_job):
        output_prefix = os.path.join(self.output_dir, tile_job.tile_id)
        fut = self.executor.submit(
            process_prediction_tile,
            self.pred_raster_path,
            tile_job,
            self.config_dict,
            self.process_type,
            output_prefix,
        )
        self.pending[fut] = tile_job
        self.submitted += 1

    def _flush_buffers(self):
        if not (self._buf_stems or self._buf_nodes or self._buf_vectors):
            return
        append_layers_to_gpkg(
            layers=[
                ('stems', self._concat(self._buf_stems)),
                ('nodes', self._concat(self._buf_nodes)),
                ('vectors', self._concat(self._buf_vectors)),
            ],
            crs=self.target_crs,
            final_path=self.output_gpkg,
        )
        self._buf_stems = []
        self._buf_nodes = []
        self._buf_vectors = []
        self._buf_tiles = 0

    @staticmethod
    def _concat(frames):
        if not frames:
            return None
        import pandas as pd
        import geopandas as gpd
        return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                geometry='geometry', crs=frames[0].crs)

    def _handle_result(self, fut):
        tile_job = self.pending.pop(fut)
        tile_job, gpkg_path = fut.result()
<<<<<<< Updated upstream
=======

        if gpkg_path is None:
            self.completed += 1
            return

>>>>>>> Stashed changes
        processed, self.target_crs = process_tile_gpkg(
            tile_job, gpkg_path, self.raster_profile, self.target_crs)
        if processed is not None:
            stems, nodes, vectors, counts = processed
            self._buf_stems.append(stems)
            if not nodes.empty:
                self._buf_nodes.append(nodes)
            if not vectors.empty:
                self._buf_vectors.append(vectors)
            self._buf_tiles += 1
            self.total_stems += counts[0]
            self.total_nodes += counts[1]
            self.total_vectors += counts[2]
            if self._buf_tiles >= self.merge_batch_tiles:
                self._flush_buffers()
        if not self.keep_temp:
            try:
                os.remove(gpkg_path)
            except Exception:
                pass
        self.completed += 1

    def poll(self, block: bool = False):
        if not self.pending:
            return 0
        timeout = None if block else 0
        done, _ = cf.wait(
            set(self.pending.keys()),
            timeout=timeout,
            return_when=cf.FIRST_COMPLETED,
        )
        if not done:
            return 0
        for fut in list(done):
            self._handle_result(fut)
        now = time.monotonic()
        if (
            self.completed == self.submitted
            or (now - self.last_report) >= self.progress_interval_s
        ):
            rate = self.completed / max(now - self.start, 1e-9)
            eta = ((self.submitted - self.completed) / rate) if rate > 0 else float('inf')
            print(
                f"Vector tiles {self.completed}/{self.submitted} | "
                f"{rate * 60:.2f} tiles/min | ETA {eta / 60:.1f} min",
                flush=True,
            )
            self.last_report = now
        return len(done)

    def drain(self):
        while self.pending:
            self.poll(block=True)
        self._flush_buffers()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
        self._flush_buffers()
        print('')
        print('TILED VECTOR SUMMARY')
        print(f'Tiles submitted:       {self.submitted}')
        print(f'Tiles completed:       {self.completed}')
        print(f'Total stems written:   {self.total_stems}')
        print(f'Total nodes written:   {self.total_nodes}')
        print(f'Total vectors written: {self.total_vectors}')
        print(f'Output saved to: {self.output_gpkg}')


def process_prediction_tiles(
    pred_raster_path: str,
    tile_jobs,
    raster_profile: dict,
    config,
    process_type: str,
    output_dir: str,
    output_gpkg: str,
    cpu_workers: int,
    keep_temp: bool = False,
):
    total_workers = max(1, int(cpu_workers or getattr(config, 'cpu_workers', 1) or 1))
    requested_tile_workers = max(1, int(getattr(config, 'vector_tile_workers', 1) or 1))
    tile_workers = max(1, min(requested_tile_workers, len(tile_jobs), total_workers))
    executor = TileVectorExecutor(
        pred_raster_path=pred_raster_path,
        raster_profile=raster_profile,
        config=config,
        process_type=process_type,
        output_dir=output_dir,
        output_gpkg=output_gpkg,
        tile_workers=tile_workers,
        keep_temp=keep_temp,
    )
    try:
        for tile_job in tile_jobs:
            executor.submit(tile_job)
            executor.poll(block=False)
        executor.drain()
    finally:
        executor.close()
    return output_gpkg
