from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class RasterInfo:
    width: int
    height: int
    bands: int
    dtype: str
    pixel_size_x: float
    pixel_size_y: float
    estimated_input_gb: float

    @classmethod
    def from_any(cls, value: Any) -> "RasterInfo":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                width=int(value.get('width', 0)),
                height=int(value.get('height', 0)),
                bands=int(value.get('bands', 0)),
                dtype=str(value.get('dtype', 'unknown')),
                pixel_size_x=float(value.get('pixel_size_x', 0.0) or 0.0),
                pixel_size_y=float(value.get('pixel_size_y', 0.0) or 0.0),
                estimated_input_gb=float(value.get('estimated_input_gb', 0.0) or 0.0),
            )
        return cls(
            width=int(getattr(value, 'width', 0)),
            height=int(getattr(value, 'height', 0)),
            bands=int(getattr(value, 'bands', 0)),
            dtype=str(getattr(value, 'dtype', 'unknown')),
            pixel_size_x=float(getattr(value, 'pixel_size_x', 0.0) or 0.0),
            pixel_size_y=float(getattr(value, 'pixel_size_y', 0.0) or 0.0),
            estimated_input_gb=float(getattr(value, 'estimated_input_gb', 0.0) or 0.0),
        )


@dataclass
class ExecutionPlan:
    process_type: str
    prediction_mode: str
    vector_mode: str
    tile_inner_px: int
    tile_overlap_m: float
    halo_px: int
    gpu_workers: int
    cpu_workers: int
    keep_temp: bool


def _cfg(config: Any, key: str, default: Any) -> Any:
    return getattr(config, key, default)


def _meters_to_pixels(tile_overlap_m: float, pixel_size_x: float, pixel_size_y: float) -> int:
    px = max(abs(pixel_size_x) or 0.0, abs(pixel_size_y) or 0.0)
    if px <= 0.0:
        return 0
    return int(math.ceil(tile_overlap_m / px))


def build_execution_plan(config: Any, hardware: Any, raster_info: Any, process_type: str) -> ExecutionPlan:
    raster = RasterInfo.from_any(raster_info)
    tile_inner_px = int(_cfg(config, 'tile_inner_px', 4096))
    tile_overlap_m = float(_cfg(config, 'tile_overlap_m', 12.0))
    halo_px = _meters_to_pixels(tile_overlap_m, raster.pixel_size_x, raster.pixel_size_y)
    keep_temp = bool(_cfg(config, 'keep_temp_tiles', False))
    threshold_gb = float(_cfg(config, 'legacy_full_array_threshold_gb', 8.0))
    max_gpu_workers = int(_cfg(config, 'max_gpu_workers', 8))
    max_cpu_workers = int(_cfg(config, 'max_cpu_workers', max(getattr(hardware, 'cpu_count', 1) - 1, 1)))

    available_gpus = int(getattr(hardware, 'gpu_count', 0))
    gpu_workers = min(max_gpu_workers, available_gpus) if available_gpus > 0 else 0
    cpu_workers = min(max_cpu_workers, max(getattr(hardware, 'cpu_count', 1) - 1, 1))

    exec_mode = str(_cfg(config, 'execution_mode', 'auto')).lower()
    pred_pref = str(_cfg(config, 'prediction_backend', 'auto')).lower()
    vec_pref = str(_cfg(config, 'vector_backend', 'auto')).lower()

    large_raster = raster.estimated_input_gb >= threshold_gb
    medium_raster = raster.estimated_input_gb >= (threshold_gb * 0.5)

    if exec_mode == 'legacy_full':
        prediction_mode = 'full'
    elif exec_mode == 'stream':
        prediction_mode = 'stream' if gpu_workers else 'cpu_stream'
    elif exec_mode == 'tiled':
        prediction_mode = 'tiled_multi_gpu' if gpu_workers > 1 else ('stream' if gpu_workers == 1 else 'cpu_stream')
    else:
        if process_type == 'Stems':
            if large_raster:
                prediction_mode = 'tiled_multi_gpu' if gpu_workers > 1 else ('stream' if gpu_workers == 1 else 'cpu_stream')
            else:
                prediction_mode = 'full'
        else:
            if large_raster or medium_raster:
                prediction_mode = 'tiled_multi_gpu' if gpu_workers > 1 else ('stream' if gpu_workers == 1 else 'cpu_stream')
            else:
                prediction_mode = 'full'

    if pred_pref == 'cpu':
        prediction_mode = 'cpu_stream'
    elif pred_pref == 'single_gpu':
        prediction_mode = 'stream' if gpu_workers else 'cpu_stream'
    elif pred_pref == 'multi_gpu':
        prediction_mode = 'tiled_multi_gpu' if gpu_workers > 1 else ('stream' if gpu_workers else 'cpu_stream')

    if vec_pref == 'global':
        vector_mode = 'global'
    elif vec_pref == 'tiled':
        vector_mode = 'tiled'
    else:
        vector_mode = 'tiled' if process_type in {'Trees', 'Nodes'} and (large_raster or medium_raster) else 'global'

    return ExecutionPlan(
        process_type=process_type,
        prediction_mode=prediction_mode,
        vector_mode=vector_mode,
        tile_inner_px=tile_inner_px,
        tile_overlap_m=tile_overlap_m,
        halo_px=halo_px,
        gpu_workers=gpu_workers,
        cpu_workers=cpu_workers,
        keep_temp=keep_temp,
    )
