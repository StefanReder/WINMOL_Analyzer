#!/usr/bin/env python

################################################################################
"""Imports"""
import os

import numpy as np
import rasterio
import tensorflow as tf
from rasterio import Affine
from rasterio.windows import Window
from skimage.transform import resize

from classes.Timer import Timer

################################################################################
"""Prediction of the semantic stem map with U-Net"""


def _to_float32_image(arr):
    if arr.dtype == np.float32:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        return (arr / 255.0).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _resampling_layout(shape, profile, config):
    height, width = int(shape[0]), int(shape[1])
    px_per_tile_x = int(np.ceil(config.tile_size / abs(profile['transform'][0])))
    px_per_tile_y = int(np.ceil(config.tile_size / abs(profile['transform'][4])))
    overlap_img_x = config.overlap_pred * px_per_tile_x / config.img_width
    overlap_img_y = config.overlap_pred * px_per_tile_y / config.img_width
    x_tiles = int(np.ceil(width / max(px_per_tile_x - overlap_img_x, 1)))
    y_tiles = int(np.ceil(height / max(px_per_tile_y - overlap_img_y, 1)))
    img_width_inner = config.img_width - config.overlap_pred
    out_width = int(x_tiles * img_width_inner + config.overlap_pred)
    out_height = int(y_tiles * img_width_inner + config.overlap_pred)
    out_transform = Affine(
        profile['transform'][0] * px_per_tile_x / config.img_width, 0.0,
        profile['transform'][2], 0.0,
        profile['transform'][4] * px_per_tile_y / config.img_width,
        profile['transform'][5]
    )
    return {
        'px_per_tile_x': px_per_tile_x,
        'px_per_tile_y': px_per_tile_y,
        'overlap_img_x': overlap_img_x,
        'overlap_img_y': overlap_img_y,
        'x_tiles': x_tiles,
        'y_tiles': y_tiles,
        'img_width_inner': img_width_inner,
        'out_width': out_width,
        'out_height': out_height,
        'out_transform': out_transform,
    }


def predict_tile_array(tile_img, model, config):
    tile_img = _to_float32_image(tile_img)
    if tile_img.ndim == 2:
        tile_img = tile_img[:, :, None]
    tile_img = tile_img[:, :, :3]
    mask = np.any(tile_img != 0, axis=2).astype(np.float32)
    tile = tf.convert_to_tensor(tile_img, dtype=tf.float32)
    tile = tf.image.resize(
        tile,
        size=[config.img_width, config.img_height],
        method='bicubic',
        antialias=False,
    )
    tile = tf.reshape(tile, shape=[1, config.img_width, config.img_width, 3])
    pred = model.predict_on_batch(tile)
    crop = config.overlap_pred // 2
    pred2 = pred[0, crop:(config.img_width - crop), crop:(config.img_width - crop), 0]
    mask_resized = resize(mask, (config.img_width, config.img_width), order=0, preserve_range=True, anti_aliasing=False)
    mask_core = mask_resized[crop:(config.img_width - crop), crop:(config.img_width - crop)]
    return (pred2.astype(np.float32) * (mask_core > 0.5).astype(np.float32)).astype(np.float32)


def predict_stream_to_raster(
    uav_path: str,
    output_stem_map: str,
    model,
    config,
    tile_jobs=None,
    output_compress: str | None = None,
):
    from utils import IO

    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of the semantic stem map")
    print("Resampling tiles while analyzing (stream mode)")

    os.makedirs(os.path.dirname(output_stem_map) or '.', exist_ok=True)
    compress = output_compress
    if compress is None:
        compress = 'DEFLATE' if getattr(config, 'compress_output', True) else None

    with rasterio.open(uav_path) as src:
        src_profile = src.profile.copy()
        layout = _resampling_layout((src.height, src.width), src_profile, config)
        out_profile = IO.build_safe_prediction_profile(
            src_profile=src_profile,
            width=layout['out_width'],
            height=layout['out_height'],
            transform=layout['out_transform'],
            compress=compress,
        )
        tmp_path = IO.atomic_tmp_path(output_stem_map)
        crop = config.overlap_pred // 2
        total_tiles = layout['x_tiles'] * layout['y_tiles']
        tiles_done = 0

        with rasterio.open(tmp_path, 'w', **out_profile) as dst:
            for i in range(layout['y_tiles']):
                x = int(np.floor(i * (layout['px_per_tile_y'] - layout['overlap_img_y'])))
                for j in range(layout['x_tiles']):
                    y = int(np.floor(j * (layout['px_per_tile_x'] - layout['overlap_img_x'])))
                    window = Window(
                        col_off=y,
                        row_off=x,
                        width=max(1, layout['px_per_tile_x'] - 1),
                        height=max(1, layout['px_per_tile_y'] - 1),
                    )
                    tile = src.read(
                        list(range(1, min(config.n_channels, src.count) + 1)),
                        window=window,
                        boundless=True,
                        fill_value=0,
                    ).transpose(1, 2, 0)
                    pred_core = predict_tile_array(tile, model, config)

                    row_out = crop + i * layout['img_width_inner']
                    col_out = crop + j * layout['img_width_inner']
                    write_h = min(pred_core.shape[0], layout['out_height'] - row_out)
                    write_w = min(pred_core.shape[1], layout['out_width'] - col_out)
                    if write_h <= 0 or write_w <= 0:
                        continue
                    pred_write = np.ascontiguousarray(pred_core[:write_h, :write_w], dtype=np.float32)
                    out_window = Window(
                        col_off=col_out,
                        row_off=row_out,
                        width=write_w,
                        height=write_h,
                    )
                    dst.write(pred_write, 1, window=out_window)
                    tiles_done += 1
                    if tiles_done == 1 or tiles_done % 25 == 0 or tiles_done == total_tiles:
                        print(
                            f"Written tile {tiles_done}/{total_tiles} "
                            f"(src {int(window.width)}x{int(window.height)} -> out {write_w}x{write_h})",
                            flush=True,
                        )

        IO.finalize_raster(tmp_path, output_stem_map)

    print(tiles_done, " tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return out_profile


def predict_stream_single_gpu(
    uav_path: str,
    output_stem_map: str,
    model,
    config,
):
    return predict_stream_to_raster(uav_path, output_stem_map, model, config)


def predict_stream_cpu(
    uav_path: str,
    output_stem_map: str,
    model,
    config,
):
    return predict_stream_to_raster(uav_path, output_stem_map, model, config)


def predict_with_resampling_stream_to_raster(uav_path, output_stem_path, model, config):
    return predict_stream_to_raster(uav_path, output_stem_path, model, config)


"""Legacy"""


def predict(img, model, config):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of semantic stem map")

    x_tiles = int(
        np.ceil(img.shape[1] / (config.img_width - config.overlap_pred)))
    y_tiles = int(
        np.ceil(img.shape[0] / (config.img_width - config.overlap_pred)))

    img_pad = np.full((
        y_tiles * (
            config.img_width - config.overlap_pred
        ) + config.overlap_pred,
        x_tiles * (
            config.img_width - config.overlap_pred
        ) + config.overlap_pred,
        config.n_channels
    ),
        fill_value=0, dtype=np.float32
    )
    img_pad[0:img.shape[0], 0:img.shape[1], ] = img

    img_width_ = config.img_width - config.overlap_pred
    prediction = np.zeros((img_pad.shape[0], img_pad.shape[1]), dtype=np.float32)
    mask = np.where(img[:, :, 0:3] == (0, 0, 0), False, True)[:, :, 0]

    for i in range(y_tiles):
        x = i * (config.img_width - config.overlap_pred)
        for j in range(x_tiles):
            y = j * (config.img_width - config.overlap_pred)
            tile = img_pad[x:x + config.img_width, y:y + config.img_width, 0:3]
            tile = tf.convert_to_tensor(tile, dtype=np.float32)
            tile = tf.reshape(
                tile,
                shape=[1, config.img_width, config.img_width, 3]
            )
            pred = model.predict_on_batch(tile)
            pred2 = pred[
                0,
                (config.overlap_pred // 2):(
                    config.img_width - config.overlap_pred // 2),
                (config.overlap_pred // 2):(
                    config.img_width - config.overlap_pred // 2),
                0
            ]
            prediction[
                (config.overlap_pred // 2 + (i) * img_width_): (
                    (config.img_width - config.overlap_pred // 2) + i
                    * img_width_
                ),
                (config.overlap_pred // 2 + (j) * img_width_): (
                    (config.img_width - config.overlap_pred // 2) + j
                    * img_width_
                )
            ] = pred2

    prediction = prediction[0:img.shape[0], 0:img.shape[1]]
    prediction = prediction * mask
    print(x_tiles * y_tiles, " tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return prediction



def predict_with_resampling_per_tile(img, profile, model, config):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of the semantic stem map")
    print("Resampling tiles while analyzing")

    layout = _resampling_layout(img.shape[:2], profile, config)
    sy = int(np.ceil(layout['y_tiles'] * (layout['px_per_tile_y'] - layout['overlap_img_y']) + layout['overlap_img_y']))
    sx = int(np.ceil(layout['x_tiles'] * (layout['px_per_tile_x'] - layout['overlap_img_x']) + layout['overlap_img_x']))
    img_pd = np.full(
        (sy, sx, config.n_channels),
        fill_value=0,
        dtype=np.float32
    )
    img_pd[0:img.shape[0], 0:img.shape[1], ] = img

    img_width_ = layout['img_width_inner']
    prediction = np.zeros((layout['out_height'], layout['out_width']), dtype=np.float32)
    mask = np.where(img_pd[:, :, 0:3] == (0, 0, 0), False, True)[:, :, 0]
    mask = resize(mask, prediction.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)

    for i in range(layout['y_tiles']):
        x = int(np.floor(i * (layout['px_per_tile_y'] - layout['overlap_img_y'])))
        for j in range(layout['x_tiles']):
            y = int(np.floor(j * (layout['px_per_tile_x'] - layout['overlap_img_x'])))
            tile = img_pd[x:x + layout['px_per_tile_x'] - 1, y:y + layout['px_per_tile_y'] - 1, 0:3]
            pred2 = predict_tile_array(tile, model, config)
            prediction[
                (config.overlap_pred // 2 + i * img_width_): (
                    (config.img_width - config.overlap_pred // 2) + i * img_width_
                ),
                (config.overlap_pred // 2 + j * img_width_): (
                    (config.img_width - config.overlap_pred // 2) + j * img_width_
                ),
            ] = pred2
    prediction = prediction * mask

    profile = profile.copy()
    profile['transform'] = layout['out_transform']
    print(layout['x_tiles'] * layout['y_tiles'], " tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return prediction, profile
