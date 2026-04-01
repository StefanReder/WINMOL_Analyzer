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
    """Normalize uint8 imagery to float32 in [0, 1]."""
    if arr.dtype == np.float32:
        return arr
    return (arr / 255.0).astype(np.float32, copy=False)


def _resampling_layout(shape, profile, config):
    """Compute source/output tile layout for per-tile resampling prediction."""
    height, width = int(shape[0]), int(shape[1])
    px_per_tile_x = int(np.ceil(config.tile_size / abs(profile['transform'][0])))
    px_per_tile_y = int(np.ceil(config.tile_size / abs(profile['transform'][4])))

    overlap_img_x = config.overlap_pred * px_per_tile_x / config.img_width
    overlap_img_y = config.overlap_pred * px_per_tile_y / config.img_width

    step_src_x = max(1, int(np.floor(px_per_tile_x - overlap_img_x)))
    step_src_y = max(1, int(np.floor(px_per_tile_y - overlap_img_y)))

    x_tiles = int(np.ceil(width / (px_per_tile_x - overlap_img_x)))
    y_tiles = int(np.ceil(height / (px_per_tile_y - overlap_img_y)))

    img_width_ = config.img_width - config.overlap_pred
    out_width = int(x_tiles * img_width_ + config.overlap_pred)
    out_height = int(y_tiles * img_width_ + config.overlap_pred)

    out_transform = Affine(
        profile['transform'][0] * px_per_tile_x / config.img_width,
        0.0,
        profile['transform'][2],
        0.0,
        profile['transform'][4] * px_per_tile_y / config.img_width,
        profile['transform'][5],
    )

    return {
        'px_per_tile_x': px_per_tile_x,
        'px_per_tile_y': px_per_tile_y,
        'overlap_img_x': overlap_img_x,
        'overlap_img_y': overlap_img_y,
        'step_src_x': step_src_x,
        'step_src_y': step_src_y,
        'x_tiles': x_tiles,
        'y_tiles': y_tiles,
        'img_width_inner': img_width_,
        'out_width': out_width,
        'out_height': out_height,
        'out_transform': out_transform,
    }



def predict(img, model, config):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of semantic stem map")

    x_tiles = int(
        np.ceil(img.shape[1] / (config.img_width - config.overlap_pred)))
    y_tiles = int(
        np.ceil(img.shape[0] / (config.img_width - config.overlap_pred)))

    # padding to full tiles
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

    # padding to full tiles
    sy = int(np.ceil(
        layout['y_tiles'] * (
            layout['px_per_tile_y'] - layout['overlap_img_y']
        ) + layout['overlap_img_y']
    ))
    sx = int(np.ceil(
        layout['x_tiles'] * (
            layout['px_per_tile_x'] - layout['overlap_img_x']
        ) + layout['overlap_img_x']
    ))
    img_pd = np.full(
        (sy, sx, config.n_channels),
        fill_value=0,
        dtype=np.float32,
    )
    img_pd[0:img.shape[0], 0:img.shape[1], ] = img

    img_width_ = layout['img_width_inner']
    prediction = np.zeros(
        (layout['out_height'], layout['out_width']),
        dtype=np.float32,
    )
    mask = np.where(img_pd[:, :, 0:3] == (0, 0, 0), False, True)[:, :, 0]
    mask = resize(mask, prediction.shape).astype(np.float32)

    for i in range(layout['y_tiles']):
        x = int(np.floor(i * (layout['px_per_tile_y'] - layout['overlap_img_y'])))
        for j in range(layout['x_tiles']):
            y = int(np.floor(j * (layout['px_per_tile_x'] - layout['overlap_img_x'])))
            tile = img_pd[
                x:x + layout['px_per_tile_x'] - 1,
                y:y + layout['px_per_tile_y'] - 1,
                0:3,
            ]
            tile = tf.convert_to_tensor(tile, dtype=np.float32)
            tile = tf.image.resize(tile,
                                   size=[config.img_width, config.img_height],
                                   method="bicubic", antialias=False)
            tile = tf.reshape(
                tile,
                shape=[1, config.img_width, config.img_width, 3]
            )

            pred = model.predict_on_batch(tile)
            pred2 = pred[
                0,
                (config.overlap_pred // 2):
                    (config.img_width - config.overlap_pred // 2),
                (config.overlap_pred // 2):
                    (config.img_width - config.overlap_pred // 2),
                0,
            ]
            prediction[
                (config.overlap_pred // 2 + i * img_width_): (
                    (config.img_width - config.overlap_pred // 2) + i
                    * img_width_
                ),
                (config.overlap_pred // 2 + j * img_width_): (
                    (config.img_width - config.overlap_pred // 2) + j
                    * img_width_
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



def predict_with_resampling_stream_to_raster(uav_path, output_stem_path, model,
                                             config):
    """Predict tile-by-tile from disk and write output directly to raster."""
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of the semantic stem map")
    print("Resampling tiles while analyzing (stream mode)")

    os.makedirs(os.path.dirname(output_stem_path) or ".", exist_ok=True)

    with rasterio.open(uav_path) as src:
        src_profile = src.profile.copy()
        layout = _resampling_layout((src.height, src.width), src_profile, config)

        out_profile = src_profile.copy()
        out_profile.update(
            dtype=rasterio.float32,
            count=1,
            width=layout['out_width'],
            height=layout['out_height'],
            transform=layout['out_transform'],
            tiled=False,
        )

        crop = config.overlap_pred // 2
        tiles_done = 0

        with rasterio.open(output_stem_path, 'w', **out_profile) as dst:
            for i in range(layout['y_tiles']):
                src_row = int(np.floor(
                    i * (layout['px_per_tile_y'] - layout['overlap_img_y'])
                ))
                for j in range(layout['x_tiles']):
                    src_col = int(np.floor(
                        j * (layout['px_per_tile_x'] - layout['overlap_img_x'])
                    ))

                    window = Window(
                        col_off=src_col,
                        row_off=src_row,
                        width=max(1, layout['px_per_tile_x'] - 1),
                        height=max(1, layout['px_per_tile_y'] - 1),
                    )

                    tile = src.read(
                        list(range(1, config.n_channels + 1)),
                        window=window,
                        boundless=True,
                        fill_value=0,
                    )
                    tile = tile[0:3, :, :].transpose(1, 2, 0)
                    tile = _to_float32_image(tile)

                    mask = np.any(tile[:, :, 0:3] != 0.0, axis=2).astype(np.float32)

                    tile_tensor = tf.convert_to_tensor(tile, dtype=np.float32)
                    tile_tensor = tf.image.resize(
                        tile_tensor,
                        size=[config.img_width, config.img_height],
                        method="bicubic",
                        antialias=False,
                    )
                    tile_tensor = tf.reshape(
                        tile_tensor,
                        shape=[1, config.img_width, config.img_width, 3],
                    )

                    pred = model.predict_on_batch(tile_tensor)
                    pred_core = pred[
                        0,
                        crop:(config.img_width - crop),
                        crop:(config.img_width - crop),
                        0,
                    ].astype(np.float32, copy=False)

                    mask_tensor = tf.convert_to_tensor(mask[..., np.newaxis],
                                                       dtype=np.float32)
                    mask_tensor = tf.image.resize(
                        mask_tensor,
                        size=[config.img_width, config.img_height],
                        method="nearest",
                        antialias=False,
                    )
                    mask_core = mask_tensor[
                        crop:(config.img_width - crop),
                        crop:(config.img_width - crop),
                        0,
                    ].numpy().astype(np.float32, copy=False)

                    pred_core *= mask_core

                    row_out = crop + i * layout['img_width_inner']
                    col_out = crop + j * layout['img_width_inner']
                    out_window = Window(
                        col_off=col_out,
                        row_off=row_out,
                        width=layout['img_width_inner'],
                        height=layout['img_width_inner'],
                    )
                    dst.write(pred_core, 1, window=out_window)
                    tiles_done += 1

        print(tiles_done, " tiles analyzed")

    t.stop()
    print("#######################################################")
    print("")
    return out_profile
