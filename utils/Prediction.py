#!/usr/bin/env python

################################################################################
"""Imports"""
import numpy as np
import tensorflow as tf
from rasterio import Affine
from skimage.transform import resize

from classes.Timer import Timer

################################################################################
"""Prediction of the semantic stem map with U-Net"""


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
        fill_value=0, dtype=np.float64
    )
    img_pad[0:img.shape[0], 0:img.shape[1], ] = img

    img_width_ = config.img_width - config.overlap_pred
    prediction = np.empty((img_pad.shape[0], img_pad.shape[1]))
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

    px_per_tile_x = int(
        np.ceil(config.tile_size / abs(profile['transform'][0])))
    px_per_tile_y = int(
        np.ceil(config.tile_size / abs(profile['transform'][4])))
    overlap_img_x = config.overlap_pred * px_per_tile_x / config.img_width
    overlap_img_y = config.overlap_pred * px_per_tile_y / config.img_width
    x_tiles = int(np.ceil(img.shape[1] / (px_per_tile_x - overlap_img_x)))
    y_tiles = int(np.ceil(img.shape[0] / (px_per_tile_y - overlap_img_y)))

    # padding to full tiles
    img_pd = np.full(
        (int(np.ceil(y_tiles * (px_per_tile_y - overlap_img_y) + overlap_img_y)),
         int(np.ceil(x_tiles * (px_per_tile_x - overlap_img_x) + overlap_img_x)),
         config.n_channels), fill_value=0, dtype=np.float64)
    img_pd[0:img.shape[0], 0:img.shape[1], ] = img

    img_width_ = config.img_width - config.overlap_pred
    prediction = np.empty((y_tiles * img_width_ + config.overlap_pred,
                           x_tiles * img_width_ + config.overlap_pred))
    mask = np.where(img_pd[:, :, 0:3] == (0, 0, 0), False, True)[:, :, 0]
    mask = resize(mask, prediction.shape)

    for i in range(y_tiles):
        x = int(np.floor(i * (px_per_tile_y - overlap_img_y))) 
        for j in range(x_tiles):
            y = int(np.floor(j * (px_per_tile_x - overlap_img_x))) 
            tile = img_pd[x:x + px_per_tile_x - 1, y:y + px_per_tile_y - 1, 0:3]
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

    profile['transform'] = Affine(
        profile['transform'][0] * px_per_tile_x / config.img_width, 0.0,
        profile['transform'][2], 0.0,
        profile['transform'][4] * px_per_tile_y / config.img_width,
        profile['transform'][5]
    )
    print(x_tiles * y_tiles, " tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return prediction, profile
