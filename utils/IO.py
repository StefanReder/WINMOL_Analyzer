#!/usr/bin/env python

###############################################################################
"""Imports"""

import json
import os
import numpy as np
import rasterio
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

###############################################################################


"""File operations"""
# Function to open the model with a fallback mechanism
def load_model_from_path(model_path):
    def custom_dropout(**kwargs):
        if 'seed' in kwargs and isinstance(kwargs['seed'], float):
            kwargs['seed'] = int(kwargs['seed'])  # Convert seed to int
        return layers.Dropout(**kwargs)

    class CustomConv2DTranspose(layers.Conv2DTranspose):
        # Remove 'groups' parameter if present
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

        def call(self, inputs, **kwargs):
            return super().call(inputs, **kwargs)

    try:
        print("Trying to load model using open_model()")
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print("open_model() failed:", e)

    try:
        print("Retrying with custom layers (Dropout, Conv2DTranspose)")
        get_custom_objects()["Dropout"] = custom_dropout
        get_custom_objects()["Conv2DTranspose"] = CustomConv2DTranspose
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print("Loading with custom layers also failed:", e)

    raise RuntimeError("Failed to load model with all methods.")


def load_orthomosaic(path, config):
    with rasterio.open(path) as src:
        img = src.read(list(range(1, config.n_channels + 1))).transpose(1, 2, 0)
        img = (img / 255).astype(np.float32)
        return img, src.profile


def load_orthomosaic_with_resampling(path, config):
    with rasterio.open(path) as src:
        scale_factor_x = src.res[0] / (config.tile_size / config.img_width)
        scale_factor_y = src.res[1] / (config.tile_size / config.img_width)
        # resample data to target shape
        img = src.read(
            list(range(1, config.n_channels + 1)),
            out_shape=(
                config.n_channels,
                int(src.height * scale_factor_y),
                int(src.width * scale_factor_x)
            ),
            resampling=Resampling.bilinear
        )
        img = img[0:3, :, :].transpose(1, 2, 0)
        # scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / img.shape[-2]),
            (src.height / img.shape[-3])
        )
        img = (img / 255).astype(np.float32)
        profile = src.profile.copy()
        profile['transform'] = transform
    return img, profile


def load_stem_map(path):
    if path.endswith('.tif') or path.endswith('.tiff'):
        print("#######################################################")
        print("#######################################################")
        print("")
        print(path)
        print("")
        with rasterio.open(path) as src:
            # crs=src.crs
            pred = src.read()
            pred = pred[0, :, :]
            profile = src.profile
        # px_size=(pred.bounds.right-pred.bounds.left)/pred.width
        # px_size=abs(src.profile['transform'][0])
        # bounds=src.bounds
        # pred=np.pad(pred,((padding,padding),(padding,padding)),
        # 'constant', constant_values=False)
        return pred, profile


def export_stem_map(pred, profile, pred_dir, pred_name):
    profile.update(dtype=rasterio.float32, count=1)
    height, width = pred.shape
    profile['width'] = width
    profile['height'] = height
    with rasterio.open(
            os.path.join(
                pred_dir, f'{pred_name}.tiff'
            ), 'w', **profile
    ) as dst:
        dst.write(pred.astype(rasterio.float32), 1)


def get_bounds_from_profile(profile):
    left = profile['transform'][2]
    right = profile['transform'][2] + profile['transform'][0] * profile['width']
    bot = profile['transform'][5] + profile['transform'][4] * profile['height']
    top = profile['transform'][5]
    return rasterio.coords.BoundingBox(left, bot, right, top)


def stems_to_geojson_(stems, profile):
    # Ensure crs_epsg is a string in the correct format
    crs_epsg = profile['crs']
    if isinstance(crs_epsg, int):
        crs_epsg = f"EPSG:{crs_epsg}"

    return {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': f'urn:ogc:def:crs:{crs_epsg}'
            }
        },
        'features': [
            {
                'type': "Feature",
                'geometry': {
                    'type': 'LineString',
                    'coordinates': stems[i].path.coords[:]
                },
                'properties': {
                    'id': i,
                    'start': stems[i].start.coords[:],
                    'stop': stems[i].stop.coords[:],
                    'path': stems[i].path.coords[:],
                    #                'Vector':stems[i].vector,
                    'd': stems[i].segment_diameter_list,
                    'l': stems[i].segment_length_list,
                    'v': stems[i].segment_volume_list,
                    'Length': stems[i].length,
                    'Volume': stems[i].volume
                },
            }
            for i in range(len(stems))
        ],
    }


def nodes_to_geojson_(stems, profile):
    # Ensure crs_epsg is a string in the correct format
    crs_epsg = profile['crs']
    if isinstance(crs_epsg, int):
        crs_epsg = f"EPSG:{crs_epsg}"

    return {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': f'urn:ogc:def:crs:{crs_epsg}'
            }
        },
        'features': [
            {
                'type': "Feature",
                'geometry': {
                    'type': 'Point',
                    'coordinates': stems[i].path.coords[j]
                },
                'properties': {
                    'stem_id': i,
                    'node': j,
                    'Vector': stems[i].vector[j].coords[:],
                    'd': stems[i].segment_diameter_list[j],
                },
            } for i in range(len(stems))
            for j in range(len(stems[i].path.coords))
        ],
    }


def vectors_to_geojson_(stems, profile):
    # Ensure crs_epsg is a string in the correct format
    crs_epsg = profile['crs']
    if isinstance(crs_epsg, int):
        crs_epsg = f"EPSG:{crs_epsg}"

    return {
        'type': 'FeatureCollection',
        'crs': {
            'type': 'name',
            'properties': {
                'name': f'urn:ogc:def:crs:{crs_epsg}'
            }
        },
        'features': [
            {
                'type': "Feature",
                'geometry': {
                    'type': 'LineString',
                    'coordinates': stems[i].vector[j].coords[:]
                },
                'properties': {
                    'stem_id': i,
                    'node': j,
                    'Vector': stems[i].vector[j].coords[:],
                    'd': stems[i].segment_diameter_list[j],
                },
            } for i in range(len(stems))
            for j in range(len(stems[i].path.coords))
        ],
    }


def stems_to_geojson(stems, profile, path):
    # second checkbox
    print("Export Stems to GeoJSON")

    fc_stems = stems_to_geojson_(stems, profile)
    s_path = path + "_stems.geojson"

    with open(s_path, 'w') as out:
        json.dump(fc_stems, out)

    print(f'\nWrote {s_path}')


def vector_to_geojson(stems, profile, path):
    # third checkbox
    print("Export Vectors to GeoJSON")

    fc_vectors = vectors_to_geojson_(stems, profile)

    v_path = path + "_vectors.geojson"

    with open(v_path, 'w') as out:
        json.dump(fc_vectors, out)
    print("")


def nodes_to_geojson(stems, profile, path):
    # third checkbox
    print("#######################################################")
    print("Export Nodes to GeoJSON")

    fc_nodes = nodes_to_geojson_(stems, profile)

    n_path = path + "_nodes.geojson"

    with open(n_path, 'w') as out:
        json.dump(fc_nodes, out)

    print('#######################################################')
    print("")


def save_image(data, output_name, size=(15, 15), dpi=300):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(output_name, dpi=dpi)
