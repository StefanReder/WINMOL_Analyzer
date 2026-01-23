#!/usr/bin/env python

###############################################################################
"""Imports"""

import json
import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
from matplotlib import pyplot as plt
from rasterio.enums import Resampling

###############################################################################


"""File operations"""


def load_model_from_path(model_path):
    # Function to open the model with a fallback mechanism
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


def merge_and_filter_tiled_results(work_dir):
    
    # Output GeoPackage
    output_gpkg  = os.path.join(work_dir, f"{prefix}_merged_data.gpkg")

    # Initialize empty GeoDataFrames
    merged_stems = gpd.GeoDataFrame()
    merged_nodes = gpd.GeoDataFrame()
    merged_vectors = gpd.GeoDataFrame()

    # Counters
    tile_count = 0
    total_stems = 0
    total_nodes = 0
    total_vectors = 0

    # List all raster files that contain _roi_stem_map and end with .tif or .tiff
    raster_files = [f for f in os.listdir(work_dir)
                    if f.endswith((".tif", ".tiff")) and "_roi_stem_map" in f]

    for raster_file in raster_files:
        # Strip suffix to get prefix (e.g. raster_100)
        prefix = raster_file
        for ext in ["_roi_stem_map.tif", "_roi_stem_map.tiff"]:
            if raster_file.endswith(ext):
                prefix = raster_file.replace(ext, "")
                break

        # Extract raster number (tile ID)
        match = re.match(r"raster_(.+)", prefix)
        if not match:
            print(
                f"Could not extract tile number from '{prefix}', skipping...")
            continue
        raster_number = match.group(1)

        raster_path = os.path.join(work_dir, raster_file)
        print(f"\nProcessing tile: {prefix}")

        # Load raster extent
        try:
            with rasterio.open(raster_path) as src:
                bounds = src.bounds
                raster_crs = src.crs
                extent_geom = box(*bounds)
        except Exception as e:
            print(f"Error reading raster {raster_file}: {e}")
            continue

        neg_buffer_geom = extent_geom.buffer(-1)

        # Load GeoJSON vector files
        stems_path   = os.path.join(work_dir, f"{prefix}_roi_stems.geojson")
        nodes_path   = os.path.join(work_dir, f"{prefix}_roi_nodes.geojson")
        vectors_path = os.path.join(work_dir, f"{prefix}_roi_vectors.geojson")

        if not (os.path.exists(stems_path) and os.path.exists(nodes_path) \
                and os.path.exists(vectors_path)):
            print(f"Missing one or more vector files for {prefix}, skipping...")
            continue

        stems = gpd.read_file(stems_path)
        nodes = gpd.read_file(nodes_path)
        vectors = gpd.read_file(vectors_path)

        for gdf in [stems, nodes, vectors]:
            if gdf.crs != raster_crs:
                gdf.to_crs(raster_crs, inplace=True)

        if 'id' not in stems.columns:
            print(f"Missing 'id' column in stems for {prefix}, skipping...")
            continue

        # Generate globally unique stem_id
        stems['stem_id'] = stems['id'].apply(lambda x: f"{raster_number}_{x}")

        # Filter out false detections origin from edge effects
        stems_filtered = stems[stems.intersects(neg_buffer_geom)]

        if stems_filtered.empty:
            print(f"No valid stems after filtering for {prefix}, skipping...")
            continue

        tile_count += 1
        n_stems = len(stems_filtered)
        total_stems += n_stems
        merged_stems = pd.concat([merged_stems, stems_filtered], \
            ignore_index=True)

        # Link nodes and vectors by stem_id
        if 'stem_id' in nodes.columns and 'stem_id' in vectors.columns:
            selected_ids = stems_filtered['id'].unique()
            selected_nodes = nodes[nodes['stem_id'].isin(selected_ids)]
            selected_vectors = vectors[vectors['stem_id'].isin(selected_ids)]

            n_nodes = len(selected_nodes)
            n_vectors = len(selected_vectors)
            total_nodes += n_nodes
            total_vectors += n_vectors

            merged_nodes = pd.concat([merged_nodes, selected_nodes], \
                ignore_index=True)
            merged_vectors = pd.concat([merged_vectors, selected_vectors], \
                ignore_index=True)

            print(
                f"Added {n_stems} stems, {n_nodes} nodes, {n_vectors} vectors.")
        else:
            print(
                f"'stem_id' missing in nodes or vectors for {prefix}, skipping related data.")

    # Save output to GeoPackage
    if not merged_stems.empty:
        merged_stems.to_file(output_gpkg, layer="stems", driver="GPKG")
    if not merged_nodes.empty:
        merged_nodes.to_file(output_gpkg, layer="nodes", driver="GPKG")
    if not merged_vectors.empty:
        merged_vectors.to_file(output_gpkg, layer="vectors", driver="GPKG")

    # Report summary
    print("\nMERGE SUMMARY")
    print(f"Tiles processed:     {tile_count}")
    print(f"Total stems added:   {total_stems}")
    print(f"Total nodes linked:  {total_nodes}")
    print(f"Total vectors linked:{total_vectors}")
    print(f"Output saved to: {output_gpkg}")
