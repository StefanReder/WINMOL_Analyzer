#!/usr/bin/env python

###############################################################################
"""Imports"""

import json
import os
import glob
import re
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point, box
from pyproj import CRS

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


def _crs_from_profile(profile):
    """Return a CRS object that GeoPandas can write reliably."""
    crs = profile.get("crs") if isinstance(profile, dict) else None
    if crs is None:
        return None
    try:
        # best: pyproj CRS
        return CRS.from_user_input(crs)
    except Exception:
        # fallback: WKT if available, else string
        try:
            return crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
        except Exception:
            return None


def _jsonify_list(x):
    # GeoPackage attributes must be scalar; store lists as JSON text
    try:
        return json.dumps([float(v) for v in x], ensure_ascii=False)
    except Exception:
        return json.dumps(x, ensure_ascii=False)


def stems_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []
    for i, s in enumerate(stems):
        # start/stop are Points -> coords like [(x,y)]
        try:
            sx, sy = list(s.start.coords)[0]
        except Exception:
            sx, sy = (None, None)
        try:
            ex, ey = list(s.stop.coords)[0]
        except Exception:
            ex, ey = (None, None)

        # geometry: LineString
        if hasattr(s.path, "geom_type"):
            geom = s.path
        else:
            geom = LineString(list(s.path.coords))
        geoms.append(geom)

        rows.append({
            "stem_id": i,
            "start_x": sx, "start_y": sy,
            "stop_x": ex, "stop_y": ey,
            "length": float(getattr(s, "length", 0.0)),
            "volume": float(getattr(s, "volume", 0.0)),
            "d_json": _jsonify_list(getattr(s, "segment_diameter_list", [])),
            "l_json": _jsonify_list(getattr(s, "segment_length_list", [])),
            "v_json": _jsonify_list(getattr(s, "segment_volume_list", [])),
        })

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def nodes_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []
    for i, s in enumerate(stems):
        coords = list(s.path.coords)
        for j, xy in enumerate(coords):
            geoms.append(Point(xy))
            # diameter at node j if available
            d = None
            try:
                d = float(s.segment_diameter_list[j])
            except Exception:
                pass
            rows.append({
                "stem_id": i,
                "node": j,
                "d": d,
            })

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def vectors_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []
    for i, s in enumerate(stems):
        for j in range(len(s.path.coords)):
            try:
                seg = s.vector[j]
                if hasattr(seg, "geom_type"):
                    geom = seg
                else:
                    geom = LineString(list(seg.coords))
            except Exception:
                continue
            geoms.append(geom)

            d = None
            try:
                d = float(s.segment_diameter_list[j])
            except Exception:
                pass

            rows.append({
                "stem_id": i,
                "node": j,
                "d": d,
            })

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def stems_to_gpkg(stems, profile, path_prefix):
    """Create/overwrite a GeoPackage at <path_prefix>.gpkg.

    Writes layer 'stems'.
    """
    gpkg_path = path_prefix + ".gpkg"
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)

    gdf = stems_to_gdf(stems, profile)
    if not gdf.empty:
        gdf.to_file(gpkg_path, layer="stems", driver="GPKG")
    print(f"Wrote {gpkg_path} (layer=stems)")
    return gpkg_path


def nodes_to_gpkg(stems, profile, path_prefix):
    """Append layer 'nodes' to <path_prefix>.gpkg."""
    gpkg_path = path_prefix + ".gpkg"
    gdf = nodes_to_gdf(stems, profile)
    if not gdf.empty:
        try:
            gdf.to_file(gpkg_path, layer="nodes", driver="GPKG", mode="a")
        except TypeError:
            # fallback if mode isn't supported in an environment
            gdf.to_file(gpkg_path, layer="nodes", driver="GPKG")
    print(f"Wrote {gpkg_path} (layer=nodes)")
    return gpkg_path


def vectors_to_gpkg(stems, profile, path_prefix):
    """Append layer 'vectors' to <path_prefix>.gpkg."""
    gpkg_path = path_prefix + ".gpkg"
    gdf = vectors_to_gdf(stems, profile)
    if not gdf.empty:
        try:
            gdf.to_file(gpkg_path, layer="vectors", driver="GPKG", mode="a")
        except TypeError:
            gdf.to_file(gpkg_path, layer="vectors", driver="GPKG")
    print(f"Wrote {gpkg_path} (layer=vectors)")
    return gpkg_path


def save_image(data, output_name, size=(15, 15), dpi=300):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(output_name, dpi=dpi)


def merge_and_filter_tiled_results(
    work_dir: str,
    output_gpkg: str | None = None,
    edge_buffer_m: float = 1.0,
): # noqa: C901
    """Merge tiled WINMOL results into one GeoPackage.

    Supported per-tile outputs in work_dir:
      1) GeoPackage (preferred): <prefix>*.gpkg with layers stems/nodes/
         vectors
      2) Legacy GeoJSON: <prefix>_roi_{stems,nodes,vectors}.geojson

    Optional per-tile raster for edge filtering:
      - <prefix>_roi_stem_map.tif or .tiff

    If a raster exists for a tile, stems are kept only if they intersect
    the inner tile extent (tile extent buffered inward by edge_buffer_m).
    This helps remove edge effects.

    Stem IDs are rewritten to be globally unique:
        stem_id = "<tile_id>_<local_stem_id>"

    Output:
      - output_gpkg (default: <work_dir>/<foldername>_merged_data.gpkg)
        Layers: stems, nodes, vectors
    """
    work_dir = os.path.abspath(work_dir)

    if output_gpkg is None:
        folder = os.path.basename(os.path.normpath(work_dir))
        output_gpkg = os.path.join(work_dir, f"{folder}_merged_data.gpkg")

    # Helper to find ID column
    def _pick_id_col(gdf):
        for c in ("stem_id", "id", "ID", "StemID", "stemID"):
            if c in gdf.columns:
                return c
        return None

    def _read_tile_gpkg(prefix):
        # accept {prefix}.gpkg or {prefix}_roi*.gpkg, etc.
        pattern = os.path.join(work_dir, f"{prefix}*.gpkg")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            return None
        gpkg = candidates[0]

        def _try(layer_names):
            for ln in layer_names:
                try:
                    return gpd.read_file(gpkg, layer=ln)
                except Exception:
                    continue
            return gpd.GeoDataFrame()

        stems = _try(["stems", "stem", "trees", "tree"])
        nodes = _try(["nodes", "node"])
        vectors = _try(["vectors", "vector", "segments", "segment"])
        return gpkg, stems, nodes, vectors

    def _read_tile_geojson(prefix):
        stems_path = os.path.join(work_dir, f"{prefix}_roi_stems.geojson")
        nodes_path = os.path.join(work_dir, f"{prefix}_roi_nodes.geojson")
        vectors_path = os.path.join(work_dir, f"{prefix}_roi_vectors.geojson")
        if not (
            os.path.exists(stems_path)
            and os.path.exists(nodes_path)
            and os.path.exists(vectors_path)
        ):
            return None
        stems = gpd.read_file(stems_path)
        nodes = gpd.read_file(nodes_path)
        vectors = gpd.read_file(vectors_path)
        return "geojson", stems, nodes, vectors

    merged_stems = gpd.GeoDataFrame()
    merged_nodes = gpd.GeoDataFrame()
    merged_vectors = gpd.GeoDataFrame()

    tile_count = total_stems = total_nodes = total_vectors = 0

    # Prefer raster-guided tile detection (best filtering)
    raster_files = [
        f
        for f in os.listdir(work_dir)
        if f.lower().endswith((".tif", ".tiff")) and "_roi_stem_map" in f
    ]

    # Fallback: if no rasters exist, treat each GPKG as a "tile" and merge
    # without edge filtering.
    if not raster_files:
        gpkg_files = sorted(
            [
                f
                for f in os.listdir(work_dir)
                if f.lower().endswith(".gpkg")
                and not f.lower().endswith("_merged_data.gpkg")
            ]
        )
        if not gpkg_files:
            raise FileNotFoundError(
                "No '*_roi_stem_map.tif(f)', no .gpkg, and no legacy GeoJSON "
                f"found in: {work_dir}"
            )
        print(
            "No '*_roi_stem_map.tif(f)' found -> merging without edge "
            "filtering."
        )
        for gpkg_file in gpkg_files:
            prefix = os.path.splitext(gpkg_file)[0]
            # tile id from 'raster_<id>' if possible
            m = re.match(r"raster_(.+)", prefix)
            tile_id = m.group(1) if m else prefix

            read = _read_tile_gpkg(prefix)
            if not read:
                continue
            _, stems, nodes, vectors = read

            if stems.empty:
                continue

            id_col = _pick_id_col(stems)
            if not id_col:
                print(f"Missing id column in {gpkg_file}, skipping.")
                continue

            stems = stems.copy()
            stems["_stem_id_local"] = stems[id_col].astype(str)
            stems["stem_id"] = tile_id + "_" + stems["_stem_id_local"]
            stems["tile_id"] = tile_id

            kept_local = set(stems["_stem_id_local"].tolist())

            def _prep_child(gdf):
                if gdf is None or gdf.empty:
                    return gpd.GeoDataFrame()
                c = _pick_id_col(gdf)
                if not c:
                    return gpd.GeoDataFrame()
                out = gdf[gdf[c].astype(str).isin(kept_local)].copy()
                out["_stem_id_local"] = out[c].astype(str)
                out["stem_id"] = tile_id + "_" + out["_stem_id_local"]
                out["tile_id"] = tile_id
                return out

            nodes = _prep_child(nodes)
            vectors = _prep_child(vectors)

            merged_stems = pd.concat(
                [merged_stems, stems],
                ignore_index=True,
            )
            if not nodes.empty:
                merged_nodes = pd.concat(
                    [merged_nodes, nodes],
                    ignore_index=True,
                )
            if not vectors.empty:
                merged_vectors = pd.concat(
                    [merged_vectors, vectors],
                    ignore_index=True,
                )

            tile_count += 1
            total_stems += len(stems)
            total_nodes += len(nodes)
            total_vectors += len(vectors)

    else:
        for raster_file in raster_files:
            # prefix from raster name (strip suffix)
            prefix = raster_file
            for ext in ["_roi_stem_map.tif", "_roi_stem_map.tiff"]:
                if raster_file.endswith(ext):
                    prefix = raster_file.replace(ext, "")
                    break

            m = re.match(r"raster_(.+)", prefix)
            if not m:
                print(f"Could not extract tile id from '{prefix}', skipping.")
                continue
            tile_id = m.group(1)

            raster_path = os.path.join(work_dir, raster_file)
            print("")
            print(f"Processing tile: {prefix}")

            # Read raster extent (for edge filtering)
            try:
                with rasterio.open(raster_path) as src:
                    bounds = src.bounds
                    raster_crs = src.crs
                    extent_geom = box(*bounds)
            except Exception as e:
                print(f"Error reading raster {raster_file}: {e}")
                continue

            neg_buffer_geom = extent_geom.buffer(-abs(edge_buffer_m))

            # Read vectors (prefer gpkg)
            read = _read_tile_gpkg(prefix)
            if read:
                _, stems, nodes, vectors = read
            else:
                read = _read_tile_geojson(prefix)
                if not read:
                    print(f"Missing vector data for {prefix}, skipping.")
                    continue
                _, stems, nodes, vectors = read

            # Ensure CRS matches raster CRS
            for gdf_name, gdf in (
                ("stems", stems),
                ("nodes", nodes),
                ("vectors", vectors),
            ):
                if gdf is None or gdf.empty:
                    continue
                if raster_crs is not None:
                    if gdf.crs is None:
                        gdf.set_crs(raster_crs, inplace=True)
                    elif gdf.crs != raster_crs:
                        try:
                            gdf.to_crs(raster_crs, inplace=True)
                        except Exception as e:
                            print(
                                f"CRS reprojection failed for {prefix} "
                                f"{gdf_name}: {e}"
                            )

            if stems.empty:
                print(f"No stems in {prefix}, skipping.")
                continue

            id_col = _pick_id_col(stems)
            if not id_col:
                print(f"Missing id column in stems for {prefix}, skipping.")
                continue

            # Create globally unique stem_id
            stems = stems.copy()
            stems["_stem_id_local"] = stems[id_col].astype(str)
            stems["stem_id"] = tile_id + "_" + stems["_stem_id_local"]
            stems["tile_id"] = tile_id

            # Edge filter
            stems_filtered = stems[stems.intersects(neg_buffer_geom)].copy()
            if stems_filtered.empty:
                print(f"No valid stems after filtering for {prefix}, skipping.")
                continue

            kept_local = set(stems_filtered["_stem_id_local"].tolist())

            def _prep_child(gdf):
                if gdf is None or gdf.empty:
                    return gpd.GeoDataFrame()
                c = _pick_id_col(gdf)
                if not c:
                    return gpd.GeoDataFrame()
                out = gdf[gdf[c].astype(str).isin(kept_local)].copy()
                out["_stem_id_local"] = out[c].astype(str)
                out["stem_id"] = tile_id + "_" + out["_stem_id_local"]
                out["tile_id"] = tile_id
                return out

            nodes_sel = _prep_child(nodes)
            vectors_sel = _prep_child(vectors)
            merged_stems = pd.concat(
                [merged_stems, stems_filtered],
                ignore_index=True,
            )
            if not nodes_sel.empty:
                merged_nodes = pd.concat(
                    [merged_nodes, nodes_sel],
                    ignore_index=True,
                )
            if not vectors_sel.empty:
                merged_vectors = pd.concat(
                    [merged_vectors, vectors_sel],
                    ignore_index=True,
                )

            tile_count += 1
            total_stems += len(stems_filtered)
            total_nodes += len(nodes_sel)
            total_vectors += len(vectors_sel)

    # Set CRS of merged outputs to whatever is present in stems/nodes/vectors
    # (they should all match when rasters exist)
    if not merged_stems.empty and merged_stems.crs is None:
        # attempt to inherit from any child
        if not merged_nodes.empty and merged_nodes.crs is not None:
            merged_stems.set_crs(merged_nodes.crs, inplace=True)
        elif not merged_vectors.empty and merged_vectors.crs is not None:
            merged_stems.set_crs(merged_vectors.crs, inplace=True)

    # Write output
    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    if not merged_stems.empty:
        merged_stems.to_file(output_gpkg, layer="stems", driver="GPKG")
    if not merged_nodes.empty:
        merged_nodes.to_file(output_gpkg, layer="nodes", driver="GPKG")
    if not merged_vectors.empty:
        merged_vectors.to_file(output_gpkg, layer="vectors", driver="GPKG")

    print("")
    print("MERGE SUMMARY")
    print(f"Tiles processed:       {tile_count}")
    print(f"Total stems written:   {total_stems}")
    print(f"Total nodes written:   {total_nodes}")
    print(f"Total vectors written: {total_vectors}")
    print(f"Output saved to: {output_gpkg}")
    return output_gpkg
