#!/usr/bin/env python

###############################################################################
"""Imports"""

import json
import os
import glob
import tempfile
import multiprocessing as mp
import shutil
import errno
import numpy as np
import rasterio
import geopandas as gpd
import fiona
import pandas as pd
import time
try:
    import pyogrio
    _HAVE_PYOGRIO = True
except Exception:
    pyogrio = None
    _HAVE_PYOGRIO = False
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point, box
from collections.abc import Mapping
from pyproj import CRS
from pathlib import Path

import utils.Quantification as Quant
from classes.Stem import Stem


###############################################################################

"""Streaming and tiling operations"""


def get_raster_info(path) -> dict:
    with rasterio.open(path) as src:
        dtype = src.dtypes[0] if src.dtypes else 'unknown'
        estimated_input_gb = (
            src.width * src.height * src.count * np.dtype(dtype).itemsize
        ) / (1024 ** 3)
        return {
            'width': int(src.width),
            'height': int(src.height),
            'bands': int(src.count),
            'dtype': str(dtype),
            'pixel_size_x': float(abs(src.transform.a)),
            'pixel_size_y': float(abs(src.transform.e)),
            'estimated_input_gb': float(estimated_input_gb),
            'crs': src.crs,
            'transform': src.transform,
        }


def atomic_tmp_path(final_path: str) -> str:
    p = Path(final_path)
    return str(p.with_suffix(p.suffix + '.tmp'))


def finalize_raster(tmp_path: str, final_path: str) -> str:
    os.replace(tmp_path, final_path)
    return final_path


def build_safe_prediction_profile(
    src_profile, width: int, height: int,
    transform, compress: str | None = 'DEFLATE',
    dtype: str = 'float32'
):
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1,
        'width': int(width),
        'height': int(height),
        'transform': transform,
        'crs': src_profile.get('crs', None),
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'BIGTIFF': 'YES',
        'interleave': 'BAND',
    }
    if compress is not None:
        c = str(compress).upper()
        if c in {'DEFLATE', 'LZW', 'ZSTD'}:
            profile['compress'] = c
            profile['predictor'] = 2
            if c == 'DEFLATE':
                profile['zlevel'] = 1
        elif c in {'NONE', 'OFF', 'FALSE', '0'}:
            pass
        else:
            profile['compress'] = 'DEFLATE'
            profile['predictor'] = 2
            profile['zlevel'] = 1
    return profile


def create_output_raster_like(
    src_path: str,
    dst_path: str,
    dtype: str = "float32",
    count: int = 1,
    width: int | None = None,
    height: int | None = None,
    transform=None,
    compress: str | None = "DEFLATE",
):
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = build_safe_prediction_profile(
            src.profile,
            width=int(width or src.width),
            height=int(height or src.height),
            transform=transform or src.transform,
            compress=compress,
        )
        profile['dtype'] = dtype
        profile['count'] = count
    with rasterio.open(dst_path, 'w', **profile):
        pass
    return profile


def read_window(path: str, window, bands:
                list[int] | None = None, boundless: bool = True, fill_value=0
                ):
    with rasterio.open(path) as src:
        indexes = bands if bands is not None else list(range(1, src.count + 1))
        return src.read(
            indexes, window=window, boundless=boundless, fill_value=fill_value)


def write_window(dst_path: str, array, window, band: int = 1):
    with rasterio.open(dst_path, 'r+') as dst:
        if array.ndim == 2:
            dst.write(array, band, window=window)
        else:
            dst.write(array, window=window)


def write_tile_raster(pred_tile, tile_profile, output_path: str):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    prof = build_safe_prediction_profile(
        tile_profile,
        width=pred_tile.shape[1],
        height=pred_tile.shape[0],
        transform=tile_profile['transform'],
        compress=None,
    )
    with rasterio.open(output_path, 'w', **prof) as dst:
        dst.write(pred_tile.astype(np.uint8), 1)
    return output_path


def iter_prediction_tiles(src_path: str, config, tile_jobs):
    with rasterio.open(src_path) as src:
        for job in tile_jobs:
            window = job.halo_window if hasattr(job, 'halo_window') else job
            arr = src.read([1], window=window, boundless=True, fill_value=0)[0]
            profile = src.profile.copy()
            profile['width'] = int(window.width)
            profile['height'] = int(window.height)
            profile['transform'] = rasterio.windows.transform(
                window, src.transform)
            yield job, arr, profile


def load_raster_window_with_profile(path: str, window):
    with rasterio.open(path) as src:
        pred = src.read(1, window=window, boundless=True, fill_value=0)
        profile = src.profile.copy()
        profile['width'] = int(window.width)
        profile['height'] = int(window.height)
        profile['transform'] = rasterio.windows.transform(
            window, src.transform)
        return pred, profile


"""File operations"""


def load_model_from_path(model_path):
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.utils import get_custom_objects

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
        img = src.read(list(range(1, config.n_channels + 1)))\
            .transpose(1, 2, 0)
        img = (img / 255).astype(np.float32)
        return img, src.profile


def load_orthomosaic_with_resampling(path, config):
    with rasterio.open(path) as src:
        scale_factor_x = src.res[0] / (config.tile_size / config.img_width)
        scale_factor_y = src.res[1] / (config.tile_size / config.img_width)
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
            pred = src.read(1)
            profile = src.profile
        return pred, profile
    raise ValueError(f'Unsupported stem map path: {path}')


def export_stem_map(pred, profile, pred_dir, pred_name, compress="DEFLATE"):
    final_path = os.path.join(pred_dir, f'{pred_name}.tiff')
    tmp_path = atomic_tmp_path(final_path)
    os.makedirs(pred_dir or '.', exist_ok=True)
    height, width = pred.shape
    safe_profile = build_safe_prediction_profile(
        src_profile=profile,
        width=width,
        height=height,
        transform=profile['transform'],
        compress=compress,
        dtype=str(pred.dtype),
    )
    with rasterio.open(tmp_path, 'w', **safe_profile) as dst:
        dst.write(pred.astype(pred.dtype, copy=False), 1)
    finalize_raster(tmp_path, final_path)


def get_bounds_from_profile(profile):
    left = profile['transform'][2]
    right = profile['transform'][2] \
        + profile['transform'][0] * profile['width']
    bot = profile['transform'][5] \
        + profile['transform'][4] * profile['height']
    top = profile['transform'][5]
    return rasterio.coords.BoundingBox(left, bot, right, top)


def _profile_get(profile, key, default=None):
    if profile is None:
        return default
    if hasattr(profile, "get"):
        return profile.get(key, default)
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _crs_from_profile(profile):
    crs_in = _profile_get(profile, "crs")
    if crs_in is None:
        return None

    try:
        crs = CRS.from_user_input(crs_in)
    except Exception:
        try:
            if hasattr(crs_in, "to_wkt"):
                crs = CRS.from_wkt(crs_in.to_wkt())
            else:
                return str(crs_in)
        except Exception:
            return str(crs_in)

    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        pass
    return CRS.from_epsg(epsg) if epsg else crs


def _jsonify_list(x):
    try:
        return json.dumps([float(v) for v in x], ensure_ascii=False)
    except Exception:
        return json.dumps(x, ensure_ascii=False)


def stems_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []
    for i, s in enumerate(stems):
        try:
            sx, sy = list(s.start.coords)[0]
        except Exception:
            sx, sy = (None, None)
        try:
            ex, ey = list(s.stop.coords)[0]
        except Exception:
            ex, ey = (None, None)

        geom = s.path if hasattr(s.path, "geom_type") else LineString(list(s.path.coords))
        geoms.append(geom)

        tree_x = getattr(s, "tree_x", sx)
        tree_y = getattr(s, "tree_y", sy)
        stem_id = getattr(s, 'stem_id', None)
        if stem_id in (None, '', -1):
            stem_id = str(i)
        rows.append({
            "stem_id": str(stem_id),
            "start_x": sx, "start_y": sy,
            "stop_x": ex, "stop_y": ey,
            "tree_x": tree_x, "tree_y": tree_y,
            "dir_x": float(getattr(s, "direction_x", 0.0) or 0.0),
            "dir_y": float(getattr(s, "direction_y", 0.0) or 0.0),
            "dir_deg": getattr(s, "direction_deg", None),
            "dir_conf": float(getattr(s, "direction_confidence", 0.0) or 0.0),
            "owner_pid": getattr(s, 'owner_partition_id', None),
            "src_tile": getattr(s, 'source_tile_id', None),
            "is_border": bool(getattr(s, 'is_border_candidate', False)),
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
            d = None
            try:
                d = float(s.segment_diameter_list[j])
            except Exception:
                pass
            rows.append({
                "stem_id": str(getattr(s, "stem_id", i)),
                "node": j,
                "d": d,
            })

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def vectors_to_gdf(stems, profile):
    crs = _crs_from_profile(profile)

    rows = []
    geoms = []

    for i in range(len(stems)):
        n_path = len(getattr(stems[i].path, "coords", []))
        vecs = getattr(stems[i], "vector", []) or []
        if len(vecs) < n_path:
            try:
                stems[i] = Quant.refresh_measurement_vectors(stems[i])
                vecs = getattr(stems[i], "vector", []) or []
            except Exception:
                vecs = getattr(stems[i], "vector", []) or []
        diams = getattr(stems[i], "segment_diameter_list", []) or []

        for j in range(n_path):
            if j >= len(vecs):
                continue

            v = vecs[j]
            try:
                raw_coords = list(v.coords[:])
            except Exception:
                continue

            coords = []
            bad_coords = False
            for xy in raw_coords:
                try:
                    x = float(xy[0])
                    y = float(xy[1])
                    if not np.isfinite(x) or not np.isfinite(y):
                        bad_coords = True
                        break
                    coords.append((x, y))
                except Exception:
                    bad_coords = True
                    break

            if bad_coords or len(coords) < 2:
                continue

            try:
                geom = LineString(coords)
            except Exception:
                continue

            try:
                if geom.is_empty or (not geom.is_valid) or geom.length <= 0:
                    continue
            except Exception:
                continue

            d = diams[j] if j < len(diams) else None
            try:
                d = float(d) if d is not None else None
            except Exception:
                d = None

            rows.append({
                "stem_id": str(getattr(stems[i], "stem_id", i)),
                "node": int(j),
                "d": d,
            })
            geoms.append(geom)

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def _safe_finalize_gpkg(tmp_path: str, final_path: str) -> str:
    tmp_path = str(tmp_path)
    final_path = str(final_path)

    tmp_dir = Path(tmp_path).parent
    dst = Path(final_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _cleanup():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _copy_to(target: Path) -> str:
        shutil.copy2(tmp_path, str(target))
        _cleanup()
        return str(target)

    def _alt_new_paths(base: Path):
        yield base.with_name(base.stem + "_new" + base.suffix)
        yield base.with_name(base.stem + f"_new_{os.getpid()}" + base.suffix)

    try:
        os.replace(tmp_path, final_path)
        _cleanup()
        return final_path

    except PermissionError:
        for alt in _alt_new_paths(dst):
            try:
                return _copy_to(alt)
            except PermissionError:
                continue
        raise

    except OSError as e:
        exdev = getattr(errno, "EXDEV", 18)
        if e.errno in (18, exdev):
            try:
                return _copy_to(dst)
            except PermissionError:
                for alt in _alt_new_paths(dst):
                    try:
                        return _copy_to(alt)
                    except PermissionError:
                        continue
            raise
        raise


def _drop_bad_geoms(gdf):
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()].copy()
    try:
        gdf = gdf[~gdf.geometry.is_empty].copy()
    except Exception:
        pass
    try:
        gdf = gdf[gdf.geometry.is_valid].copy()
    except Exception:
        pass
    try:
        geom_types = gdf.geometry.geom_type.astype(str)
        is_pointlike = geom_types.isin(["Point", "MultiPoint"])

        lengths = gdf.geometry.length
        keep = is_pointlike | ((~lengths.isna()) & (lengths > 0))

        gdf = gdf[keep].copy()
    except Exception:
        pass
    return gdf


def _normalize_dtypes(gdf):
    if gdf is None or gdf.empty:
        return gdf

    def _all_instance(values, types_):
        try:
            return all(isinstance(v, types_) for v in values)
        except Exception:
            return False

    for col in list(gdf.columns):
        if col == gdf.geometry.name:
            continue

        s = gdf[col]
        dt = str(s.dtype).lower()
        non_null = [v for v in s.tolist() if pd.notna(v)]

        if not non_null:
            gdf[col] = s.astype(object).where(~s.isna(), None)
            continue

        if "int" in dt and "interval" not in dt:
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            continue

        if "float" in dt:
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            continue

        if "bool" in dt:
            gdf[col] = s.astype(object).where(~s.isna(), None)
            continue

        if "string" in dt or _all_instance(non_null, str):
            gdf[col] = s.astype(object).where(~s.isna(), None)
            continue

        if _all_instance(non_null, (int, np.integer)):
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            continue

        if _all_instance(non_null, (int, float, np.integer, np.floating)):
            gdf[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            continue

        gdf[col] = s.map(lambda v: None if pd.isna(v)
                         else str(v)).astype(object)

    return gdf


def _schema_type_for_series(series):
    non_null = [v for v in series.tolist() if pd.notna(v)]
    if not non_null:
        return "str"

    def _all_instance(values, types_):
        try:
            return all(isinstance(v, types_) for v in values)
        except Exception:
            return False

    dt = str(series.dtype).lower()
    if "int" in dt and "interval" not in dt:
        return "int"
    if "float" in dt:
        return "float"
    if _all_instance(non_null, (bool, np.bool_)):
        return "int"
    if _all_instance(non_null, (int, np.integer)):
        return "int"
    if _all_instance(non_null, (int, float, np.integer, np.floating)):
        return "float"
    return "str"


def _infer_geometry_type(gdf):
    try:
        geom_types = [g.geom_type for g in gdf.geometry
                      if g is not None and not g.is_empty]
    except Exception:
        geom_types = []

    if not geom_types:
        return "Unknown"

    unique = set(geom_types)
    if len(unique) == 1:
        return next(iter(unique))

    if unique <= {"LineString", "MultiLineString"}:
        return "MultiLineString"
    if unique <= {"Point", "MultiPoint"}:
        return "MultiPoint"
    if unique <= {"Polygon", "MultiPolygon"}:
        return "MultiPolygon"
    return "Unknown"


def _infer_fiona_schema(gdf):
    props = {}
    geom_col = gdf.geometry.name
    for col in gdf.columns:
        if col == geom_col:
            continue
        props[str(col)] = _schema_type_for_series(gdf[col])
    return {
        "geometry": _infer_geometry_type(gdf),
        "properties": props,
    }


def _feature_records(gdf, schema):
    geom_col = gdf.geometry.name
    prop_types = schema.get("properties", {})

    for _, row in gdf.iterrows():
        geom = row[geom_col]
        if geom is None:
            continue
        try:
            if geom.is_empty:
                continue
        except Exception:
            pass

        props = {}
        for col, typ in prop_types.items():
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif typ == "int":
                try:
                    props[col] = int(val)
                except Exception:
                    props[col] = None
            elif typ == "float":
                try:
                    props[col] = float(val)
                except Exception:
                    props[col] = None
            else:
                props[col] = str(val)

        yield {
            "geometry": geom.__geo_interface__,
            "properties": props,
        }


def _fiona_write_layer(path, layer_name, gdf, crs, append=False):
    schema = _infer_fiona_schema(gdf)

    layer_exists = False
    if os.path.exists(path):
        try:
            layer_exists = layer_name in set(fiona.listlayers(path))
        except Exception:
            layer_exists = False

    mode = "a" if append and layer_exists else "w"

    crs_wkt = None
    if crs is not None:
        try:
            crs_wkt = CRS.from_user_input(crs).to_wkt()
        except Exception:
            try:
                crs_wkt = crs.to_wkt()
            except Exception:
                crs_wkt = None

    print(
        f"Fiona schema for layer '{layer_name}': {schema} | mode {mode} | "
        f"layer_exists {layer_exists}",
    )

    kwargs = {
        "driver": "GPKG",
        "layer": layer_name,
        "schema": schema,
    }
    if crs_wkt is not None:
        kwargs["crs_wkt"] = crs_wkt

    with fiona.open(path, mode=mode, **kwargs) as dst:
        dst.writerecords(_feature_records(gdf, schema))


def _write_layers_to_temp_gpkg(  # noqa: C901
    layers, crs, final_path: str) \
        -> str:
    final_parent = Path(final_path).parent
    final_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="winmol_gpkg_", dir=str(final_parent))
    tmp_path = str(Path(tmp_dir) / (Path(final_path).stem + ".gpkg"))

    def _prep(name, gdf):
        if gdf is None:
            return None
        gdf = _drop_bad_geoms(_normalize_dtypes(gdf))
        if gdf is None or gdf.empty:
            return None

        if crs is not None:
            try:
                if (gdf.crs is None) or (gdf.crs != crs):
                    gdf = gdf.set_crs(crs, allow_override=True)
            except Exception:
                try:
                    gdf = gdf.set_crs(crs, allow_override=True)
                except Exception:
                    pass

        return gdf

    prepared = []
    for name, gdf in layers:
        pgdf = _prep(name, gdf)
        if pgdf is not None:
            prepared.append((name, pgdf))

    if not prepared:
        layer_name = layers[0][0] if layers else "layer"
        empty = gpd.GeoDataFrame(
            {"geometry": []}, geometry="geometry", crs=crs)
        empty.to_file(tmp_path, layer=layer_name, driver="GPKG", index=False)
        return tmp_path

    if _HAVE_PYOGRIO:
        try:
            first = True
            for name, gdf in prepared:
                pyogrio.write_dataframe(
                    gdf,
                    tmp_path,
                    layer=name,
                    driver="GPKG",
                    append=(not first),
                )
                first = False
            return tmp_path
        except Exception as e:
            print("pyogrio GeoPackage write failed, "
                  f"falling back to Fiona: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    first = True
    for name, gdf in prepared:
        print(f"Writing GPKG layer '{name}' with {len(gdf)} features")
        print(f"Layer '{name}' dtypes: {dict(gdf.dtypes.astype(str))}")
        try:
            if first:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                _fiona_write_layer(tmp_path, name, gdf, crs=crs, append=False)
                first = False
            else:
                _fiona_write_layer(tmp_path, name, gdf, crs=crs, append=True)
        except Exception as e:
            raise RuntimeError(f"Failed while writing GeoPackage layer '{name}'"
                               f" at'{tmp_path}'") from e
    return tmp_path


def write_stems_to_gpkg(stems, profile, path_prefix):
    final_path = str(Path(path_prefix).with_suffix(".gpkg"))
    crs = _crs_from_profile(profile)
    gdf_stems = stems_to_gdf(stems, profile)

    tmp_path = _write_layers_to_temp_gpkg(
        layers=[("stems", gdf_stems)],
        crs=crs,
        final_path=final_path,
    )
    return _safe_finalize_gpkg(tmp_path, final_path)


def write_all_layers_to_gpkg(stems, profile, path_prefix):
    final_path = str(Path(path_prefix).with_suffix(".gpkg"))
    crs = _crs_from_profile(profile)

    gdf_stems = stems_to_gdf(stems, profile)
    gdf_vectors = vectors_to_gdf(stems, profile)
    gdf_nodes = nodes_to_gdf(stems, profile)

    tmp_path = _write_layers_to_temp_gpkg(
        layers=[
            ("stems", gdf_stems),
            ("vectors", gdf_vectors),
            ("nodes", gdf_nodes),
        ],
        crs=crs,
        final_path=final_path,
    )
    print("Geopackage written to temporary file:", tmp_path)
    return _safe_finalize_gpkg(tmp_path, final_path)


def save_image(data, output_name, size=(15, 15), dpi=300):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(output_name, dpi=dpi)


#############################################################################


"""Merge and filter tiled results"""


def _pick_id_col(gdf):
    for c in ("stem_id", "id", "ID", "StemID", "stemID"):
        if c in gdf.columns:
            return c
    return None


def _tile_id_from_prefix(prefix):
    if prefix.startswith("raster_"):
        return prefix[len("raster_"):]
    return prefix


def _default_output_gpkg(
    work_dir,
    output_gpkg=None,
    input_raster_path=None,
    stem_map_path=None,
):
    if output_gpkg:
        out_path = Path(output_gpkg)
        if out_path.suffix.lower() != ".gpkg":
            out_path = out_path.with_suffix(".gpkg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    base = "winmol_merged"
    if input_raster_path:
        try:
            base = Path(input_raster_path).stem
        except Exception:
            base = "winmol_merged"

    out_dir = Path(stem_map_path).parent if stem_map_path else Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{base}.gpkg")


def _remove_existing_output(output_gpkg):
    if not output_gpkg:
        return
    base = Path(output_gpkg)
    for ext in ("", "-wal", "-shm", ".gpkg-journal"):
        candidate = Path(str(base) + ext) if ext.startswith('-') else Path(str(base) + ext)
        try:
            if candidate.exists():
                candidate.unlink()
        except Exception:
            pass


def _ensure_crs(gdf, target_crs):
    if gdf is None or gdf.empty or target_crs is None:
        return gdf
    try:
        if gdf.crs is None:
            gdf = gdf.set_crs(target_crs, allow_override=True)
            return gdf
        if str(gdf.crs) != str(target_crs):
            return gdf.to_crs(target_crs)
    except Exception:
        return gdf
    return gdf


def _detect_tiles(work_dir, output_gpkg=None):
    root = Path(work_dir)
    output_path = Path(output_gpkg).resolve() if output_gpkg else None
    gpkg_paths = sorted(root.rglob('*.gpkg'))
    tiles = []
    for gpkg_path in gpkg_paths:
        try:
            if output_path is not None and gpkg_path.resolve() == output_path:
                continue
        except Exception:
            pass
        prefix = gpkg_path.stem
        raster_path = gpkg_path.with_name(f"{prefix}_roi_stem_map.tif")
        if not raster_path.exists():
            alt = root / f"{prefix}_roi_stem_map.tif"
            raster_path = alt if alt.exists() else None
        tiles.append((prefix, str(gpkg_path), str(raster_path) if raster_path else None))
    return tiles


def _process_tile(prefix, gpkg_path, raster_path, edge_buffer_m, target_crs):
    stems_gdf = _read_gpkg_layer(gpkg_path, ['stems'])
    if stems_gdf is None or stems_gdf.empty:
        return None, target_crs
    current_crs = target_crs
    if current_crs is None:
        current_crs = getattr(stems_gdf, 'crs', None)
        if current_crs is None and raster_path:
            try:
                with rasterio.open(raster_path) as src:
                    current_crs = src.crs
            except Exception:
                current_crs = None
    stems_gdf = _ensure_crs(stems_gdf, current_crs)
    if 'tile_id' not in stems_gdf.columns:
        stems_gdf['tile_id'] = _tile_id_from_prefix(prefix)
    if 'src_tile' not in stems_gdf.columns:
        stems_gdf['src_tile'] = _tile_id_from_prefix(prefix)
    return stems_gdf, current_crs


def _read_gpkg_layer(gpkg_path, layer_names):
    errors = []
    for ln in layer_names:
        try:
            with fiona.open(gpkg_path, layer=ln) as src:
                feats = list(src)
                try:
                    crs = src.crs_wkt or src.crs
                except Exception:
                    crs = None

            if not feats:
                gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
            else:
                gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
            print(
                f"MERGE READ OK | file {gpkg_path} | layer {ln} | rows {len(gdf)}",
                flush=True,
            )
            return gdf
        except Exception as exc:
            errors.append((ln, exc))

    if errors:
        tried = ", ".join(
            f"{ln}: {type(exc).__name__}: {exc}" for ln, exc in errors
        )
        print(
            f"MERGE READ FAIL | file {gpkg_path} | tried [{tried}]",
            flush=True,
        )
    else:
        print(
            f"MERGE READ FAIL | file {gpkg_path} | tried []",
            flush=True,
        )
    return gpd.GeoDataFrame(geometry=[])




def _concat_merged_layer(gdfs):
    if not gdfs:
        return None

    non_empty = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
    if not non_empty:
        return None

    first = non_empty[0]
    geom_col = first.geometry.name
    crs = first.crs
    merged = pd.concat(non_empty, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry=geom_col, crs=crs)



def _write_merged(output_gpkg, merged_stems, merged_nodes, merged_vectors):
    stems_gdf = _concat_merged_layer(merged_stems)
    nodes_gdf = _concat_merged_layer(merged_nodes)
    vectors_gdf = _concat_merged_layer(merged_vectors)

    crs = None
    for gdf in (stems_gdf, nodes_gdf, vectors_gdf):
        if gdf is not None and getattr(gdf, 'crs', None) is not None:
            crs = gdf.crs
            break

    tmp_path = _write_layers_to_temp_gpkg(
        layers=[
            ("stems", stems_gdf),
            ("nodes", nodes_gdf),
            ("vectors", vectors_gdf),
        ],
        crs=crs,
        final_path=output_gpkg,
    )
    return _safe_finalize_gpkg(tmp_path, output_gpkg)


def _json_list_or_empty(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _stem_from_row(row):
    geom = getattr(row, 'geometry', None)
    if geom is None or getattr(geom, 'is_empty', True):
        return None
    try:
        coords = list(geom.coords)
    except Exception:
        return None
    if len(coords) < 2:
        return None
    sx = getattr(row, 'start_x', None)
    sy = getattr(row, 'start_y', None)
    ex = getattr(row, 'stop_x', None)
    ey = getattr(row, 'stop_y', None)
    start_xy = coords[0] if sx is None or sy is None else (float(sx), float(sy))
    stop_xy = coords[-1] if ex is None or ey is None else (float(ex), float(ey))
    stem = Stem(
        start=Point(start_xy),
        stop=Point(stop_xy),
        path=LineString(coords),
        vector=[],
        segment_diameter_list=[float(v) for v in _json_list_or_empty(getattr(row, 'd_json', []))],
        segment_length_list=[float(v) for v in _json_list_or_empty(getattr(row, 'l_json', []))],
        segment_volume_list=[float(v) for v in _json_list_or_empty(getattr(row, 'v_json', []))],
        stem_id=getattr(row, 'stem_id', None),
        crs=getattr(row, 'crs', None),
        tree_x=getattr(row, 'tree_x', None),
        tree_y=getattr(row, 'tree_y', None),
        direction_x=float(getattr(row, 'dir_x', 0.0) or 0.0),
        direction_y=float(getattr(row, 'dir_y', 0.0) or 0.0),
        direction_deg=getattr(row, 'dir_deg', None),
        direction_confidence=float(getattr(row, 'dir_conf', 0.0) or 0.0),
        owner_partition_id=getattr(row, 'owner_pid', None),
        source_tile_id=getattr(row, 'src_tile', getattr(row, 'tile_id', None)),
        is_border_candidate=bool(getattr(row, 'is_border', False)),
    )
    stem = Quant.refresh_stem_direction(stem)
    stem = Quant.refresh_measurement_vectors(stem)
    return stem


def _stems_from_gdf(stems_gdf):
    stems = []
    if stems_gdf is None or stems_gdf.empty:
        return stems
    crs = getattr(stems_gdf, 'crs', None)
    for row in stems_gdf.itertuples(index=False):
        stem = _stem_from_row(row)
        if stem is None:
            continue
        stem.crs = crs
        stems.append(stem)
    return stems


def _stems_to_layer_gdfs(stems, crs):
    profile = {'crs': crs}
    stems_gdf = stems_to_gdf(stems, profile) if stems else None
    nodes_gdf = nodes_to_gdf(stems, profile) if stems else None
    vectors_gdf = vectors_to_gdf(stems, profile) if stems else None
    return stems_gdf, nodes_gdf, vectors_gdf


def _tile_edge_band_geom(raster_path, edge_buffer_m):
    if not raster_path:
        return None, None
    try:
        with rasterio.open(raster_path) as src:
            geom = box(*src.bounds)
            inner = geom.buffer(-abs(edge_buffer_m))
            if inner.is_empty:
                band = geom
            else:
                band = geom.difference(inner)
            return band, src.crs
    except Exception:
        return None, None


def _reverse_stem_profile(stem):
    coords = list(stem.path.coords)
    rev_coords = list(reversed(coords))
    rev_d = list(reversed(list(getattr(stem, 'segment_diameter_list', []))))
    rev_l = list(reversed(list(getattr(stem, 'segment_length_list', []))))
    rev_v = list(reversed(list(getattr(stem, 'segment_volume_list', []))))
    return Stem(
        start=Point(rev_coords[0]),
        stop=Point(rev_coords[-1]),
        path=LineString(rev_coords),
        vector=list(getattr(stem, 'vector', [])),
        segment_diameter_list=rev_d,
        segment_length_list=rev_l,
        segment_volume_list=rev_v,
        crs=getattr(stem, 'crs', None),
    )


def _orient_stem_along_merged_path(stem, merged_path):
    try:
        s_proj = merged_path.project(Point(stem.start.coords[0]))
        e_proj = merged_path.project(Point(stem.stop.coords[0]))
    except Exception:
        return stem
    if s_proj <= e_proj:
        return stem
    return _reverse_stem_profile(stem)


def _match_edge_contributors(merged_stem, original_edge_stems, used, tol=1e-6):
    contributors = []
    try:
        band = merged_stem.path.buffer(max(float(tol), 1e-6))
    except Exception:
        band = None
    for idx, stem in enumerate(original_edge_stems):
        if idx in used:
            continue
        try:
            same_geom = stem.path.equals(merged_stem.path)
        except Exception:
            same_geom = False
        try:
            intersects = band.intersects(stem.path) if band is not None else False
        except Exception:
            intersects = False
        try:
            endpoints_on_path = (
                merged_stem.path.distance(stem.start) <= tol
                or merged_stem.path.distance(stem.stop) <= tol
            )
        except Exception:
            endpoints_on_path = False
        if same_geom or intersects or endpoints_on_path:
            contributors.append((idx, stem))
    return contributors


def _rebuild_connected_stem_profile(merged_stem, contributors):
    if not contributors:
        stem = Stem(
            start=Point(merged_stem.start.coords[0]),
            stop=Point(merged_stem.stop.coords[0]),
            path=LineString(list(merged_stem.path.coords)),
            vector=list(getattr(merged_stem, 'vector', [])),
            segment_diameter_list=list(getattr(merged_stem, 'segment_diameter_list', [])),
            segment_length_list=list(getattr(merged_stem, 'segment_length_list', [])),
            segment_volume_list=list(getattr(merged_stem, 'segment_volume_list', [])),
            crs=getattr(merged_stem, 'crs', None),
        )
        return Quant.quantify_stem(stem)

    oriented = []
    for _, stem in contributors:
        s = _orient_stem_along_merged_path(stem, merged_stem.path)
        try:
            start_proj = merged_stem.path.project(Point(s.start.coords[0]))
            stop_proj = merged_stem.path.project(Point(s.stop.coords[0]))
            pos = min(start_proj, stop_proj)
        except Exception:
            pos = 0.0
        oriented.append((pos, s))
    oriented.sort(key=lambda item: item[0])

    coords = []
    diameters = []
    for _, stem in oriented:
        stem_coords = list(stem.path.coords)
        stem_d = list(getattr(stem, 'segment_diameter_list', []))
        if not stem_coords:
            continue
        if not stem_d:
            stem_d = [0.0] * len(stem_coords)
        if len(stem_d) < len(stem_coords):
            fill = stem_d[-1] if stem_d else 0.0
            stem_d = stem_d + [fill] * (len(stem_coords) - len(stem_d))
        elif len(stem_d) > len(stem_coords):
            stem_d = stem_d[:len(stem_coords)]

        if not coords:
            coords = stem_coords[:]
            diameters = stem_d[:]
            continue

        if tuple(coords[-1]) == tuple(stem_coords[0]):
            coords.extend(stem_coords[1:])
            diameters.extend(stem_d[1:])
        else:
            coords.extend(stem_coords)
            diameters.extend(stem_d)

    if len(coords) < 2:
        coords = list(merged_stem.path.coords)
    if len(diameters) < len(coords):
        fill = diameters[-1] if diameters else 0.0
        diameters = diameters + [fill] * (len(coords) - len(diameters))
    elif len(diameters) > len(coords):
        diameters = diameters[:len(coords)]

    rebuilt = Stem(
        start=Point(coords[0]),
        stop=Point(coords[-1]),
        path=LineString(coords),
        vector=list(getattr(merged_stem, 'vector', [])),
        segment_diameter_list=diameters,
        segment_length_list=[],
        segment_volume_list=[],
        crs=getattr(merged_stem, 'crs', None),
    )
    return Quant.quantify_stem(rebuilt)


def _reconstruct_edge_stems_for_tiled_merge(
    merged_stems,
    tiles,
    target_crs,
    edge_buffer_m,
    config=None,
    stem_map_path=None,
):
    stems_gdf = _concat_merged_layer(merged_stems)
    if stems_gdf is None or stems_gdf.empty:
        return stems_gdf, None, None

    if target_crs is not None:
        stems_gdf = _ensure_crs(stems_gdf, target_crs)

    edge_indices = set()
    for prefix, _gpkg_path, raster_path in tiles:
        tile_id = _tile_id_from_prefix(prefix)
        tile_rows = stems_gdf[stems_gdf.get('tile_id', '') == tile_id]
        if tile_rows.empty:
            continue
        edge_band, raster_crs = _tile_edge_band_geom(raster_path, edge_buffer_m)
        if edge_band is None:
            continue
        tile_rows = _ensure_crs(tile_rows, target_crs or raster_crs)
        try:
            idx = tile_rows[tile_rows.intersects(edge_band)].index.tolist()
        except Exception:
            idx = []
        edge_indices.update(idx)

    print(
        f"MERGE EDGE RECON | candidates {len(edge_indices)} | total {len(stems_gdf)} | edge_buffer_m {edge_buffer_m}",
        flush=True,
    )

    all_stems = _stems_from_gdf(stems_gdf)
    if not edge_indices:
        return _stems_to_layer_gdfs(all_stems, target_crs)

    edge_gdf = stems_gdf.loc[sorted(edge_indices)].copy()
    inner_gdf = stems_gdf.drop(index=sorted(edge_indices)).copy()

    inner_stems = _stems_from_gdf(inner_gdf)
    original_edge_stems = _stems_from_gdf(edge_gdf)

    if not original_edge_stems:
        return _stems_to_layer_gdfs(inner_stems, target_crs)

    import utils.Vectorization as Vec

    if config is None:
        from classes.Config import Config
        recon_cfg = Config()
    else:
        recon_cfg = config

    merged_edge_stems = Vec.connect_stems(list(original_edge_stems), recon_cfg)
    Vec.rebuild_endnodes_from_stems(merged_edge_stems)

    used = set()
    rebuilt_edge_stems = []
    for merged_stem in merged_edge_stems:
        contributors = _match_edge_contributors(merged_stem, original_edge_stems, used)
        for idx, _ in contributors:
            used.add(idx)
        rebuilt_edge_stems.append(
            _rebuild_connected_stem_profile(merged_stem, contributors)
        )

    for idx, stem in enumerate(original_edge_stems):
        if idx not in used:
            rebuilt_edge_stems.append(Quant.quantify_stem(stem))

    final_stems = inner_stems + rebuilt_edge_stems
    print(
        f"MERGE EDGE RECON | inner {len(inner_stems)} | reconstructed_edge {len(rebuilt_edge_stems)} | final {len(final_stems)}",
        flush=True,
    )
    return _stems_to_layer_gdfs(final_stems, target_crs)




def _default_vector_partition_overlap_m(config):
    value = getattr(config, 'vector_partition_overlap_m', None)
    if value not in (None, ''):
        return float(value)
    return float(getattr(config, 'max_distance', 8.0)) + 2.0 * float(getattr(config, 'measuring_point_spacing_m', 0.5))


def _default_vector_partition_border_band_m(config):
    value = getattr(config, 'vector_partition_border_band_m', None)
    if value not in (None, ''):
        return float(value)
    return _default_vector_partition_overlap_m(config)


def _partition_bounds_from_stems(stems_gdf, partition_size_m):
    minx, miny, maxx, maxy = stems_gdf.total_bounds
    if not np.isfinite([minx, miny, maxx, maxy]).all():
        minx = miny = 0.0
        maxx = maxy = partition_size_m
    return float(minx), float(miny), float(maxx), float(maxy)


def _partition_id(ix, iy):
    return f"vp_{int(ix):05d}_{int(iy):05d}"


def _partition_polygon(ix, iy, minx, miny, size_m):
    x0 = minx + ix * size_m
    y0 = miny + iy * size_m
    return box(x0, y0, x0 + size_m, y0 + size_m)


def _owner_xy_from_row(row):
    tx = getattr(row, 'tree_x', None)
    ty = getattr(row, 'tree_y', None)
    if tx is not None and ty is not None and np.isfinite([tx, ty]).all():
        return float(tx), float(ty)
    sx = getattr(row, 'start_x', None)
    sy = getattr(row, 'start_y', None)
    if sx is not None and sy is not None and np.isfinite([sx, sy]).all():
        return float(sx), float(sy)
    geom = getattr(row, 'geometry', None)
    try:
        p = geom.centroid
        return float(p.x), float(p.y)
    except Exception:
        return 0.0, 0.0


def _assign_owner_partition(tree_x, tree_y, minx, miny, size_m):
    ix = int(np.floor((float(tree_x) - minx) / size_m))
    iy = int(np.floor((float(tree_y) - miny) / size_m))
    return _partition_id(ix, iy), ix, iy


def _intersecting_partitions(geom, minx, miny, size_m, overlap_m):
    minxg, minyg, maxxg, maxyg = geom.bounds
    ix0 = int(np.floor((minxg - overlap_m - minx) / size_m))
    iy0 = int(np.floor((minyg - overlap_m - miny) / size_m))
    ix1 = int(np.floor((maxxg + overlap_m - minx) / size_m))
    iy1 = int(np.floor((maxyg + overlap_m - miny) / size_m))
    out = []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            cell = _partition_polygon(ix, iy, minx, miny, size_m).buffer(overlap_m)
            try:
                if cell.intersects(geom):
                    out.append((ix, iy, _partition_id(ix, iy)))
            except Exception:
                continue
    return out


def build_vector_partitions(merged_stems_gdf, config):
    if merged_stems_gdf is None or merged_stems_gdf.empty:
        return {}, {}
    size_m = float(getattr(config, 'vector_partition_size_m', 80.0) or 80.0)
    overlap_m = _default_vector_partition_overlap_m(config)
    border_m = _default_vector_partition_border_band_m(config)
    minx, miny, maxx, maxy = _partition_bounds_from_stems(merged_stems_gdf, size_m)
    metadata = {
        'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
        'size_m': size_m, 'overlap_m': overlap_m, 'border_m': border_m,
        'crs': merged_stems_gdf.crs,
    }
    partition_rows = {}
    partition_meta = {}
    for row in merged_stems_gdf.itertuples(index=False):
        geom = getattr(row, 'geometry', None)
        if geom is None or getattr(geom, 'is_empty', True):
            continue
        tree_x, tree_y = _owner_xy_from_row(row)
        owner_pid, owner_ix, owner_iy = _assign_owner_partition(tree_x, tree_y, minx, miny, size_m)
        for ix, iy, pid in _intersecting_partitions(geom, minx, miny, size_m, overlap_m):
            core_geom = _partition_polygon(ix, iy, minx, miny, size_m)
            border_geom = core_geom.boundary.buffer(border_m, cap_style=3, join_style=2)
            expanded_geom = core_geom.buffer(overlap_m, cap_style=3, join_style=2)
            partition_meta[pid] = {
                'ix': ix, 'iy': iy, 'core_geom': core_geom,
                'border_geom': border_geom, 'expanded_geom': expanded_geom,
            }
            rec = row._asdict()
            rec['owner_pid'] = owner_pid
            rec['_owner_ix'] = owner_ix
            rec['_owner_iy'] = owner_iy
            rec['_partition_id'] = pid
            rec['_is_owner_partition'] = bool(pid == owner_pid)
            try:
                rec['_is_border_candidate'] = bool(geom.intersects(border_geom))
            except Exception:
                rec['_is_border_candidate'] = False
            partition_rows.setdefault(pid, []).append(rec)
    partitions = {}
    for pid, rows in partition_rows.items():
        geom_col = 'geometry'
        partitions[pid] = {
            'partition_id': pid,
            'stems_gdf': gpd.GeoDataFrame(rows, geometry=geom_col, crs=merged_stems_gdf.crs),
            **partition_meta[pid],
        }
    return partitions, metadata


def _owner_partition_for_stem(stem, metadata):
    tx = getattr(stem, 'tree_x', None)
    ty = getattr(stem, 'tree_y', None)
    if tx is None or ty is None:
        try:
            tx, ty = stem.start.coords[0]
        except Exception:
            tx, ty = (metadata['minx'], metadata['miny'])
    pid, _ix, _iy = _assign_owner_partition(tx, ty, metadata['minx'], metadata['miny'], metadata['size_m'])
    return pid


def _partition_border_for_pid(pid, metadata, partition_meta):
    if partition_meta and pid in partition_meta:
        return partition_meta[pid]['border_geom']
    parts = pid.split('_')
    ix, iy = int(parts[-2]), int(parts[-1])
    return _partition_polygon(ix, iy, metadata['minx'], metadata['miny'], metadata['size_m']).boundary.buffer(metadata['border_m'], cap_style=3, join_style=2)


def process_vector_partition(payload, config, metadata):
    import utils.Vectorization as Vec
    stems_gdf = payload.get('stems_gdf')
    pid = payload.get('partition_id')
    if stems_gdf is None or stems_gdf.empty:
        return {'partition_id': pid, 'core_stems_gdf': gpd.GeoDataFrame(), 'border_stems_gdf': gpd.GeoDataFrame()}
    stems = _stems_from_gdf(stems_gdf)
    stems = Quant.ensure_stem_ids(stems, prefix=f"{pid}_part")
    stems = Quant.refresh_stems_direction_bulk(stems, config=config)
    stems, _dup = Vec.remove_duplicates_spatial(stems, config=config)
    stems = Vec.connect_stems(stems, config)
    stems = Quant.refresh_stems_direction_bulk(stems, config=config)
    border_geom = payload.get('border_geom')
    core = []
    border = []
    for stem in stems:
        owner_pid = _owner_partition_for_stem(stem, metadata)
        stem.owner_partition_id = owner_pid
        stem.is_border_candidate = False
        if owner_pid != pid:
            continue
        try:
            is_border = bool(border_geom is not None and stem.path.intersects(border_geom))
        except Exception:
            is_border = False
        stem.is_border_candidate = is_border
        if is_border:
            border.append(stem)
        else:
            core.append(stem)
    profile = {'crs': metadata.get('crs')}
    return {
        'partition_id': pid,
        'core_stems_gdf': stems_to_gdf(core, profile) if core else gpd.GeoDataFrame(geometry=[], crs=metadata.get('crs')),
        'border_stems_gdf': stems_to_gdf(border, profile) if border else gpd.GeoDataFrame(geometry=[], crs=metadata.get('crs')),
    }


def _neighbor_partition_pairs(partition_ids):
    idx = {}
    for pid in partition_ids:
        parts = pid.split('_')
        idx[pid] = (int(parts[-2]), int(parts[-1]))
    by_xy = {v: k for k, v in idx.items()}
    pairs = []
    for pid, (ix, iy) in idx.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = by_xy.get((ix + dx, iy + dy))
            if other is not None:
                pairs.append((pid, other))
    return pairs


def stitch_partition_border_pair(pair_payload, config, metadata):
    import utils.Vectorization as Vec
    pid_a, pid_b, gdf_a, gdf_b = pair_payload
    merged = _concat_merged_layer([gdf_a, gdf_b])
    if merged is None or merged.empty:
        return gpd.GeoDataFrame(geometry=[], crs=metadata.get('crs'))
    stems = _stems_from_gdf(merged)
    stems = Quant.ensure_stem_ids(stems, prefix=f"{pid_a}_{pid_b}_border")
    stems = Quant.refresh_stems_direction_bulk(stems, config=config)
    stems, _dup = Vec.remove_duplicates_spatial(stems, config=config)
    stems = Vec.connect_stems(stems, config)
    stems = Quant.refresh_stems_direction_bulk(stems, config=config)
    out = []
    for stem in stems:
        owner_pid = _owner_partition_for_stem(stem, metadata)
        if owner_pid not in (pid_a, pid_b):
            continue
        stem.owner_partition_id = owner_pid
        stem.is_border_candidate = True
        out.append(stem)
    profile = {'crs': metadata.get('crs')}
    return stems_to_gdf(out, profile) if out else gpd.GeoDataFrame(geometry=[], crs=metadata.get('crs'))


def finalize_partitioned_stems(core_partition_outputs, border_outputs, output_gpkg, config, target_crs):
    import utils.Vectorization as Vec
    all_layers = []
    all_layers.extend([g for g in core_partition_outputs if g is not None and not g.empty])
    all_layers.extend([g for g in border_outputs if g is not None and not g.empty])
    stems_gdf = _concat_merged_layer(all_layers)
    if stems_gdf is None or stems_gdf.empty:
        return _write_merged(output_gpkg, [], [], [])
    stems_gdf = _ensure_crs(stems_gdf, target_crs)
    stems = _stems_from_gdf(stems_gdf)
    stems = Quant.ensure_stem_ids(stems, prefix='final')
    stems = Quant.refresh_stems_direction_bulk(stems, config=config)
    stems, _dup = Vec.remove_duplicates_spatial(stems, config=config)
    stems = [Quant.clean_diameter(stem) for stem in stems]
    stems = Quant.compute_stem_volumes(stems, config=config)
    stems_gdf, nodes_gdf, vectors_gdf = _stems_to_layer_gdfs(stems, target_crs)
    final_stems_layers = [stems_gdf] if stems_gdf is not None and not stems_gdf.empty else []
    final_nodes_layers = [nodes_gdf] if nodes_gdf is not None and not nodes_gdf.empty else []
    final_vectors_layers = [vectors_gdf] if vectors_gdf is not None and not vectors_gdf.empty else []
    return _write_merged(output_gpkg, final_stems_layers, final_nodes_layers, final_vectors_layers)


def _process_vector_partition_star(args):
    return process_vector_partition(*args)


def _stitch_partition_border_pair_star(args):
    return stitch_partition_border_pair(*args)


def merge_and_filter_tiled_results(
    work_dir: str,
    output_gpkg: str | None = None,
    edge_buffer_m: float = 1.0,
    config=None,
    stem_map_path: str | None = None,
    input_raster_path: str | None = None,
):
    merge_total_t0 = time.perf_counter()
    work_dir = os.path.abspath(work_dir)
    output_gpkg = _default_output_gpkg(
        work_dir=work_dir,
        output_gpkg=output_gpkg,
        input_raster_path=input_raster_path,
        stem_map_path=stem_map_path,
    )

    _remove_existing_output(output_gpkg)

    tiles = _detect_tiles(work_dir, output_gpkg)
    recursive_candidates = sorted(str(p) for p in Path(work_dir).rglob('*.gpkg'))
    print(
        f'MERGE DISCOVERY | root {work_dir} | gpkg_files {len(recursive_candidates)}',
        flush=True,
    )
    for candidate in recursive_candidates[:5]:
        print(f'MERGE INPUT | {candidate}', flush=True)
    if not tiles:
        raise FileNotFoundError(f"No .gpkg files found in: {work_dir}")

    if config is None:
        from classes.Config import Config
        config = Config()

    merged_stems = []
    target_crs = None
    tile_count = 0

    tile_scan_t0 = time.perf_counter()
    for prefix, gpkg_path, raster_path in tiles:
        out, target_crs = _process_tile(
            prefix,
            gpkg_path,
            raster_path,
            edge_buffer_m,
            target_crs,
        )
        if out is None:
            continue
        merged_stems.append(out)
        tile_count += 1
    tile_scan_s = time.perf_counter() - tile_scan_t0

    if not merged_stems:
        print("")
        print("MERGE SUMMARY")
        print("Tiles processed:       0")
        print("Total stems written:   0")
        print("Total nodes written:   0")
        print("Total vectors written: 0")
        print(f"No output GPKG created: {output_gpkg} (0 features written)")
        print(f"MERGE TIMINGS | tile_scan_s {tile_scan_s:.3f} | recon_s 0.000 | write_s 0.000 | total_s {time.perf_counter() - merge_total_t0:.3f}", flush=True)
        return output_gpkg

    merge_mode = str(getattr(config, 'tiled_merge_mode', 'edge_only') or 'edge_only').lower()

    if merge_mode == 'partitioned':
        all_tile_stems = _concat_merged_layer(merged_stems)
        partitions, metadata = build_vector_partitions(all_tile_stems, config)
        print(f"MERGE PARTITIONS | count {len(partitions)}", flush=True)

        partition_tasks = [(payload, config, metadata) for payload in partitions.values()]
        workers = max(1, int(getattr(config, 'vector_partition_workers', 1) or 1))
        partition_results = []
        if workers > 1 and len(partition_tasks) > 1:
            with mp.Pool(workers) as pool:
                for result in pool.imap_unordered(_process_vector_partition_star, partition_tasks):
                    partition_results.append(result)
        else:
            for task in partition_tasks:
                partition_results.append(_process_vector_partition_star(task))

        core_outputs = [r.get('core_stems_gdf') for r in partition_results if r]
        border_map = {r.get('partition_id'): r.get('border_stems_gdf') for r in partition_results if r}
        neighbor_pairs = _neighbor_partition_pairs(list(border_map.keys()))
        print(f"MERGE BORDER PAIRS | count {len(neighbor_pairs)}", flush=True)
        border_tasks = []
        for pid_a, pid_b in neighbor_pairs:
            gdf_a = border_map.get(pid_a)
            gdf_b = border_map.get(pid_b)
            if (gdf_a is None or gdf_a.empty) and (gdf_b is None or gdf_b.empty):
                continue
            border_tasks.append((pid_a, pid_b, gdf_a, gdf_b))

        border_outputs = []
        if workers > 1 and len(border_tasks) > 1:
            with mp.Pool(workers) as pool:
                args = [(task, config, metadata) for task in border_tasks]
                for result in pool.imap_unordered(_stitch_partition_border_pair_star, args):
                    border_outputs.append(result)
        else:
            for task in border_tasks:
                border_outputs.append(_stitch_partition_border_pair_star((task, config, metadata)))

        recon_t0 = time.perf_counter()
        written_gpkg = finalize_partitioned_stems(
            core_outputs,
            border_outputs,
            output_gpkg,
            config,
            target_crs or metadata.get('crs'),
        )
        recon_s = time.perf_counter() - recon_t0

        written_layers = []
        try:
            written_layers = list(fiona.listlayers(written_gpkg))
        except Exception as exc:
            print(
                f"MERGE VERIFY FAIL | file {written_gpkg} | {type(exc).__name__}: {exc}",
                flush=True,
            )

        final_stems = _read_gpkg_layer(written_gpkg, ['stems'])
        final_nodes = _read_gpkg_layer(written_gpkg, ['nodes'])
        final_vectors = _read_gpkg_layer(written_gpkg, ['vectors'])

        print("")
        print("MERGE SUMMARY")
        print(f"Tiles processed:       {tile_count}")
        print(f"Vector partitions:    {len(partitions)}")
        print(f"Border stitch pairs:  {len(border_tasks)}")
        print(f"Total stems written:   {0 if final_stems is None else len(final_stems)}")
        print(f"Total nodes written:   {0 if final_nodes is None else len(final_nodes)}")
        print(f"Total vectors written: {0 if final_vectors is None else len(final_vectors)}")
        print(f"Layers written:        {written_layers}")
        print(f"Output saved to: {written_gpkg}")
        print(
            f"MERGE TIMINGS | tile_scan_s {tile_scan_s:.3f} | recon_s {recon_s:.3f} | write_s 0.000 | total_s {time.perf_counter() - merge_total_t0:.3f}",
            flush=True,
        )
        return written_gpkg

    recon_t0 = time.perf_counter()
    final_stems_gdf, final_nodes_gdf, final_vectors_gdf = _reconstruct_edge_stems_for_tiled_merge(
        merged_stems=merged_stems,
        tiles=tiles,
        target_crs=target_crs,
        edge_buffer_m=edge_buffer_m,
        config=config,
        stem_map_path=stem_map_path,
    )
    recon_s = time.perf_counter() - recon_t0

    write_t0 = time.perf_counter()
    written_gpkg = _write_merged(
        output_gpkg,
        [final_stems_gdf] if final_stems_gdf is not None and not final_stems_gdf.empty else [],
        [final_nodes_gdf] if final_nodes_gdf is not None and not final_nodes_gdf.empty else [],
        [final_vectors_gdf] if final_vectors_gdf is not None and not final_vectors_gdf.empty else [],
    )
    write_s = time.perf_counter() - write_t0
    
    written_layers = []
    try:
        written_layers = list(fiona.listlayers(written_gpkg))
    except Exception as exc:
        print(
            f"MERGE VERIFY FAIL | file {written_gpkg} | {type(exc).__name__}: {exc}",
            flush=True,
        )

    final_stems = _read_gpkg_layer(written_gpkg, ['stems'])
    final_nodes = _read_gpkg_layer(written_gpkg, ['nodes'])
    final_vectors = _read_gpkg_layer(written_gpkg, ['vectors'])

    print("")
    print("MERGE SUMMARY")
    print(f"Tiles processed:       {tile_count}")
    print(f"Edge recon buffer m:   {edge_buffer_m}")
    print(f"Total stems written:   {0 if final_stems is None else len(final_stems)}")
    print(f"Total nodes written:   {0 if final_nodes is None else len(final_nodes)}")
    print(f"Total vectors written: {0 if final_vectors is None else len(final_vectors)}")
    print(f"Layers written:        {written_layers}")
    print(f"Output saved to: {written_gpkg}")
    print(f"MERGE TIMINGS | tile_scan_s {tile_scan_s:.3f} | recon_s {recon_s:.3f} | write_s {write_s:.3f} | total_s {time.perf_counter() - merge_total_t0:.3f}", flush=True)
    return written_gpkg
