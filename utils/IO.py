#!/usr/bin/env python

###############################################################################
"""Imports"""

import json
import os
import glob
import tempfile
import shutil
import errno
import numpy as np
import rasterio
import geopandas as gpd
import fiona
import pandas as pd
try:
    import pyogrio
    _HAVE_PYOGRIO = True
except Exception:
    pyogrio = None
    _HAVE_PYOGRIO = False
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point, box
from collections.abc import Mapping
from pyproj import CRS
from pathlib import Path


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
    transform, compress: str | None = 'DEFLATE'):

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
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
               list[int] | None = None, boundless: bool = True, fill_value=0):
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
        dst.write(pred_tile.astype(np.float32), 1)
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
    )
    with rasterio.open(tmp_path, 'w', **safe_profile) as dst:
        dst.write(pred.astype(rasterio.float32), 1)
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

    for i in range(len(stems)):
        n_path = len(getattr(stems[i].path, "coords", []))
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
                "stem_id": int(i),
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

        gdf[col] = s.map(lambda v: None if pd.isna(v) \
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
        geom_types = [g.geom_type for g in gdf.geometry \
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
    mode = "a" if append else "w"

    crs_wkt = None
    if crs is not None:
        try:
            crs_wkt = CRS.from_user_input(crs).to_wkt()
        except Exception:
            try:
                crs_wkt = crs.to_wkt()
            except Exception:
                crs_wkt = None

    print(f"Fiona schema for layer '{layer_name}': {schema}")

    kwargs = {
        "driver": "GPKG",
        "layer": layer_name,
        "schema": schema,
    }
    if crs_wkt is not None:
        kwargs["crs_wkt"] = crs_wkt

    # Always use "w" for creating a layer
    with fiona.open(path, mode=mode, **kwargs) as dst:
        dst.writerecords(_feature_records(gdf, schema))


def _write_layers_to_temp_gpkg(layers, crs, final_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="winmol_gpkg_")
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


def _read_gpkg_layer(gpkg_path, layer_names):
    for ln in layer_names:
        try:
            return gpd.read_file(gpkg_path, layer=ln)
        except Exception:
            continue
    return gpd.GeoDataFrame(geometry=[])


def _read_tile_gpkg(gpkg_path):
    stems = _read_gpkg_layer(gpkg_path, ["stems", "stem", "trees", "tree"])
    nodes = _read_gpkg_layer(gpkg_path, ["nodes", "node"])
    vectors = _read_gpkg_layer(
        gpkg_path,
        ["vectors", "vector", "segments", "segment"],
    )
    return stems, nodes, vectors


def _ensure_crs(gdf, target_crs):
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()
    if target_crs is None:
        return gdf
    if gdf.crs is None:
        gdf = gdf.copy()
        gdf.set_crs(target_crs, inplace=True)
        return gdf
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _raster_filter_geom(raster_path, edge_buffer_m):
    if not raster_path:
        return None, None
    try:
        with rasterio.open(raster_path) as src:
            geom = box(*src.bounds)
            return geom.buffer(-abs(edge_buffer_m)), src.crs
    except Exception:
        return None, None


def _detect_tiles(work_dir, output_gpkg):
    rasters = [
        f
        for f in os.listdir(work_dir)
        if f.lower().endswith((".tif", ".tiff")) and "_roi_stem_map" in f
    ]

    tiles = []

    if rasters:
        for rf in sorted(rasters):
            prefix = rf
            for ext in ("_roi_stem_map.tif", "_roi_stem_map.tiff"):
                if rf.endswith(ext):
                    prefix = rf.replace(ext, "")
                    break

            gpkg = sorted(glob.glob(os.path.join(work_dir, f"{prefix}*.gpkg")))
            if not gpkg:
                continue

            tiles.append((prefix, gpkg[0], os.path.join(work_dir, rf)))

        return tiles

    for gpkg in sorted(glob.glob(os.path.join(work_dir, "*.gpkg"))):
        if output_gpkg:
            if os.path.abspath(gpkg) == os.path.abspath(output_gpkg):
                continue
        prefix = os.path.splitext(os.path.basename(gpkg))[0]
        tiles.append((prefix, gpkg, None))

    return tiles


def _default_output_gpkg(work_dir, output_gpkg):
    if output_gpkg is not None:
        return output_gpkg
    folder = os.path.basename(os.path.normpath(work_dir))
    return os.path.join(work_dir, f"{folder}_merged_data.gpkg")


def _globalize_stems(stems, tile_id, filter_geom):
    id_col = _pick_id_col(stems)
    if not id_col:
        return gpd.GeoDataFrame(), set()

    stems = stems.copy()
    stems["_stem_id_local"] = stems[id_col].astype(str)
    stems["stem_id"] = tile_id + "_" + stems["_stem_id_local"]
    stems["tile_id"] = tile_id

    if filter_geom is not None:
        stems = stems[stems.intersects(filter_geom)].copy()

    kept_local = set(stems["_stem_id_local"].tolist())
    return stems, kept_local


def _select_child(gdf, tile_id, kept_local):
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()

    id_col = _pick_id_col(gdf)
    if not id_col:
        return gpd.GeoDataFrame()

    out = gdf[gdf[id_col].astype(str).isin(kept_local)].copy()
    if out.empty:
        return gpd.GeoDataFrame()

    out["_stem_id_local"] = out[id_col].astype(str)
    out["stem_id"] = tile_id + "_" + out["_stem_id_local"]
    out["tile_id"] = tile_id
    return out


def _process_tile(prefix, gpkg_path, raster_path, edge_buffer_m, target_crs):
    tile_id = _tile_id_from_prefix(prefix)

    filter_geom, raster_crs = _raster_filter_geom(raster_path, edge_buffer_m)

    stems, nodes, vectors = _read_tile_gpkg(gpkg_path)
    if stems is None or stems.empty:
        return None, target_crs

    if target_crs is None:
        target_crs = stems.crs or raster_crs

    stems = _ensure_crs(stems, target_crs)
    nodes = _ensure_crs(nodes, target_crs)
    vectors = _ensure_crs(vectors, target_crs)

    stems, kept_local = _globalize_stems(stems, tile_id, filter_geom)
    if stems.empty:
        return None, target_crs

    nodes_sel = _select_child(nodes, tile_id, kept_local)
    vectors_sel = _select_child(vectors, tile_id, kept_local)

    counts = (len(stems), len(nodes_sel), len(vectors_sel))
    return (stems, nodes_sel, vectors_sel, counts), target_crs


def _write_merged(output_gpkg, merged_stems, merged_nodes, merged_vectors):
    if merged_stems:
        gpd.GeoDataFrame(
            pd.concat(merged_stems, ignore_index=True),
        ).to_file(output_gpkg, layer="stems", driver="GPKG")

    if merged_nodes:
        gpd.GeoDataFrame(
            pd.concat(merged_nodes, ignore_index=True),
        ).to_file(output_gpkg, layer="nodes", driver="GPKG")

    if merged_vectors:
        gpd.GeoDataFrame(
            pd.concat(merged_vectors, ignore_index=True),
        ).to_file(output_gpkg, layer="vectors", driver="GPKG")


def merge_and_filter_tiled_results(
    work_dir: str,
    output_gpkg: str | None = None,
    edge_buffer_m: float = 1.0,
):
    work_dir = os.path.abspath(work_dir)
    output_gpkg = _default_output_gpkg(work_dir, output_gpkg)

    if os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    tiles = _detect_tiles(work_dir, output_gpkg)
    if not tiles:
        raise FileNotFoundError(f"No .gpkg files found in: {work_dir}")

    merged_stems = []
    merged_nodes = []
    merged_vectors = []

    target_crs = None
    tile_count = 0
    total_stems = 0
    total_nodes = 0
    total_vectors = 0

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

        stems, nodes, vectors, (n_s, n_n, n_v) = out
        merged_stems.append(stems)
        if not nodes.empty:
            merged_nodes.append(nodes)
        if not vectors.empty:
            merged_vectors.append(vectors)

        tile_count += 1
        total_stems += n_s
        total_nodes += n_n
        total_vectors += n_v

    _write_merged(output_gpkg, merged_stems, merged_nodes, merged_vectors)

    print("")
    print("MERGE SUMMARY")
    print(f"Tiles processed:       {tile_count}")
    print(f"Total stems written:   {total_stems}")
    print(f"Total nodes written:   {total_nodes}")
    print(f"Total vectors written: {total_vectors}")
    print(f"Output saved to: {output_gpkg}")
    return output_gpkg
