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
            pred = src.read()
            pred = pred[0, :, :]
            profile = src.profile
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

    # Normalize to pyproj.CRS as reliably as possible
    try:
        crs = CRS.from_user_input(crs_in)
    except Exception:
        # common case: rasterio.crs.CRS -> convert via WKT
        try:
            if hasattr(crs_in, "to_wkt"):
                crs = CRS.from_wkt(crs_in.to_wkt())
            else:
                return str(crs_in)  # last resort (often "EPSG:25833")
        except Exception:
            return str(crs_in)

    # Optional: prefer EPSG if available (keeps things clean & portable)
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        pass
    return CRS.from_epsg(epsg) if epsg else crs


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
    # Match vectors_to_geojson_ CRS behavior
    crs_epsg = profile.get("crs", None)
    if isinstance(crs_epsg, int):
        crs_epsg = f"EPSG:{crs_epsg}"

    rows = []
    geoms = []

    for i in range(len(stems)):
        # Same iteration intent as your geojson function:
        # for j in range(len(stems[i].path.coords))
        # but be safe if vector/diameter lists are shorter.
        n_path = len(getattr(stems[i].path, "coords", []))
        vecs = getattr(stems[i], "vector", []) or []
        diams = getattr(stems[i], "segment_diameter_list", []) or []

        for j in range(n_path):
            # Ensure we don't crash if vector list is shorter than path coords
            if j >= len(vecs):
                continue

            v = vecs[j]

            # Expect v to be a shapely LineString-like with .coords[:]
            try:
                coords = list(v.coords[:])
            except Exception:
                continue

            # Build geometry
            try:
                geom = LineString(coords)
            except Exception:
                continue

            # Diameter (same indexing as your geojson function; missing -> None)
            d = diams[j] if j < len(diams) else None

            rows.append({
                "stem_id": i,
                "node": j,
                # IMPORTANT: GPKG attrs must be scalar -> store as JSON string
                "Vector": json.dumps(coords, ensure_ascii=False),
                "d": d,
            })
            geoms.append(geom)

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs_epsg)


def _safe_finalize_gpkg(tmp_path: str, final_path: str) -> str:
    """
    Finalize a temporary GeoPackage in a robust, cross-platform way.

    - Try atomic move (os.replace) when possible.
    - If cross-device/cross-mount (Win errno 18 or POSIX EXDEV), copy instead.
    - If target is locked, copy to *_new.gpkg (and if needed, *_new_<pid>.gpkg).
    - Always removes the temporary directory on success.
    Returns the path that was actually written.
    """
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

    # 1) Fast path: same filesystem -> atomic move
    try:
        os.replace(tmp_path, final_path)
        _cleanup()
        return final_path

    except PermissionError:
        # Destination locked: write alternative
        for alt in _alt_new_paths(dst):
            try:
                return _copy_to(alt)
            except PermissionError:
                continue
        raise

    except OSError as e:
        # Cross-device/mount move error:
        # Windows commonly reports errno 18; POSIX uses errno.EXDEV.
        exdev = getattr(errno, "EXDEV", 18)
        if e.errno in (18, exdev):
            # Copy to intended destination; if locked, fall back to *_new*
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
    gdf = gdf[gdf.geometry.notna()]
    try:
        gdf = gdf[~gdf.geometry.is_empty]
    except Exception:
        pass
    return gdf


def _normalize_dtypes(gdf):
    """
    Fiona can choke on pandas extension dtypes (string, Int64, boolean).
    Convert them to plain Python/object or numpy types.
    """
    if gdf is None or gdf.empty:
        return gdf

    for col in list(gdf.columns):
        if col == gdf.geometry.name:
            continue
        s = gdf[col]
        dt = str(s.dtype).lower()

        if "string" in dt:
            gdf[col] = s.astype(object).where(~s.isna(), None)
        elif dt in ("int64", "float64", "object"):
            continue
        elif "int" in dt:
            # nullable ints -> object (or float) is safest
            gdf[col] = s.astype("Int64").astype(object).where(~s.isna(), None)
        elif "bool" in dt:
            gdf[col] = s.astype(object).where(~s.isna(), None)
        else:
            # fallback: make it object
            gdf[col] = s.astype(object).where(~s.isna(), None)

    return gdf


def _write_layers_to_temp_gpkg(layers, crs, final_path: str) -> str:
    """
    layers: list of (layer_name, gdf) pairs
    Writes to a temp gpkg and returns the temp file path.

    Robustness goals:
      - Works with rasterio Profile/dict-like profiles upstream
      - Enforces CRS consistently across layers (prevents "unknown CRS" in QGIS)
      - Uses pyogrio if available, falls back to geopandas/fiona
      - Handles empty layers safely
    """
    tmp_dir = tempfile.mkdtemp(prefix="winmol_gpkg_")
    tmp_path = str(Path(tmp_dir) / (Path(final_path).stem + ".gpkg"))

    def _prep(name, gdf):
        """Normalize + drop bad geoms + enforce CRS."""
        if gdf is None:
            return None
        gdf = _drop_bad_geoms(_normalize_dtypes(gdf))
        if gdf is None or gdf.empty:
            return None

        # Enforce the CRS we derived from the raster profile.
        # Even if gdf.crs is set, we prefer the profile CRS
        if crs is not None:
            try:
                if (gdf.crs is None) or (gdf.crs != crs):
                    gdf = gdf.set_crs(crs, allow_override=True)
            except Exception:
                # Last-resort: try setting anyway
                try:
                    gdf = gdf.set_crs(crs, allow_override=True)
                except Exception:
                    pass

        return gdf

    # Collect prepared non-empty layers first
    prepared = []
    for name, gdf in layers:
        pgdf = _prep(name, gdf)
        if pgdf is not None:
            prepared.append((name, pgdf))

    # If *everything* is empty, still create a valid (empty) gpkg
    if not prepared:
        layer_name = layers[0][0] if layers else "layer"
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
        empty.to_file(tmp_path, layer=layer_name, driver="GPKG", index=False)
        return tmp_path

    # Prefer pyogrio if installed/available
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
        except Exception:
            # If pyogrio fails, fall back to geopandas/fiona
            pass

    # Fallback: geopandas/fiona
    first = True
    for name, gdf in prepared:
        if first:
            gdf.to_file(tmp_path, layer=name, driver="GPKG", index=False)
            first = False
        else:
            gdf.to_file(
                tmp_path, layer=name, driver="GPKG", index=False, mode="a")

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
    """Merge tiled WINMOL results into one GeoPackage (GPKG only)."""
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
