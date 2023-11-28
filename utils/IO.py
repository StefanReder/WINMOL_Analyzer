#!/usr/bin/env python

##################################################################################
'''Imports'''

import numpy as np
import rasterio
from rasterio.enums import Resampling
import json
import os


##################################################################################
'''File operations'''

def load_orthomosaic(path,config):
    with rasterio.open(path) as src:
        img = src.read(list(range(1,config.n_Channels+1))).transpose(1, 2, 0)
        img = (img / 255).astype(np.float32)
        return img, src.profile

def load_orthomosaic_with_resampling(path, config):
    with rasterio.open(path) as src:
        scale_factor_x =src.res[0]/(config.tile_size/config.IMG_width)
        scale_factor_y =src.res[1]/(config.tile_size/config.IMG_width)
        # resample data to target shape
        img = src.read(
            list(range(1,config.n_Channels+1)),
            out_shape=(
                config.n_Channels,
                int(src.height * scale_factor_y),
                int(src.width * scale_factor_x)
            ),
            resampling=Resampling.bilinear
        )
        img=img[0:3,:,:].transpose(1, 2, 0)
        # scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / img.shape[-2]),
            (src.height / img.shape[-3])
        )
        img = (img / 255).astype(np.float32)
        profile = src.profile.copy()
        profile['transform']=transform
    return img, profile

def load_stem_map(path):
    if path.endswith('.tif')or path.endswith('.tiff'):
        print("#######################################################")
        print("#######################################################")
        print("")
        print(path)
        print("")
        with rasterio.open(path) as src:
           # crs=src.crs
            pred=src.read()
            pred=pred[0,:,:]
            profile=src.profile
       #     px_size=(pred.bounds.right-pred.bounds.left)/pred.width
       #     px_size=abs(src.profile['transform'][0])
      #      bounds=src.bounds
   #         pred=np.pad(pred,((padding,padding),(padding,padding)),'constant', constant_values=False)
        return pred, profile

def export_stem_map(pred, profile, pred_dir, pred_name):
    profile.update(dtype=rasterio.float32, count=1)
    height, width = pred.shape
    profile['width']=width
    profile['height']=height
    with rasterio.open(os.path.join(pred_dir, f'pred_{pred_name}.tiff'), 'w', **profile) as dst:
            dst.write(pred.astype(rasterio.float32), 1)

def get_bounds_from_profile(profile):
    left=profile['transform'][2]
    right=profile['transform'][2]+profile['transform'][0]*profile['width']
    bot=profile['transform'][5]+profile['transform'][4]*profile['height']
    top=profile['transform'][5]
    return rasterio.coords.BoundingBox(left, bot, right, top)

def stems_to_geojson_(stems):
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type' :"Feature",
                'geometry':{
                    'type': 'LineString',
                    'coordinates': stems[i].path.coords[:]
                },
                'properties': {
                    'id': i,
                    'start': stems[i].start.coords[:],
                    'stop': stems[i].stop.coords[:],
                    'path': stems[i].path.coords[:],
    #                'Vector':stems[i].vector,
                    'd':   stems[i].d,
                    'l': stems[i].l,
                    'v':stems[i].v,
                    'Length':stems[i].Length,
                    'Volume':stems[i].Volume
                },
            }
            for i in range(len(stems))
        ],
    }

def nodes_to_geojson_(stems):
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type' :"Feature",
                'geometry':{
                    'type': 'Point',
                    'coordinates': stems[i].path.coords[j]
                },
                'properties': {
                    'stem_id': i,
                    'node': j,
                    'Vector':stems[i].vector[j].coords[:],
                    'd':   stems[i].d[j],
                },
            }for i in range(len(stems))
        for j in range(len(stems[i].path.coords))
        ],
    }


def vectors_to_geojson_(stems):
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type' :"Feature",
                'geometry':{
                    'type': 'LineString',
                    'coordinates': stems[i].vector[j].coords[:]
                },
                'properties': {
                    'stem_id': i,
                    'node': j,
                    'Vector':stems[i].vector[j].coords[:],
                    'd':   stems[i].d[j],
                },
            }for i in range(len(stems))
        for j in range(len(stems[i].path.coords))
        ],
    }



def stems_to_geojson(stems, path):
    print("#######################################################")
    print("Export to GeoJSON")

    fc_stems=stems_to_geojson_(stems)
    fc_nodes=nodes_to_geojson_(stems)
    fc_vectors=vectors_to_geojson_(stems)
    s_path= path+"_stems.geojson"
    n_path= path+"_nodes.geojson"
    v_path= path+"_vectors.geojson"

    with open(s_path, 'w') as out:
        json.dump(fc_stems, out)
    print(f'Wrote {s_path}')

    with open(n_path, 'w') as out:
        json.dump(fc_nodes, out)
    print(f'Wrote {n_path}')

    with open(v_path, 'w') as out:
        json.dump(fc_vectors, out)
    print(f'Wrote {v_path}')

    print("#######################################################")
    print("")

def save_image(data, outputname, size=(15, 15), dpi=300):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(outputname, dpi=dpi)



