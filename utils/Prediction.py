#!/usr/bin/env python

##################################################################################
'''Imports'''
import numpy as np
import tensorflow as tf
from rasterio import Affine
from skimage.transform import resize
from standalone.WINMOL_Analyzer import Timer



##################################################################################
'''Prediction of the semantic stem map with U-Net'''

def predict(img, model, config):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of semantic stem map")

    xtiles = int(np.floor(img.shape[1] / (config.IMG_width - config.overlap_pred))) + 1
    ytiles = int(np.floor(img.shape[0] / (config.IMG_width - config.overlap_pred))) + 1

    #padding to full tiles
    img_pad = np.full((ytiles * (config.IMG_width - config.overlap_pred) + config.overlap_pred,
                       xtiles * (config.IMG_width - config.overlap_pred) + config.overlap_pred, config.n_Channels),
                      fill_value=0, dtype=np.float64)
    img_pad[0:img.shape[0], 0:img.shape[1], ] = img

    IMG_width_ = config.IMG_width - config.overlap_pred
    prediction = np.empty((img_pad.shape[0],img_pad.shape[1]))
    mask= np.where(img[:,:,0:3]==(0,0,0),False,True)[:,:,0]


    for i in range(ytiles):
        x = i * (config.IMG_width - config.overlap_pred)
        for j in range(xtiles):
            y = j * (config.IMG_width - config.overlap_pred)
            tile = img_pad[x:x + config.IMG_width, y:y + config.IMG_width, 0:3]
            tile = tf.convert_to_tensor(tile, dtype =np.float32)
            tile = tf.reshape(tile,shape=[1,config.IMG_width,config.IMG_width,3])
            pred = model.predict_on_batch(tile)
            pred2= pred[0, (config.overlap_pred // 2):(config.IMG_width - config.overlap_pred // 2),
                   (config.overlap_pred // 2):(config.IMG_width - config.overlap_pred // 2), 0]
            prediction[(config.overlap_pred // 2 + (i) * (IMG_width_)):((config.IMG_width - config.overlap_pred // 2) + (i) * +IMG_width_),
            (config.overlap_pred // 2 + (j) * (IMG_width_)):((config.IMG_width - config.overlap_pred // 2) + (j) * +IMG_width_)]=pred2

    prediction=prediction[0:img.shape[0],0:img.shape[1]]
   # prediction=np.where(np.all(img!=-255),prediction,-255)
    prediction = prediction * mask
    print(xtiles*ytiles," tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return prediction

def predict_with_resampling_per_tile(img, profile ,model, config):
    t = Timer()
    t.start()
    print("#######################################################")
    print("Prediction of the semantic stem map")
    print("Resampling tiles while analyzing")


    px_per_tile_x = int(np.floor(config.tile_size / abs(profile['transform'][0])))
    px_per_tile_y = int(np.floor(config.tile_size / abs(profile['transform'][4])))
    overlapp_img_x=int(np.floor(config.overlap_pred * px_per_tile_x / config.IMG_width))
    overlapp_img_y=int(np.floor(config.overlap_pred * px_per_tile_y / config.IMG_width))
    xtiles = int(np.floor(img.shape[1] / (px_per_tile_x - overlapp_img_x))) + 1
    ytiles = int(np.floor(img.shape[0] / (px_per_tile_y - overlapp_img_y))) + 1

    #padding to full tiles
    img_pad = np.full((ytiles * (px_per_tile_y - overlapp_img_y) + overlapp_img_y, xtiles * (px_per_tile_x - overlapp_img_x) + overlapp_img_x,
                        config.n_Channels),fill_value=0, dtype=np.float64)
    img_pad[0:img.shape[0], 0:img.shape[1], ] = img

    IMG_width_ = config.IMG_width - config.overlap_pred
    prediction = np.empty((ytiles * IMG_width_ + config.overlap_pred, xtiles * IMG_width_ + config.overlap_pred))
    mask = np.where(img_pad[:,:,0:3]==(0,0,0),False,True)[:,:,0]
    mask =  resize(mask, prediction.shape)

    for i in range(ytiles):
        x = i * (px_per_tile_y - overlapp_img_y)
        for j in range(xtiles):
            y = j * (px_per_tile_x - overlapp_img_x)
            tile = img_pad[x:x + px_per_tile_x-1, y:y + px_per_tile_y-1, 0:3]
            tile = tf.convert_to_tensor(tile, dtype =np.float32)
            tile = tf.image.resize(tile, size= [config.IMG_width, config.IMG_height],method="bicubic", antialias=False)
            tile = tf.reshape(tile,shape=[1,config.IMG_width,config.IMG_width,3])

            pred = model.predict_on_batch(tile)
            pred2= pred[0, (config.overlap_pred // 2):(config.IMG_width - config.overlap_pred // 2),
                   (config.overlap_pred // 2):(config.IMG_width - config.overlap_pred // 2), 0]
            prediction[(config.overlap_pred // 2 + (i) * (IMG_width_)):((config.IMG_width - config.overlap_pred // 2) + (i) * +IMG_width_),
            (config.overlap_pred // 2 + (j) * (IMG_width_)):((config.IMG_width - config.overlap_pred // 2) + (j) * +IMG_width_)]=pred2
    prediction=prediction*mask

    profile['transform']=Affine(profile['transform'][0]*px_per_tile_x/config.IMG_width, 0.0, profile['transform'][2], 0.0,
                                profile['transform'][4]*px_per_tile_y/config.IMG_width, profile['transform'][5])
    print(xtiles*ytiles," tiles analyzed")
    t.stop()
    print("#######################################################")
    print("")
    return prediction, profile

