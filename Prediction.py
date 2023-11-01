#!/usr/bin/env python

##################################################################################
'''Imports'''
import numpy as np
import tensorflow as tf
import cv2 as cv2
import rasterio
from rasterio import Affine

from WINMOL_Analyzer import Timer


################################################################################## 
'''Prediction of the semantic stem map with U-Net'''

def predict(img, model, config):
    t = Timer()
    t.start()
    print("#######################################################")   
    print("Prediction of semantic stem map")   
    
    xtiles = int(np.floor(img.shape[1] / (config.IMG_width - config.overlapp_pred))) + 1
    ytiles = int(np.floor(img.shape[0] / (config.IMG_width - config.overlapp_pred))) + 1

    #padding to full tiles
    img_pad = np.full((ytiles * (config.IMG_width - config.overlapp_pred) + config.overlapp_pred, 
                   xtiles * (config.IMG_width - config.overlapp_pred) + config.overlapp_pred, config.n_Channels),
                   fill_value=-255, dtype=np.float64)
    img_pad[0:img.shape[0], 0:img.shape[1], ] = img
    
    
    IMG_width_ = config.IMG_width - config.overlapp_pred
    prediction = np.empty((img_pad.shape[0],img_pad.shape[1]))

    for i in range(ytiles):
        x = i * (config.IMG_width - config.overlapp_pred) 
        for j in range(xtiles):
            y = j * (config.IMG_width - config.overlapp_pred) 
            tile = img_pad[x:x + config.IMG_width, y:y + config.IMG_width, 0:3]
            tile = tile.reshape(1, config.IMG_width, config.IMG_width, 3).astype(np.float32)
            pred = model.predict(tile)
            pred2=pred[0,(config.overlapp_pred//2):(config.IMG_width-config.overlapp_pred//2),
                       (config.overlapp_pred//2):(config.IMG_width-config.overlapp_pred//2),0]
            prediction[(config.overlapp_pred//2+(i)*(IMG_width_)):((config.IMG_width-config.overlapp_pred//2)+(i)*+IMG_width_),
                       (config.overlapp_pred//2+(j)*(IMG_width_)):((config.IMG_width-config.overlapp_pred//2)+(j)*+IMG_width_)]=pred2

    prediction=prediction[0:img.shape[0],0:img.shape[1]]
    prediction=np.where(np.all(img!=-255),prediction,-255)
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
    overlapp_img_x=int(np.floor(config.overlapp_pred*px_per_tile_x/config.IMG_width))
    overlapp_img_y=int(np.floor(config.overlapp_pred*px_per_tile_y/config.IMG_width))
    xtiles = int(np.floor(img.shape[1] / (px_per_tile_x - overlapp_img_x))) + 1
    ytiles = int(np.floor(img.shape[0] / (px_per_tile_y - overlapp_img_y))) + 1   

    #padding to full tiles
    img_pad = np.full((ytiles * (px_per_tile_y - overlapp_img_y) + overlapp_img_y, xtiles * (px_per_tile_x - overlapp_img_x) + overlapp_img_x,
                        config.n_Channels),fill_value=-255, dtype=np.float64)
    img_pad[0:img.shape[0], 0:img.shape[1], ] = img
    
    IMG_width_ = config.IMG_width - config.overlapp_pred
    prediction = np.empty((ytiles * IMG_width_ + config.overlapp_pred, xtiles * IMG_width_ + config.overlapp_pred))
   
    for i in range(ytiles):
        x = i * (px_per_tile_y - overlapp_img_y) 
        for j in range(xtiles):
            y = j * (px_per_tile_x - overlapp_img_x) 
            tile = img_pad[x:x + px_per_tile_x-1, y:y + px_per_tile_y-1, 0:3]
            tile = cv2.resize(tile, (config.IMG_width, config.IMG_width), interpolation=cv2.INTER_CUBIC)
            tile = tile.reshape(1, config.IMG_width, config.IMG_width, 3).astype(np.float32)
            pred = model.predict(tile)
            pred2=pred[0,(config.overlapp_pred//2):(config.IMG_width-config.overlapp_pred//2),
                       (config.overlapp_pred//2):(config.IMG_width-config.overlapp_pred//2),0]
            prediction[(config.overlapp_pred//2+(i)*(IMG_width_)):((config.IMG_width-config.overlapp_pred//2)+(i)*+IMG_width_),
                       (config.overlapp_pred//2+(j)*(IMG_width_)):((config.IMG_width-config.overlapp_pred//2)+(j)*+IMG_width_)]=pred2
    profile['transform']=Affine(profile['transform'][0]*px_per_tile_x/config.IMG_width, 0.0, profile['transform'][2], 0.0, profile['transform'][4]*px_per_tile_y/config.IMG_width, profile['transform'][5])   
    print(xtiles*ytiles," tiles analyzed")
    t.stop()
    print("#######################################################")   
    print("")  
    return prediction, profile
    

