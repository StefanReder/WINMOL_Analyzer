##################################################################################
'''Imports'''
#import os
#import sys
#import time
#import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg16 import preprocess_input
#import rasterio
#from rasterio.windows import Window
#import cv2


class Config(object):
    #Coniguration for the semantic segmentation
    tile_size = 15
    IMG_width=512
    IMG_height=512
    IMG_bit=8
    n_Channels=3
    num_classes=1
    overlapp_pred=8
    #Configuration for the stem vectorization
    min_length = 2.0
    max_distance=8
    max_tree_height=32
    tolerance_angle=7
    
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
    #    self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
    #    if self.IMAGE_RESIZE_MODE == "crop":
     #       self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
     #           self.IMAGE_CHANNEL_COUNT])
      #  else:
       #     self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
        #        self.IMAGE_CHANNEL_COUNT])



    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        # for a in dir(self):
        #     if not a.startswith("__") and not callable(getattr(self, a)):
        #         print("{:30} {}".format(a, getattr(self, a)))
        print("\n")