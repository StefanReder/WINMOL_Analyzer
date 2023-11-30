#!/usr/bin/env python

################################################################################
"""Imports"""

import os
import sys

from tensorflow import keras

from classes.Config import Config
from classes.Timer import Timer
from utils import IO
from utils import Prediction as Pred
from utils import Quantification as Quant
from utils import Skeletonization as Skel
from utils import Vectorization as Vec

if __name__ == '__main__':
    tt = Timer()
    tt.start()

    model_path = str(sys.argv[1])
    img_path = str(sys.argv[2])
    pred_dir = str(sys.argv[3])
    output_dir = str(sys.argv[4])

    config = Config()
    config.display()

    # Load the model from the HDF5 file
    model = keras.models.load_model(model_path, compile=False)

    # Summary of the loaded model
    model.summary()

    file_name = os.path.splitext(os.path.basename(img_path))[0]
    pred_name = pred_dir + 'pred_' + file_name + '.tiff'

    img, profile = IO.load_orthomosaic(img_path, config)

    pred, profile = Pred.predict_with_resampling_per_tile(
        img,
        profile,
        model,
        config
    )
    segments = Skel.find_segments(pred, config, profile)
    segments = Vec.restore_geoinformation(segments, config, profile)
    stems = Vec.build_stem_parts(segments)
    stems = Vec.connect_stems(stems, config)
    end_nodes = Vec.rebuild_endnodes_from_stems(stems)
    stems = Quant.quantify_stems(stems, pred, profile)

    IO.export_stem_map(pred, profile, pred_dir, file_name)
    IO.stems_to_geojson(stems, output_dir + file_name)

    tt.stop()
