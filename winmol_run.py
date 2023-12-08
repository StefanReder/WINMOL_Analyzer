#!/usr/bin/env python
import os
import sys

from tensorflow import keras

from classes.Config import Config
from classes.Timer import Timer
from utils import IO
from utils import Prediction as Pred

if __name__ == '__main__':

    # Create a timer to measure the execution time of the script
    tt = Timer()
    tt.start()

    # Extract command-line arguments
    model_path = str(sys.argv[1])
    img_path = str(sys.argv[2])
    pred_dir = str(sys.argv[3])
    output_dir = str(sys.argv[4])

    print("Command-line arguments:")
    print("Model Path:", model_path)
    print("Image Path:", img_path)
    print("Semantic Stem Map Directory:", pred_dir)
    if output_dir:
        print("Detected Wind-thrown Trees Directory:", output_dir)

    # Create a Config instance and display its settings
    config = Config()
    print("\nConfiguration Settings:")
    config.display()

    model = keras.models.load_model(model_path, compile=False)

    # Display a summary of the loaded model architecture
    print("\nLoaded Model Summary:")
    model.summary()

    # Extract the base name of the input image file
    file_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load the input orthomosaic image and its profile using IO module
    print("\nLoading Orthomosaic Image:")
    img, profile = IO.load_orthomosaic(img_path, config)

    # Perform prediction on the input image with resampling
    print("\nPerforming Prediction with Resampling:")
    pred, profile = Pred.predict_with_resampling_per_tile(
        img,
        profile,
        model,
        config
    )

    # Export the predicted stem map and stems information to GeoJSON
    # (first checkbox)
    print("\nExporting Predicted Stem Map:")
    IO.export_stem_map(pred, profile, pred_dir, file_name)

    # Stop the timer and display the elapsed time
    tt.stop()
