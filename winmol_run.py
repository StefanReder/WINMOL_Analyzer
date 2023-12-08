#!/usr/bin/env python

################################################################################
"""Imports"""




import os
import sys
from tensorflow import keras

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))

sys.path.append(parent_path)
from classes.Config import Config
from classes.Timer import Timer
from utils import IO
from utils import Prediction as Pred
from utils import Quantification as Quant
from utils import Skeletonization as Skel
from utils import Vectorization as Vec


class ImageProcessing:
    def __init__(self, model_path, img_path, stem_dir, trees_dir, process_type):
        self.model_path = model_path
        self.img_path = img_path
        self.stem_dir = stem_dir
        self.trees_dir = trees_dir
        self.process_type = process_type
        self.config = Config()

    def stem_processing(self):
        model = keras.models.load_model(self.model_path, compile=False)
        print("\nLoaded Model Summary:")
        model.summary()

        file_name = os.path.splitext(os.path.basename(self.img_path))[0]

        print("\nLoading Orthomosaic Image:")
        img, profile = IO.load_orthomosaic(self.img_path, self.config)

        print("\nPerforming Prediction with Resampling:")
        pred, profile = Pred.predict_with_resampling_per_tile(img, profile, model, self.config)

        print("\nExporting Predicted Stem Map:")
        IO.export_stem_map(pred, profile, self.stem_dir, file_name)
        return pred, profile, file_name

    def trees_processing(self, pred, profile):
        segments = Skel.find_segments(pred, self.config, profile)
        segments = Vec.restore_geoinformation(segments, self.config, profile)
        stems = Vec.build_stem_parts(segments)
        stems = Vec.connect_stems(stems, self.config)
        end_nodes = Vec.rebuild_endnodes_from_stems(stems)
        stems = Quant.quantify_stems(stems, pred, profile)
        return stems

    def nodes_processing(self, stems):
        IO.stems_to_geojson(stems, os.path.join(self.trees_dir, "nodes_output.geojson"))



    def display_starting_text(self):
        print("Command-line arguments:")
        print("Model Path:", self.model_path)
        print("Image Path:", self.img_path)
        print("Semantic Stem Map Directory:", self.stem_dir)
        print("Process type:", process_type)
        if self.trees_dir:
            print("Detected Wind-thrown Trees Directory:", self.trees_dir)
        print("\nConfiguration Settings:")
        self.config.display()

    def main(self):
        if self.process_type == "Stems":
            self.stem_processing()
        elif self.process_type == "Trees":
            pred, profile, _ = self.stem_processing()
            self.trees_processing(pred, profile)
        elif self.process_type == "Nodes":
            pred, profile, file_name = self.stem_processing()
            stems = self.trees_processing(pred, profile)
            self.nodes_processing(stems)


if __name__ == '__main__':
    # Create a timer to measure the execution time of the script
    tt = Timer()
    tt.start()
    # Extract command-line arguments
    model_path = str(sys.argv[1])
    img_path = str(sys.argv[2])
    stem_dir = str(sys.argv[3])
    trees_dir = str(sys.argv[4])
    process_type = str(sys.argv[5])

    # Create an instance of the ImageProcessing class and run the main method
    image_processor = ImageProcessing(
        model_path,
        img_path,
        stem_dir,
        trees_dir,
        process_type
    )
    image_processor.main()

    # Stop the timer and display the elapsed time
    tt.stop()
