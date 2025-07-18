#!/usr/bin/env python
import os
import sys
import subprocess
import tensorflow as tf

from classes.Config import Config
from classes.Timer import Timer
from utils import IO
from utils import Prediction as Pred
from utils import Skeletonization as Skel
from utils import Vectorization as Vec
from utils import Quantification as Quant
print("imports finished")


class ImageProcessing:
    def __init__(
            self,
            model_path,
            uav_path,
            stem_path,
            trees_path,
            nodes_path,
            process_type
    ):
        print("Initialization")
        self.model_path = model_path
        self.uav_path = uav_path
        self.stem_path = stem_path
        self.trees_path = trees_path
        self.nodes_path = nodes_path
        self.process_type = process_type
        self.config = Config()

    def stem_processing(self):
        print("\nLoading Model...")
        model = IO.load_model_from_path(self.model_path)

        # Loading Orthomosaic Image:
        print("\nLoading Orthomosaic Image...")
        img, profile = IO.load_orthomosaic(self.uav_path, self.config)

        print("\nPerforming Prediction with Resampling...")
        pred, profile = Pred.predict_with_resampling_per_tile(
            img, profile, model, self.config
        )

        print("\nExporting Predicted Stem Map...")
        stem_file_name = os.path.splitext(os.path.basename(self.stem_path))[0]
        stem_dir = os.path.dirname(self.stem_path)
        # exporting as tiff
        IO.export_stem_map(pred, profile, stem_dir, stem_file_name)
        return pred, profile

    def trees_processing(self, pred, profile):
        print("\nFinding Stem Segments...")
        segments = Skel.find_segments(pred, self.config, profile)
        print("\nRestoring Geoinformation...")
        segments = Vec.restore_geoinformation(segments, self.config, profile)
        print("\nBuilding Stem Parts...")
        stems = Vec.build_stem_parts(segments)
        print("\nConnecting Stem Parts...")
        stems = Vec.connect_stems(stems, self.config)
        print("\nRebuilding End Nodes...")
        Vec.rebuild_endnodes_from_stems(stems)
        print("\nQuantifying Stems...")
        stems = Quant.quantify_stems(stems, pred, profile)
        # exporting as geojson
        IO.stems_to_geojson(stems, profile, self.trees_path)
        return stems

    def nodes_processing(self, stems, profile):
        IO.vector_to_geojson(stems, profile, self.nodes_path)
        IO.nodes_to_geojson(stems, profile, self.nodes_path)

    def check_DL_env(self):
        # Check if NVIDIA GPU is available and available for processing
        def get_nvidia_driver_version():
            try:
                result = subprocess.run(["nvidia-smi",
                                         "--query-gpu=driver_version",
                                         "--format=csv,noheader"],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)
                if result.returncode == 0:
                    print(f"NVIDIA GPU Driver Version: {result.stdout.strip()}")
                else:
                    print("Failed to retrieve NVIDIA driver version.")
            except FileNotFoundError:
                print("No NVIDIA GPU available or drivers not installed.")

        get_nvidia_driver_version()
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            cuda_version = tf.sysconfig.get_build_info().get(
                'cuda_version', 'Unknown')
            cudnn_version = tf.sysconfig.get_build_info().get(
                'cudnn_version', 'Unknown')
            print(f"CUDA is available: {cuda_version}")
            print(f"cuDNN version: {cudnn_version}")
            print("Num GPUs for CUDA processing:", len(physical_devices))
            print("Tensorflow version:", tf.__version__)
            print("Keras version:", tf.keras.__version__)

        except Exception as e:
            print("Tensorflow error: ", e)

    def display_starting_text(self):
        print("Check CUDA environment")
        self.check_DL_env()
        print("Command-line arguments:")
        print("Model Path:", self.model_path)
        print("Image Path:", self.uav_path)
        print("Semantic Stem Map Path:", self.stem_path)
        print("Process type:", process_type)
        if self.trees_path:
            print("Detected Wind-thrown Trees Path:", self.trees_path)
        if self.nodes_path:
            print("Detected Wind-thrown Nodes Path:", self.nodes_path)
        # print configuration settings
        self.config.display()

    def main(self):
        if self.process_type == "Stems":  # 34 lines
            self.stem_processing()
        elif self.process_type == "Trees":  # 118 lines
            pred, profile = self.stem_processing()
            self.trees_processing(pred, profile)
        elif self.process_type == "Nodes":  # 125 lines
            pred, profile = self.stem_processing()
            stems = self.trees_processing(pred, profile)
            self.nodes_processing(stems, profile)


if __name__ == '__main__':
    # Create a timer to measure the execution time of the script
    tt = Timer()
    tt.start()
    print("Start timer")
    # Extract command-line arguments
    model_path = str(sys.argv[1])
    uav_path = str(sys.argv[2])
    stem_path = str(sys.argv[3])
    trees_path = str(sys.argv[4])
    nodes_path = str(sys.argv[5])
    process_type = str(sys.argv[6])

    # Create an instance of the ImageProcessing class and run the main method
    image_processor = ImageProcessing(
        model_path,
        uav_path,
        stem_path,
        trees_path,
        nodes_path,
        process_type
    )
    image_processor.display_starting_text()
    image_processor.main()

    # Stop the timer and display the elapsed time
    print("Stop timer")
    tt.stop()
