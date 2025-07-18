import csv
import os
import subprocess

# Fixed paths for models
MODEL_PATHS = {
    'spruce': './standalone/model/model_UNet_SpecDS_Spruce_512_2023-02-27_061925.hdf5',
    'beech': './standalone/model/model_UNet_SpecDS_Beech_512_2023-02-28_042751.hdf5e',
    'general': './standalone/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5'
}

def process_orthomosaics(file_list_path, output_folder):
    # Read the CSV file with the list of files
    with open(file_list_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            input_image = row['input']  # Input image path
            model_name = row['model']  # Model name
            # Check if the model name is valid
            if model_name not in MODEL_PATHS:
                print(f"Invalid model '{model_name}' for image {input_image}. Skipping...")
                continue
            model_path = MODEL_PATHS[model_name]
            process_image(input_image, model_path, output_folder)

def process_image(input_image, model_path, output_folder):
    # Generate the output paths based on the input image
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_stem_map = os.path.join(output_folder, base_name + '_stem_map.tif')
    output_stems = os.path.join(output_folder, base_name)
    output_nodes = os.path.join(output_folder, base_name)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare the command
    command = [
        'python3', '-u', 'winmol_run.py', 
        'model', model_path, 
        'input', input_image, 
        'output_stem_map', output_stem_map, 
        'output_stems', output_stems, 
        'output_nodes', output_nodes
    ]

    # Execute the command
    try:
        print(f"Processing {input_image} with model {model_path}...")
        subprocess.run(command, check=True)
        print(f"Processing completed for {input_image}. Outputs saved in {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_image}: {e}")

# Example usage:
file_list_path = 'orthomosaic_file_list.csv'  # CSV file containing input paths and model names
output_folder = 'output_folder'  # Folder where the output files will be saved

process_orthomosaics(file_list_path, output_folder)