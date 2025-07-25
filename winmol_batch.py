import os
import subprocess
import argparse

# Fixed paths
MODEL_PATHS = {
    "spruce": (
        "./standalone/model/"
        "model_UNet_SpecDS_Spruce_512_2023-02-27_061925.hdf5"
    ),
    "beech": (
        "./standalone/model/"
        "model_UNet_SpecDS_Beech_512_2023-02-28_042751.hdf5"
    ),
    "general": (
        "./standalone/model/"
        "model_UNet_GenDS_512_2023-02-27_211141.hdf5"
    ),
}


def process_all_orthomosaics(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Invalid model name: {model_name}. Must be one of "
            f"{list(MODEL_PATHS.keys())}."
        )
        
    model_path = MODEL_PATHS[model_name]

    orthos = [
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(('.tif', '.tiff'))
    ]

    if not orthos:
        print(f"No orthomosaics found in {INPUT_FOLDER}.")
        return

    for ortho in orthos:
        process_image(ortho, model_path)


def process_image(input_image, model_path):
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_stem_map = os.path.join(
        OUTPUT_FOLDER, base_name + '_stem_map.tif'
    )
    output_stems = os.path.join(OUTPUT_FOLDER, base_name)
    output_nodes = os.path.join(OUTPUT_FOLDER, base_name)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    command = [
        'python3', '-u', 'winmol_run.py',
        model_path,
        input_image,
        output_stem_map,
        output_stems,
        output_nodes,
        'Nodes'
    ]

try:
        print(
            f"Processing {input_image} with model "
            f"{os.path.basename(model_path)}..."
        )
        subprocess.run(command, check=True)
        print(f" ^|^s Done: {base_name}")
    except subprocess.CalledProcessError as e:
        print(f" ^|^w Failed: {input_image}. Reason: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process orthomosaics in ./standalone/input."
    )
    parser.add_argument(
        "model",
        choices=MODEL_PATHS.keys(),
        help="Model to use (spruce, beech, general)"
    )

    args = parser.parse_args()
    process_all_orthomosaics(args.model)
    
