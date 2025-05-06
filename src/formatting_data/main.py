from pathlib import Path
from src.formatting_data.taco2yolo import taco2yolo
from src.formatting_data.merge_classes import merge_classes_to_superclasses
from src.utils import get_config
import yaml

config_path = "./config/config.yaml"

config = get_config(config_path)

root_dir = config['dataset_original']['root_dir']
path_to_annotations = config['dataset_original']['annotation_file']
original_images_dir = config['dataset_original']['images']
new_root_dir = config['dataset']['root_dir']

merge_classes_to_superclasses(
    input_json_path=f"{root_dir}/annotations.json",
    output_json_path=path_to_annotations,
    images_dir=f"{root_dir}/images_original",
    output_images_dir=original_images_dir
)

taco2yolo(path_to_annotations, original_images_dir, f"{new_root_dir}/labels")

# This code is to fix a bug.
# The bug was that in the dataset.yaml file
# created by the function taco2yolo, the path
# was skipping the root directory.
dataset = config['dataset']['dataset']
with open(dataset, 'r') as file:
    data = yaml.safe_load(file)
data['path'] = f"../{Path.cwd().parts[-1]}"  # Root directory
with open(dataset, 'w') as file:
    yaml.dump(data, file)
