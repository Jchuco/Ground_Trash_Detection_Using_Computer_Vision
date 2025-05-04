import os
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

merge_classes_to_superclasses(
    input_json_path=f"{root_dir}/annotations.json",
    output_json_path=path_to_annotations,
    images_dir=f"{root_dir}/images_original",
    output_images_dir=original_images_dir
)

train_dir = config['train']['labels']
val_dir = config['val']['labels']
new_root_dir = config['dataset']['root_dir']

for i_dir in [train_dir, val_dir]:
    a = Path(i_dir)
    a.mkdir(parents=True, exist_ok=True)


taco2yolo(path_to_annotations, original_images_dir, f"{new_root_dir}/labels")


dataset = config['dataset']['dataset']
with open(dataset, 'r') as file:
    data = yaml.safe_load(file)
data['path'] = f"../{Path.cwd().parts[-1]}"  # current path
with open(dataset, 'w') as file:
    yaml.dump(data, file)
