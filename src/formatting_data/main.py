from src.formatting_data.taco2yolo import taco2yolo
from src.formatting_data.merge_classes import merge_classes_to_superclasses
import os
from src.utils import get_config

config_path = "./config/config.yaml"

config = get_config(config_path)

path_to_annotations = config['dataset']['annotation_file']
path_to_images = config['dataset']['root_dir']
path_to_new_annotations = "./data"
output_path = f"{config['dataset']['root_dir']}/labels"

merge_classes_to_superclasses(
    input_json_path=f"{config['dataset']['root_dir']}/annotations.json",
    output_json_path=path_to_annotations
)

for i in range(1, 16):
    p = f'{output_path}/batch_{i}'
    os.makedirs(p, exist_ok=True)

taco2yolo(path_to_annotations, path_to_images, path_to_new_annotations, output_path)
