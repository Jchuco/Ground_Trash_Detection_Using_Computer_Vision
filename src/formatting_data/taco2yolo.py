from pylabel import importer
import os

path_to_annotations = "./data/annotations_5classes.json"
path_to_images = "./data"

dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="BCCD_coco")

for i in range(1, 16):
    p = f'./data/labels/batch_{i}'
    os.makedirs(p, exist_ok=True)

dataset.path_to_annotations = "./data"
dataset.export.ExportToYoloV5(output_path = "./data/labels")[0]