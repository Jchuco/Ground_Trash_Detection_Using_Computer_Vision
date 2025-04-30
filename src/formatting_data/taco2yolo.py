import json
import os
import yaml

config_path = "./config/config.yaml"

with open(config_path) as f:
    config = yaml.safe_load(f)
    
root_dir = config['dataset']['root_dir']
    
with open(config['dataset']['annotation_file'], "r") as f:
    data = json.load(f)

image_id_to_path = {}
for image in data["images"]:
    batch = image["file_name"].split("/")[0]
    img_name = os.path.basename(image["file_name"])
    image_id_to_path[image["id"]] = os.path.join("./data", batch, img_name)

for ann in data["annotations"]:
    image_path = image_id_to_path[ann["image_id"]]
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    

    x_min, y_min, w, h = ann["bbox"]
    class_id = ann["category_id"] - 1

    img_info = next(img for img in data["images"] if img["id"] == ann["image_id"])
    img_w, img_h = img_info["width"], img_info["height"]

    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    width_norm = w / img_w
    height_norm = h / img_h

    with open(txt_path, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")