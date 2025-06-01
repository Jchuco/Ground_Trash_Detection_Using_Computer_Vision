from ultralytics import YOLO
from src.utils import get_config

config = get_config("./config/config.yaml")

dataset = config['dataset']['dataset']

model = YOLO(config['model']['name']).load(config['model_best']['name'])

model.train(
    data=dataset,
    epochs=50,
    lr0=1e-3,
    rect=True,
    imgsz=640,
    batch=16,
    patience=30,
    half=True,
    classes=[3, 4],
    # Augmentations
    hsv_h=0.015,   # Hue shift
    hsv_s=0.7,     # Saturation shift
    hsv_v=0.4,     # Value shift
    flipud=0.5,    # Vertical flip probability
    fliplr=0.5,    # Horizontal flip probability
    degrees=10,    # Rotation degrees
    scale=0.5,     # Scale augmentation

)
