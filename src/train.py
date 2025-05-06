from ultralytics import YOLO
from src.utils import get_config

config = get_config("config/config.yaml")

model = YOLO(config['model']['name'])

dataset = config['dataset']['dataset']

model.train(
    data=dataset,
    epochs=20,
    lr0=1e-3,
    freeze=10,
    # Augmentations
    hsv_h=0.015,   # Hue shift
    hsv_s=0.7,     # Saturation shift
    hsv_v=0.4,     # Value shift
    flipud=0.5,    # Vertical flip probability
    fliplr=0.5,    # Horizontal flip probability
)

model.train(
    data=dataset,
    epochs=40,
    lr0=5e-4,
    freeze=8,
    cos_lr=True,
    # Augmentations
    hsv_h=0.015,   # Hue shift
    hsv_s=0.7,     # Saturation shift
    hsv_v=0.4,     # Value shift
    flipud=0.5,    # Vertical flip probability
    fliplr=0.5,    # Horizontal flip probability
)

model.train(
    data=dataset,
    epochs=40,
    lr0=1e-4,
    freeze=0,
    # Augmentations
    hsv_h=0.015,   # Hue shift
    hsv_s=0.7,     # Saturation shift
    hsv_v=0.4,     # Value shift
    flipud=0.5,    # Vertical flip probability
    fliplr=0.5,    # Horizontal flip probability
)

# Export to ONNX
model.export(format='onnx')
