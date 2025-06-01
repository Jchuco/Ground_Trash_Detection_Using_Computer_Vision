from ultralytics import YOLO
from src.utils import get_config

config = get_config("config/config.yaml")

model = YOLO("./models/best.pt")

dataset = "./dataset_augmented/dataset.yaml"

model.train(
    data=dataset,
    epochs=20,
    lr0=1e-3,
    freeze=10,
    rect=True,
    patience=10,
    half=True,
    # Augmentations
    hsv_h=0.015,   # Hue shift
    hsv_s=0.7,     # Saturation shift
    hsv_v=0.4,     # Value shift
    flipud=0.5,    # Vertical flip probability
    fliplr=0.5,    # Horizontal flip probability
    degrees=10,    # Rotation degrees
    scale=0.5,     # Scale augmentation
)

# model.train(
#     data=dataset,
#     epochs=20,
#     lr0=1e-3,
#     freeze=10,
#     rect=True,
#     imgsz=768,
#     batch=8,
#     patience=10,
#     half=True,
#     # Augmentations
#     hsv_h=0.015,   # Hue shift
#     hsv_s=0.7,     # Saturation shift
#     hsv_v=0.4,     # Value shift
#     flipud=0.5,    # Vertical flip probability
#     fliplr=0.5,    # Horizontal flip probability
#     degrees=10,    # Rotation degrees
#     scale=0.5,     # Scale augmentation
# )
#
# model.train(
#     data=dataset,
#     epochs=40,
#     lr0=5e-4,
#     freeze=8,
#     rect=True,
#     device=0,
#     cos_lr=True,
#     imgsz=768,
#     batch=8,
#     patience=10,
#     half=True,
#     # Augmentations
#     hsv_h=0.015,   # Hue shift
#     hsv_s=0.7,     # Saturation shift
#     hsv_v=0.4,     # Value shift
#     flipud=0.5,    # Vertical flip probability
#     fliplr=0.5,    # Horizontal flip probability
#     degrees=10,    # Rotation degrees
#     scale=0.5,     # Scale augmentation
# )
#
# model.train(
#     data=dataset,
#     epochs=40,
#     lr0=1e-4,
#     freeze=0,
#     rect=True,
#     device=0,
#     imgsz=768,
#     batch=8,
#     patience=10,
#     half=True,
#     # Augmentations
#     hsv_h=0.015,   # Hue shift
#     hsv_s=0.7,     # Saturation shift
#     hsv_v=0.4,     # Value shift
#     flipud=0.5,    # Vertical flip probability
#     fliplr=0.5,    # Horizontal flip probability
#     degrees=10,    # Rotation degrees
#     scale=0.5,     # Scale augmentation
# )

# Export to ONNX
model.export(format='onnx')
