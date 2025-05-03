from ultralytics import YOLO



model = YOLO('./models/yolov8n.pt')

dataset = "./data/dataset.yaml"
print("Phase 1: Training head only")
model.train(
    data=dataset,
    epochs=20,
    lr0=1e-3,
    freeze=[x for x in range(10)],
    imgsz=100,
    batch=15,
)

model.export(format='onnx')
