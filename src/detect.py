import os
import cv2
from ultralytics import YOLO

from src.utils import get_config

config = get_config("./config/config.yaml")

modelo_path = config['model_best']['name']
images_path = config['real_dataset']['input_dir']
output_path = config['real_dataset']['output_dir']
os.makedirs(output_path, exist_ok=True)

min_area = 1000


model = YOLO(modelo_path)

for file_name in os.listdir(images_path):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_path, file_name)
        
        results = model.predict(source=image_path, conf=0.3, save=False)
      
        img = cv2.imread(image_path)
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                conf = float(box.conf[0])
                label_type = int(box.cls[0])

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 139), thickness=10)
                
                array_type = ["plastic", "paper", "metal", "glass", "other"]
                label = array_type[label_type] if label_type < len(array_type) else str(label_type)
                text = f"{label} {conf:.2f}"
                text_pos = (int(x1), max(int(y1) - 20, 30))  
                
                cv2.putText(
                    img,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    5.0,
                    (250, 250, 250),
                    6,
                    lineType=cv2.LINE_AA
                )

        image_output_path = os.path.join(output_path, file_name)
        cv2.imwrite(image_output_path, img)
        print(f"Process: {file_name}")
