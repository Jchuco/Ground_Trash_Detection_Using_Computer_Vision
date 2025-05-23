import os
import cv2
from ultralytics import YOLO


modelo_path = "../models/trained_models/best.pt"

pasta_imagens = "real_images"

pasta_saida = "real_images_output"
os.makedirs(pasta_saida, exist_ok=True)


min_area = 1000


model = YOLO(modelo_path)

for nome_arquivo in os.listdir(pasta_imagens):
    if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        caminho_img = os.path.join(pasta_imagens, nome_arquivo)

        
        results = model.predict(source=caminho_img, conf=0.3, save=False)

      
        img = cv2.imread(caminho_img)

        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                conf = float(box.conf[0])
                classe = int(box.cls[0])

                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 139), thickness=10)

                
                classes = ["plastic", "paper", "metal", "glass", "other"]  
                label = classes[classe] if classe < len(classes) else str(classe)
                text = f"{label} {conf:.2f}"
                text_pos = (int(x1), max(int(y1) - 20, 30))  
                
    
                cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 5.0, (250, 250, 250), 6, lineType=cv2.LINE_AA)



        
        caminho_saida = os.path.join(pasta_saida, nome_arquivo)
        cv2.imwrite(caminho_saida, img)
        print(f"Processada: {nome_arquivo}")
