# Ground Garbage Detection Using Computer Vision

This project presents an automated system for detecting and classifying 
ground garbage using deep learning and computer vision. Leveraging the 
YOLOv8 model, the system is trained on a customized version of the **TACO 
(Trash Annotations in Context)** dataset, which has been simplified into 
five superclasses: **plastic**, **paper**, **metal**, **glass**, and **other**. The goal is 
to provide a real-time, efficient solution for urban waste detection, 
aiding in environmental sustainability and smart city initiatives.

## Setup Instructions

### Dataset Preparation
To run this project, you need to download the TACO dataset from Kaggle:
[TACO Dataset on Kaggle](https://www.kaggle.com/datasets/kneroma/tacotrashdataset?resource=download-directory&select=data)

After downloading:
1. Extract the downloaded dataset
2. Place the extracted files in the root directory of the project inside a folder named `data`
3. The directory structure should look like this:
```
   project_root/  
   │  
   ├── data/  
   │   ├── annotations.json  
   │   ├── batch_1/  
   │   │   ├── *.JPG 
   │   │   ├── ... 
   │   │   └── *.JPG  
   │   └── batch_15  
   │       ├── *.JPG 
   │       ├── ... 
   │       └── *.JPG  
   ├── src/  
   ├── README.md  
   └── ...
```

## Key Features
- **Dataset**: Customized TACO dataset with annotations converted into 
five broad superclasses for improved generalization.
- **Model**: YOLOv8-nano for real-time detection.
- **Training**: Multi-phase training with data augmentation.
- **Results**: High performance on well-represented classes, with challenges
in detecting small objects like glass.

## Future Work
- Improve detection of small objects.
- Integrate Edge AI for faster predictions.
- Deploy on mobile robots for automated street cleaning.

For details, refer to the full [project paper](./docs/Ground_Trash_Detection_Using_Computer_Vision_paper.pdf) in the repository. 
This project is inspired by advancements in waste detection and aims to 
contribute to cleaner urban environments.

## Authors
Project developed by:
- Frederico Correia Cerqueira
- Joana Chuço
- Gonçalo Abreu