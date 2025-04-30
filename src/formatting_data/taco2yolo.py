from pylabel import importer
import os


def taco2yolo(path_to_annotations, path_to_images, path_to_new_annotations, output_path): 
    
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="TACODataset")

    dataset.path_to_annotations = path_to_new_annotations
    dataset.export.ExportToYoloV5(output_path = output_path)[0]