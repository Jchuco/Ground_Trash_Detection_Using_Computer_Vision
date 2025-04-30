from pylabel import importer
import os


def taco2yolo(path_to_annotations, path_to_images, path_to_new_annotations, output_path): 
    """
    Converts TACO dataset annotations (COCO JSON format) to YOLOv5 format.

    Args:
        path_to_annotations (str): Path to the input COCO JSON annotations file.
        path_to_images (str): Path to the directory containing the images.
        path_to_new_annotations (str): Path where YOLO-style annotations will be saved (must exist).
        output_path (str): Directory where the new dataset will be exported.
    """
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="TACODataset")

    dataset.path_to_annotations = path_to_new_annotations
    dataset.export.ExportToYoloV5(output_path = output_path)[0]