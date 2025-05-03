from pylabel import importer


def taco2yolo(path_to_annotations, path_to_images, output_path):
    """
    Converts TACO dataset annotations (COCO JSON format) to YOLOv5 format.


    Args:
        path_to_annotations (str): Path to the input COCO JSON annotations file.
        path_to_images (str): Path to the directory containing the images.
        output_path (str): Directory where the new dataset will be exported.
    """
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="TACODataset")
    dataset.path_to_annotations = ""
    dataset.splitter.GroupShuffleSplit(train_pct=0.8, val_pct=0, test_pct=0.2)
    dataset.export.ExportToYoloV5(output_path=output_path, use_splits=True, copy_images=True)
