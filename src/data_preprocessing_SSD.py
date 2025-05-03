import os
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch
import torchvision.transforms.functional as F
from src.utils import get_config


class TACODataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing TACO dataset (COCO format) with optional augmentations.
    
    Args:
        root_dir (str): Path to the root directory containing images
        annotation_file (str): Path to COCO format annotation JSON file
        transform (albumentations.Compose): Transformations to apply to images
        augment (bool): Whether to apply data augmentation (default: False)
    """
    
    def __init__(self, root_dir, annotation_file, transform=None, augment=False):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.augment = augment

        self.id_to_filename = {
            img['id']: img['file_name'] for img in self.coco.dataset['images']
        }

        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=20, p=0.3),
            A.Blur(blur_limit=3, p=0.1)
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1,
        ))

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Loads and processes a single image and its annotations.
        
        Args:
            idx (int): Index of the image to load
            
        Returns:
            tuple: (image_tensor, target_dict) where target_dict contains:
                - boxes: Normalized bounding boxes [x_min, y_min, x_max, y_max]
                - labels: Class labels for each box
                - image_id: Original image ID
        """
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERRO] Imagem não encontrada ou inválida: {img_path}")
            return torch.zeros((3, 416, 416)), {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([-1])
            }

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            x = max(0, float(x))
            y = max(0, float(y))
            w = min(float(w), width - x)
            h = min(float(h), height - y)

            if w > 0 and h > 0:
                boxes.append([
                    max(0, x) / width,
                    max(0, y) / height,
                    min(width, x + w) / width,
                    min(height, y + h) / height
                ])
                labels.append(ann['category_id'])

        if self.augment and len(boxes) > 0:
            augmented = self.augmentation(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = augmented['image']
            boxes = []
            for box in augmented['bboxes']:
                x_min, y_min, x_max, y_max = box
                boxes.append([
                    x_min * width,
                    y_min * height,
                    x_max * width,
                    y_max * height
                ])

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.as_tensor([img_id]),
        }

        return image, target


def create_data_loaders(config_path):
    """
    Creates training and validation data loaders from configuration.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        tuple: (train_loader, val_loader) PyTorch DataLoader instances
    """
    config = get_config(config_path)

    transform = A.Compose([
        A.Resize(height=416, width=416),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = TACODataset(
        root_dir=config['train']['images'],
        annotation_file=config['train']['annotations'],
        transform=transform,
        augment=True
    )

    val_dataset = TACODataset(
        root_dir=config['val']['images'],
        annotation_file=config['val']['annotations'],
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val']['batch_size'],
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    return train_loader, val_loader
