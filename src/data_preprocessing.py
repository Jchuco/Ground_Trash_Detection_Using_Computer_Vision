import os, yaml
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torch


class TACODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, augment=False):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.augment = augment

        # Mapeamento de categorias TACO para IDs numéricos
        self.id_to_filename = {
            img['id']: img['file_name'] for img in self.coco.dataset['images']
        }
        # Augmentations
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=20, p=0.3),
            A.Blur(blur_limit=3, p=0.1)
        ], bbox_params=A.BboxParams(format='coco'))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # Carregar imagem
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Carregar anotações
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Extrair bounding boxes e labels
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # COCO to [x1, y1, x2, y2]
            labels.append(ann['category_id'])

        if self.augment and len(boxes) > 0:
            augmented = self.augmentation(
                image=image,
                bboxes=boxes,
                category_id=labels
            )
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['category_id']

        # Converter para tensor
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.as_tensor([img_id])
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def create_data_loaders(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    transform = A.Compose([
        A.Resize(height=416, width=416),
        A.Normalize()
    ])
    
    train_dataset = TACODataset(
        images_path=config['train_images'],
        annotations_path=config['train_annotations'],
        transform=transform,
        augment=True
    )
    
    val_dataset = TACODataset(
        images_path=config['val_images'],
        annotations_path=config['val_annotations'],
        transform=transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    return train_loader, val_loader
