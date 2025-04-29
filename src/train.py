import os
import yaml
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from src.data_preprocessing import TACODataset  
from torchvision import transforms

# Carregar configuração
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inicializar dataset e dataloader
transform = transforms.ToTensor()
dataset = TACODataset(
    root_dir=config['dataset']['root_dir'],
    annotation_file=config['dataset']['annotation_file'],
    #transform=transform,
    #augment=True
)
dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Inicializar modelo
model = fasterrcnn_resnet50_fpn(pretrained=config['model']['pretrained'])
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Otimizador
optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

# Loop de treino
for epoch in range(config['train']['num_epochs']):
    epoch_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
    
    print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Loss: {epoch_loss:.4f}")
