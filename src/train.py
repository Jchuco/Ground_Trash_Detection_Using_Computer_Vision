import os
import yaml
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torch.utils.data import DataLoader
from data_preprocessing_SSD import TACODataset  
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

# Verificar se o dataset está a ser carregado corretamente
print(f"Dataset contém {len(dataset)} imagens.")
dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Inicializar modelo
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.train()

# Mover o modelo para o dispositivo (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Otimizador
optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

# Função de salvar modelo
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Loop de treino
for epoch in range(config['train']['num_epochs']):
    print(f"Starting epoch {epoch+1}/{config['train']['num_epochs']}")
    epoch_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}")

        # Mover imagens e alvos para o dispositivo (GPU ou CPU)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calcular perdas
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Atualizar pesos
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Somar perdas para a época
        epoch_loss += losses.item()

    print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Loss: {epoch_loss:.4f}")

# Salvar o modelo final
os.makedirs(config['output_dir'], exist_ok=True)
save_model(model, f"{config['output_dir']}/model_final.pth")
print(f"Model saved to {config['output_dir']}/model_final.pth")

if __name__ == "__main__":
    print("Treino concluido com sucesso!")
