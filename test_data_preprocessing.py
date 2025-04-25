import yaml
import torch
from torch.utils.data import DataLoader
from src.data_preprocessing import create_data_loaders

def test_dataloaders(config_path):
    print("=== Iniciando Teste de Data Loaders ===")
    
    # 1. Teste de Criação dos Loaders
    try:
        train_loader, val_loader = create_data_loaders(config_path)
        print("✅ Loaders criados com sucesso")
    except Exception as e:
        print(f"❌ Falha na criação: {str(e)}")
        return
    
    # 2. Teste de Formato do Train Loader
    try:
        train_batch = next(iter(train_loader))
        images, targets = train_batch
        print("\n=== Formato do Train Loader ===")
        print(f"Número de batches: {len(train_loader)}")
        print(f"Tamanho do batch: {len(images)}")
        print(f"Tipo das imagens: {type(images[0])}")
        print(f"Tipo dos targets: {type(targets[0])}")
        print(f"Chaves dos targets: {targets[0].keys()}")
    except Exception as e:
        print(f"❌ Falha no train loader: {str(e)}")
    
    # 3. Teste de Formato do Val Loader
    try:
        val_batch = next(iter(val_loader))
        images, targets = val_batch
        print("\n=== Formato do Val Loader ===")
        print(f"Número de batches: {len(val_loader)}")
        print(f"Tamanho do batch: {len(images)}")
        print(f"Exemplo de shape de imagem: {images[0].shape}")
        print(f"Número de bboxes no 1º item: {len(targets[0]['boxes'])}")
    except Exception as e:
        print(f"❌ Falha no val loader: {str(e)}")
    
    # 4. Teste de Conteúdo
    try:
        print("\n=== Inspeção de Conteúdo ===")
        print("Primeira imagem no train loader:")
        print(f"- Shape: {images[0].shape}")
        print(f"- Valor médio: {images[0].mean().item():.4f}")
        print("\nPrimeiro target no train loader:")
        print(f"- Número de bboxes: {len(targets[0]['boxes'])}")
        print(f"- Exemplo de bbox: {targets[0]['boxes'][0]}")
        print(f"- Label correspondente: {targets[0]['labels'][0]}")
    except Exception as e:
        print(f"❌ Falha na inspeção de conteúdo: {str(e)}")

if __name__ == "__main__":
    # Configuração de teste (crie um arquivo config_test.yaml)
    test_dataloaders("./config.yaml")