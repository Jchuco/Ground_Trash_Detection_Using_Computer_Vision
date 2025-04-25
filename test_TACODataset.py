import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data_preprocessing import TACODataset
import unittest

root_dir = "./data/"
annotation_path = "./data/annotations.json"

class TestTacoDataset(unittest.TestCase):
    
    def visualize_sample(dataset, idx, save_path=None):
        """Visualiza uma amostra do dataset com caixas delimitadoras"""
        image, target = dataset[idx]

        if torch.is_tensor(image):
            image = image.numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
        else:
            image = np.array(image)

        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image,
                        f"{label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
    
    def test_run_init(self):

        # 1. Teste de Inicialização
        try:
            dataset = TACODataset(root_dir, annotation_path)
            print("✅ Classe inicializada com sucesso")
            print(f"Total de amostras: {len(dataset)}")
        except Exception as e:
            print(f"❌ Falha na inicialização: {str(e)}")
            assert False
        assert True

    def test_access_data(self):
        # 2. Teste de Acesso aos Dados
        try:
            dataset = TACODataset(root_dir, annotation_path)
            sample = dataset[0]
            image, target = sample
            print("✅ Acesso aos dados funcionando")
            print(f"Tipo da imagem: {type(image)}")
            print(f"Chaves do target: {target.keys()}")
        except Exception as e:
            print(f"❌ Falha no acesso aos dados: {str(e)}")
            assert False
        assert True

        # 3. Teste de Formato das Anotações
    def test_format(self):
        dataset = TACODataset(root_dir, annotation_path)
        sample = dataset[0]
        image, target = sample
        assert 'boxes' in target, "Campo 'boxes' faltando"
        assert 'labels' in target, "Campo 'labels' faltando"
        print(f"✅ Formato das anotações correto")
        print(f"Número de objetos na amostra: {len(target['boxes'])}")

    def test_dataloader(self):
        # 4. Teste de DataLoader
        try:
            dataset = TACODataset(root_dir, annotation_path)
            loader = DataLoader(dataset, shuffle=True)
            batch = next(iter(loader))
            print("✅ DataLoader funcionando")
            print(f"Tamanho do batch: {len(batch)}")
            print(f"Formato das imagens: {batch[0][0].shape}")
        except Exception as e:
            print(f"❌ Falha no DataLoader: {str(e)}")
            assert False
        assert True
            

        # # 5. Visualização de Amostras
        # print("\n=== Visualização de Amostras ===")
        # for i in range(2):  # Visualiza 2 amostras aleatórias
        #     idx = torch.randint(0, len(dataset), (1,)).item()
        #     print(f"\nAmostra {i + 1} (Índice {idx}):")
        #     print(f"Categorias presentes: {[dataset.id_to_filename[l.item()] for l in target['labels']]}")
        #     visualize_sample(dataset, idx, save_path=f"sample_{i}.png")


if __name__ == "__main__":
    unittest.main()
