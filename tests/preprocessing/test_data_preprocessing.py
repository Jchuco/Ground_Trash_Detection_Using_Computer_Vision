import yaml
from src.data_preprocessing import create_data_loaders
import unittest

config_path = "./config/config.yaml"

class TestDataPrepocessing(unittest.TestCase):
    
    
    
    def test_dataloaders(self):      
        try:
            create_data_loaders(config_path)
            assert True
        except Exception as e:
            assert False
        
    def test_format_train_loader(self):
        
        train_loader, _ = create_data_loaders(config_path)
        train_batch = next(iter(train_loader))
        images, targets = train_batch
        with open(config_path) as f:
            config = yaml.safe_load(f)
            assert len(images) == config['batch_size']
        assert len(images)== len(targets)
        
    def test_format_val_loader(self):
        
        val_loader, _ = create_data_loaders(config_path)
        val_batch = next(iter(val_loader))
        images, targets = val_batch
        with open(config_path) as f:
            config = yaml.safe_load(f)
            assert len(images) == config['batch_size']
        assert len(images)== len(targets)
            

if __name__ == "__main__":
    unittest.main()