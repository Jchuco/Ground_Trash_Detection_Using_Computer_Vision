from src.data_preprocessing_SSD import create_data_loaders
import unittest
from src.utils import get_config

config_path = "./config/config.yaml"


class TestDataPreprocessing(unittest.TestCase):

    def test_dataloaders(self):
        try:
            create_data_loaders(config_path)
            assert True
        except Exception as _:
            assert False

    def test_format_train_loader(self):

        train_loader, _ = create_data_loaders(config_path)
        train_batch = next(iter(train_loader))
        images, targets = train_batch
        config = get_config(config_path)
        assert len(images) == config['train']['batch_size']
        assert len(images) == len(targets)

    def test_format_val_loader(self):
        _, val_loader = create_data_loaders(config_path)
        val_batch = next(iter(val_loader))
        images, targets = val_batch
        config = get_config(config_path)
        assert len(images) == config['val']['batch_size']
        assert len(images) == len(targets)


if __name__ == "__main__":
    unittest.main()
