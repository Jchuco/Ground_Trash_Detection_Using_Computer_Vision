from torch import Tensor
from src.data_preprocessing_SSD import TACODataset
import unittest

root_dir = "./data/"
annotation_path = "./data/annotations_5classes.json"


class TestTacoDataset(unittest.TestCase):

    def test_run_init(self):
        dataset = TACODataset(root_dir, annotation_path)
        assert len(dataset) != 0

    # def test_access_data(self):
    #     dataset = TACODataset(root_dir, annotation_path)
    #     sample = dataset[0]
    #     image, target = sample
    #
    #     assert type(image) == Tensor
    #     assert len(target.keys()) == 3
    #     assert 'boxes' in target
    #     assert 'labels' in target
    #     assert 'image_id' in target


if __name__ == "__main__":
    unittest.main()
