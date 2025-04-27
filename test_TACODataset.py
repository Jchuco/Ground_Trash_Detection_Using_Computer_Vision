import cv2
import numpy as np
from src.data_preprocessing import TACODataset
import unittest

root_dir = "./data/"
annotation_path = "./data/annotations.json"

class TestTacoDataset(unittest.TestCase):
    
    def test_run_init(self):
        try:
            dataset = TACODataset(root_dir, annotation_path) 
            assert len(dataset) != 0
        except Exception as e:
            assert False

    def test_access_data(self):        
        dataset = TACODataset(root_dir, annotation_path)
        sample = dataset[0]
        image, target = sample
        
        assert type(image) == np.ndarray
        assert len(target.keys()) == 3
        assert 'boxes' in target
        assert 'labels' in target
        assert 'image_id' in target

if __name__ == "__main__":
    unittest.main()