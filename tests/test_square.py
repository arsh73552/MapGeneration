import sys
import torch
import pytest
from PIL import Image
sys.path.append('..')

from helper import crop

class TestCrop():
    
    def test_crop_non_tensor(self):
        img = Image.open('testInput.jpg')
        with pytest.raises(ValueError):
            crop(img, (1, 3, 300, 300))
    
    def test_crop_incorrect_dimensions(self):
        testTensor = torch.randn(1, 3, 300, 600)
        with pytest.raises(ValueError):
            crop(testTensor, (1, 3, 4))

    def test_crop_larger_new_shape(self):
        testTensor = torch.randn(1, 3, 300, 600)
        with pytest.raises(ValueError):
            crop(testTensor, (1, 5, 20, 30))

    def test_crop_final_shape(self):
        testTensor = torch.randn(1, 3, 300, 600)
        finalTensor = (1, 3, 300, 300)
        croppedTensor = crop(testTensor, finalTensor)
        assert(tuple(croppedTensor.shape) == finalTensor)
