import sys
import torch
import pytest
from PIL import Image
sys.path.append('..')

from helper import show_tensor_images

class TestShowTensorImages():
    def test_show_non_tensor(self):
        img = Image.open('testInput.jpg')
        with pytest.raises(ValueError):
            show_tensor_images(img, size = (3, 300, 300))
    
    def test_show_incorrect_dimensions(self):
        testTensor = torch.randn(1, 3, 300, 600)
        with pytest.raises(ValueError):
            show_tensor_images(testTensor, size = (1, 3, 4, 5))

    def test_show_larger_new_shape(self):
        testTensor = torch.randn(1, 3, 300, 600)
        with pytest.raises(ValueError):
            show_tensor_images(testTensor, size = (3, 300, 700))
    
    def test_show_non_positive_num(self):
        testTensor = torch.randn(1, 3, 300, 600)
        with pytest.raises(ValueError):
            show_tensor_images(testTensor, 0, size = (3, 300, 300))

