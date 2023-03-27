import sys
import pytest

sys.path.append('..')

from Unet import UNet

class TestUNet():
    def test_unet_non_int(self):
        with pytest.raises(ValueError):
            UNet(1.5, 2, 3)
    
    def test_unet_positive_input_channels(self):
        with pytest.raises(ValueError):
            UNet(-1, 1, 2)

    def test_unet_positive_output_channels(self):
        with pytest.raises(ValueError):
            UNet(1, -1, 2)
    
    def test_unet_positive_hidden_channels(self):
        with pytest.raises(ValueError):
            UNet(1, 1, -2)