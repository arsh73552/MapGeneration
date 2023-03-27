import sys
import pytest

sys.path.append('..')

from expandingBlock import ExpandingBlock

class TestExpandingBlock():
    def test_unet_non_int(self):
        with pytest.raises(ValueError):
            ExpandingBlock(1.5)
    
    def test_unet_positive_input_channels(self):
        with pytest.raises(ValueError):
            ExpandingBlock(-1)