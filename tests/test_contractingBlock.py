import pytest
import sys

sys.path.append('..')

from contractingBlock import ContractingBlock

class TestContractingBlock():

    def test_num_channels_positive(self):
        with pytest.raises(ValueError):
                ContractingBlock(-1)
        
    def test_num_channels_zero(self):
         with pytest.raises(ValueError):
                ContractingBlock(0)