import torch
from torch import nn
from getFinalDimensionsRight import FeatureMapBlock
from contractingBlock import ContractingBlock

class Discriminator(nn.Module):
    def __init__(self, num_input_channels, hidden_channels = 8) -> None:
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(num_input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_batch_norm=False)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)
    
    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn