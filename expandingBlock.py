import torch
from torch import nn
from helper import crop

class ExpandingBlock(nn.Module):
    def __init__(self, num_input_channels, use_dropout = False, use_batch_norm = True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor= 2, mode = "nearest")
        self.firstConvLayer = nn.Conv2d(num_input_channels, num_input_channels // 2, kernel_size=2)
        self.secondConvLayer = nn.Conv2d(num_input_channels, num_input_channels // 2, kernel_size=3, padding=1)
        self.thirdConvLayer = nn.Conv2d(num_input_channels // 2, num_input_channels // 2, kernel_size=2, padding=1)
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        if use_batch_norm:
            self.batchnorm = nn.BatchNorm2d(num_input_channels // 2)
        self.use_batch_norm = use_batch_norm
        self.activation = nn.LeakyReLU()

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.firstConvLayer(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.secondConvLayer(x)
        if self.use_batch_norm:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.thirdConvLayer(x)
        if self.use_batch_norm:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x
        

