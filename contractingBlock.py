from torch import nn

class ContractingBlock(nn.Module):
    def __init__(self, num_input_channels, use_dropout = False, use_batch_norm = True):
        super(ContractingBlock, self).__init__()
        self.firstConvLayer = nn.Conv2d(num_input_channels, num_input_channels * 2, kernel_size = 3, padding = 1)
        self.secondConvLayer = nn.Conv2d(num_input_channels * 2, num_input_channels * 2, kernel_size = 3, padding = 1)
        self.activation = nn.LeakyReLU(0.2)
        if use_batch_norm:
            self.batchNorm = nn.BatchNorm2d(num_input_channels * 2)
        self.use_batch_norm = use_batch_norm
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout
        self.pooling = nn.MaxPool2d(kernel_size= 2, stride= 2)

    def forward(self, x):
        x = self.firstConvLayer(x)
        if self.use_batch_norm:
            x = self.batchNorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.secondConvLayer(x)
        if self.use_batch_norm:
            x = self.batchNorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x

