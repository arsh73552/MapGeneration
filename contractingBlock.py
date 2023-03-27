from torch import nn

class ContractingBlock(nn.Module):
    def __init__(self, num_input_channels, use_dropout = False, use_batch_norm = True, dropout_prob = 0.5):
        '''
            Input Params:
                1. Num Input Channels:
                    Number of Input Channels for contracting block
                2. Use Dropout: 
                    Boolean represnting if dropout is to be used or not in Contracting Block.
                3. Use Batch Norm:
                    Boolean representing if Batch Normalization is to be used in Contracting Block.
                4. Dropout Prob:
                    Probability p with which a neuron is kept alive.
            Accepts num_input_channels as input and upscales using a convLayer then uses pooling to contract the block.
        '''
        super(ContractingBlock, self).__init__()
        if num_input_channels <= 0:
            raise ValueError(
                'Expected positive Input Channels, '
                f'got {num_input_channels}.'
            )
        self.firstConvLayer = nn.Conv2d(num_input_channels, num_input_channels * 2, kernel_size = 3, padding = 1)
        self.secondConvLayer = nn.Conv2d(num_input_channels * 2, num_input_channels * 2, kernel_size = 3, padding = 1)
        self.activation = nn.LeakyReLU(0.2)
        if use_batch_norm:
            self.batchNorm = nn.BatchNorm2d(num_input_channels * 2)
        self.use_batch_norm = use_batch_norm
        if use_dropout:
            self.dropout = nn.Dropout(dropout_prob)
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