import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    """A Bottleneck Residual Block is a variant of the residual block that utilises 1x1 convolutions
    to create a bottleneck in the first and final layers of the block. The main function of a bottleneck block
    is to reduces the number of parameters and matrix multiplications required in the residual connection. This
    makes residual blocks as thin as possible so we can increase depth and have less parameters, which results in
    better computational efficiency."""

    expansion = 4

    def __init__(self, in_channels, out_channels, skip_connection=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.skip_connection = skip_connection
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.skip_connection:
            residual = self.skip_connection(residual)
        out += residual
        out = self.relu(out)
        return out
