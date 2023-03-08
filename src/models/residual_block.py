import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    """
    A residual block is a block of convolutional layers with a skip connection. The skip
    connection is a shortcut that allows the input of the block to be added to the output of the
    convolutional layers. This allows the gradient to flow through the block without
    being affected by the convolutional layers, so the network can learn identity mappings.
    This greatly improves the performance of the network.
    """

    def __init__(self, in_channels, out_channels, stride=1, skip_connection=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.skip_connection = skip_connection
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.skip_connection:
            residual = self.skip_connection(x)
        out += residual
        out = self.relu(out)
        return out
