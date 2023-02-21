import torch
import torch.functional as F
import torch.nn as nn


class ResidualBlock(nn.Modeule):
    def __init__(self):
        super(ResidualBlock, self).__init__()
