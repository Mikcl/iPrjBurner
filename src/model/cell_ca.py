import torch
from torch import nn

class CellCA(nn.Module):
    def __init__(self, channels, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channels*3,out_channels=dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim, out_channels=channels, kernel_size=1),
        )
    def forward(self, x):
        return self.net(x)