import torch
from torch import nn

class CellGrowingCA(nn.Module):
    def __init__(self, channels, dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels*3,out_channels=dim, kernel_size=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=channels, kernel_size=1)
        with torch.no_grad():
            self.conv2.weight.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return  x