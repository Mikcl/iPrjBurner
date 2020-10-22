'''
    Training Regimes employable.
'''
import torch
from torch import nn;

class GrowingCA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, min_steps=64, max_steps=96):
        iter_n = torch.randint(min_steps,max_steps, (1,))
        for _ in torch.range(0,iter_n[0]):
            x = self.model(x)
        x = x[:,:4,:,:]
        return x
