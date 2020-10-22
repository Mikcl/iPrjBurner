import numpy as np
import torch
from torch import nn

CHANNEL_N = 16          # Number of CA state channels
CELL_FIRE_RATE = 0.5    # How often cell fires.

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev) 


def get_living_mask(x):
  alpha = x[:, 3:4, :, :]
  # input 4d tensor, ksize (size of window for each dimension), stride for each dimension 
  # return torch.nn.functional.max_pool2d(alpha, 3, 1)
  return torch.nn.functional.max_pool2d(input=alpha, kernel_size=3, stride=1, padding=1) > 0.1


class Cell(nn.Module):
    def __init__(self,model,channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.model = model
    
    def perceive(self, x, angle=0.0):
        identify = np.float32([0, 1, 0])
        identify = torch.Tensor(np.outer(identify, identify))
        sobel = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter

        dx = torch.Tensor(sobel)
        dy = torch.Tensor(sobel.T)

        c, s = torch.cos(torch.Tensor([angle])), torch.sin(torch.Tensor([angle]))
        x_direction =  c*dx-s*dy
        y_direction =  s*dx+c*dy
        
        i_kernel = identify[None, None, ...].repeat(self.channel_n, 1, 1, 1).to(device)  # TODO this will always be the same.
        i_v = nn.functional.conv2d(x, i_kernel, padding=1, groups=self.channel_n)

        x_kernel = x_direction[None, None, ...].repeat(self.channel_n, 1, 1, 1).to(device)
        x_v = nn.functional.conv2d(x, x_kernel, padding=1, groups=self.channel_n)
        y_kernel = y_direction[None, None, ...].repeat(self.channel_n, 1, 1, 1).to(device)
        y_v = nn.functional.conv2d(x, y_kernel, padding=1, groups=self.channel_n)

        stacked_image = torch.cat([i_v, x_v, y_v], 1)
        return stacked_image
    
    def forward(self,  x, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.model(y)*step_size

        update_mask = torch.FloatTensor(*list(x[:,:1,:,:].size())).uniform_(0.0, 1.0) <= self.fire_rate
        x = x + (dx * (update_mask).type(torch.FloatTensor)).to(device)

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        x = x * life_mask.type(torch.FloatTensor).to(device)
        return x