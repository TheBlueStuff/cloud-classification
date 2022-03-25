import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models



class ResNet50(nn.Module):
    
    def __init__(self, out_dim):
        super().__init__()
        
        self.cnn = torch.nn.Sequential(
            *(list(models.resnet50(pretrained=True).children())[:-1])
        )
        
        self.linear = nn.Linear(2048, out_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        self.linear.bias.data.fill_(0)
        
    def forward(self, x):
        
        x = F.relu(self.cnn(x).view(-1, 2048))
        x = self.linear(x)
        
        return x