#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class SRM_UNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass



test_m = SRM_UNet()
print(summary(test_m, (3, 256, 256), device='cpu'))