#!/usr/bin/python

import os
import sys

import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

from custom_dataloader import get_dataloaders

train_dataloader, test_dataloader = get_dataloaders(batch_size=16, num_workers=8)

# TODO 1: Import pytorch lightning, wandb(logging), timm and use it to load a pretrained model
# TODO 2: Modify the model's classifier to output 3 classes instead of X (defined by the model)
# TODO 3: Train the model + Logging & Saving best ckpt for 10 epochs and report test accuracy 

