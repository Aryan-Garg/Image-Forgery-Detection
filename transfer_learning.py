#!/usr/bin/python

# Imports
### System
import os
import sys

### Python
import numpy as np
import cv2 as cv

### Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### WandB, Pytorch Lightning & torchsummary
import wandb
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import DeviceStatsMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary

### Timm
import timm

### Custom
from custom_dataloader import get_dataloaders

train_dataloader, test_dataloader = get_dataloaders(batch_size=16, num_workers=8)

# TODO 1: Import pytorch lightning, wandb(logging), timm and use it to load a pretrained model
# TODO 2: Modify the model's classifier to output 3 classes instead of X (defined by the model)
# TODO 3: Train the model + Logging & Saving best ckpt for 10 epochs and report test accuracy 

