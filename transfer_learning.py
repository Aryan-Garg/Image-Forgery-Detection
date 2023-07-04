#!/usr/bin/python

# Imports
### System
import os
import argparse

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

# DONE_TODO 1: Import pytorch lightning, wandb(logging), timm and use it to load a pretrained model 
# TODO 2: Modify the model's classifier to output 3 classes instead of X (defined by the model)
# TODO 3: Train the model + Logging & Saving best ckpt for 10 epochs and report test accuracy 

def get_args():
    args = argparse.ArgumentParser(description='Transfer Learning')
    args.add_argument('--model', type=str, default='vgg16', required=True, help='Model to use [vgg16, vgg19, resnet50, resnet101, effb4, effb7]')
    args.add_argument('--batch_size', type=int, default=16, help='Batch size')

    return args.parse_args()


def get_model(modelName):
    if modelName == 'vgg16':
        model = timm.create_model('vgg16', pretrained=True)
    elif modelName == 'vgg19':
        model = timm.create_model('vgg19', pretrained=True)
    elif modelName == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
    elif modelName == 'resnet101':
        model = timm.create_model('resnet101', pretrained=True)
    elif modelName == 'effb4':
        model = timm.create_model('efficientnet_b4', pretrained=True)
    elif modelName == 'effb7':
        model = timm.create_model('efficientnet_b7', pretrained=True)

    return model


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Transfer Learning')
    model = get_model(args.model)
    print(summary(model, (3, 224, 224)))
    
