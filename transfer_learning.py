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

### TODO: Use WandB, Pytorch Lightning & torchsummary 
# import wandb
# import lightning.pytorch as pl
# from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
# from lightning.pytorch import Callback
# from lightning.pytorch.callbacks import DeviceStatsMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor
# from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary

### Timm
import timm
import timm.optim

### Custom
from custom_dataloader import get_dataloaders

# DONE_TODO 1: Import pytorch lightning, wandb(logging), timm and use it to load a pretrained model 
# TODO 2: Modify the model's classifier to output 3 classes instead of X (defined by the model)
# TODO 3: Train the model + Logging & Saving best ckpt for 10 epochs and report test accuracy 

def get_args():
    args = argparse.ArgumentParser(description='Transfer Learning')
    args.add_argument('--model', '-m', type=str, default='vgg16', required=True, help='Model to use [vgg16, vgg19, resnet50, resnet101, effb4, effb5]')
    args.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args.add_argument('--device', '-d', type=str, default='cuda', required=True, help='Device to use [cpu, cuda:0, cuda:1, cuda]')
    # args.print_help()
    return args.parse_args()


def get_model(modelName):
    # VGG Fam
    if modelName == 'vgg16':
        model = timm.create_model('vgg16', pretrained=True, num_classes=3)
    elif modelName == 'vgg19':
        model = timm.create_model('vgg19', pretrained=True, num_classes=3)
    
    # Res Fam: Using catavgmax pooling to increase number of features
    elif modelName == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=3, 
                                  global_pool='catavgmax')
    elif modelName == 'resnet101':
        model = timm.create_model('resnet101', pretrained=True, num_classes=3,
                                  global_pool='catavgmax')
        
    # EfficientNet Fam: Using catavgmax pooling here as well
    elif modelName == 'effb4':
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=3,
                                  global_pool='catavgmax')
    elif modelName == 'effb5':
        model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=3,
                                  global_pool='catavgmax')

    return model


def set_device(device):
    if device == 'cpu':
        device = torch.device('cpu')
    elif device == 'cuda':
        device = torch.device('cuda')
    elif 'cuda:' in device:
        device = torch.device(device)
    else:
        raise ValueError('[set_device]: Invalid device')
    
    return device


def check_args(args):
    if args.model not in ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'effb4', 'effb5']:
        raise ValueError('[!] Invalid model')

    if 'cuda' in args.device and not torch.cuda.is_available():
        raise ValueError('[!] Cuda not available')
    
    if 'cuda:' in args.device:
        if int(args.device[-1]) >= torch.cuda.device_count():
            raise ValueError('[!] Invalid cuda device. You have lesser cuda devices than the one you specified')

    args.device = set_device(args.device)
    

def calculate_accuracy(yhat, y):
    return 100. * torch.sum(yhat == y) / len(y)


def test_step(model, test_loader, device, epoch, best_test_acc, modelName, saveCKPT=False):
    model.eval()
    with torch.no_grad():
        test_acc = 0.
        for i, batch in enumerate(test_loader):
            img = batch['image'].to(device)
            # mask = batch['mask'] # Not Needed in classification setting
            y = batch['class_label'].to(device)
            print(f"{i+1} | {img.shape} | {y}")

            y_hat = model(img)
            test_acc += calculate_accuracy(y_hat, y) # TODO: Calculate acc metric here

        # TODO: Log test metrics here
        test_acc = 1. * test_acc / len(test_loader)
        print(f"Test acc: {test_acc}")

        if saveCKPT and test_acc < best_test_acc:
            # check if checkpoints dir exists; if not make it
            if not os.exists("./checkpoints"):
                os.makedirs("./checkpoints")

            torch.save(model.state_dict(), os.path.join("checkpoints/", f'{modelName}_acc_{test_acc:.3f}' + str(epoch + 1) + '.pt'))
            best_test_acc = test_acc
        
        return best_test_acc


def train_classifier(model, train_loader, test_loader, device, modelName, epochs=10, use_amp=True):
    for param in model.parameters():
        param.requires_grad = False

    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    opt = timm.optim.AdamW((param for param in model.parameters() if param.requires_grad), 
                           lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-5)

    ce_loss = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # amp == automati mixed precison

    train_epoch_acc = 0.
    best_test_acc = 0.
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        model.train()
        train_step_acc = 0.
        for i, batch in enumerate(train_loader):
            img = batch['image'].to(device)
            # mask = batch['mask'] # Not Needed in classification setting
            y = batch['class_label'].to(device)
            print(f"{i+1} | {img.shape} | {y}")

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_hat = model(img)
                l = ce_loss(y_hat, y)

            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # DONE_TODO: Log train metrics here
            train_step_acc += 100. * torch.sum(y_hat == y) / len(y) # DONE_TODO: Calculate train accuracy here 

        scheduler.step()
        train_epoch_acc = train_step_acc / len(train_loader)
        print(f"Epoch: {epoch} | Accuracy: {train_epoch_acc}")
        # Test to save best model ckpt
        best_test_acc = test_step(model, test_loader, device, epoch, best_test_acc, modelName, saveCKPT=True)


if __name__ == '__main__':
    args = get_args()
    # print("Total devices:", torch.cuda.device_count())

    check_args(args) # will also set args.device properly
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(args.device)
    else:
        device_name = 'cpu'
    print("--------------------------------------------\nSelected device:", device_name,"\n--------------------------------------------")

    model = get_model(args.model) # will also set classifier to output 3 classes
    # print(summary(model.to(args.device), (1, 3, 224, 224)))
    print("Clf layer of loaded model: ", model.get_classifier())

    train_dataloader, test_dataloader = get_dataloaders(batch_size=16, num_workers=8)

    train_classifier(model, train_dataloader, test_dataloader, args.device, args.model)
