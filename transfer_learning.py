#!/usr/bin/python

# Imports
### System
import os
import argparse

### Python
import numpy as np
import cv2 as cv
from tqdm.auto import tqdm

### Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### TODO: Use WandB, Pytorch Lightning & torchsummary 
import wandb
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import DeviceStatsMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary

### Timm
import timm
import timm.optim

### Custom
from custom_dataloader import get_dataloaders

# DONE_TODO 1: Import pytorch lightning, wandb(logging), timm and use it to load a pretrained model 
# Done_TODO 2: Modify the model's classifier to output 3 classes instead of X (defined by the model)
# Done_TODO 3: Train the model + Logging & Saving best ckpt for 10 epochs and report test accuracy 


seed_everything(42)


def get_args():
    args = argparse.ArgumentParser(description='Transfer Learning')
    args.add_argument('--model', '-m', type=str, default='vgg16', required=True, help='Model to use [vgg16, vgg19, resnet50, resnet101, effb4, effb5]')
    args.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train for')
    args.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args.add_argument('--device', '-d', type=str, default='cuda', required=True, help='Device to use [cpu, cuda:0, cuda:1, cuda]')
    args.add_argument('--mode', '-md', type=str, default='train', help='Mode to run: [train, trainX, test]. train = finetune only classifier layer. trainX = finetune last few layers including the classifier. test = test the model')
    args.add_argument('--ckpt_path', '-cp', type=str, default="", help='Path to checkpoint to load')
    args.add_argument('--lr', '-lr', type=float, default=1e-3, help='Learning rate')
    # args.print_help()
    return args.parse_args()


def get_model(modelName):
    # VGG Fam
    if modelName == 'vgg16':
        model = timm.create_model('vgg16', pretrained=True)
    elif modelName == 'vgg19':
        model = timm.create_model('vgg19', pretrained=True)
    
    # Res Fam: Using catavgmax pooling to increase number of features
    elif modelName == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
    elif modelName == 'resnet101':
        model = timm.create_model('resnet101', pretrained=True)
        
    # EfficientNet Fam: Using catavgmax pooling here as well
    elif modelName == 'effb4':
        model = timm.create_model('efficientnet_b4', pretrained=True)
    elif modelName == 'effb5':
        model = timm.create_model('efficientnet_b5', pretrained=True)

    return model


def check_args(args):
    if args.model not in ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'effb4', 'effb5']:
        raise ValueError('[!] Invalid model')

    if 'cuda' in args.device and not torch.cuda.is_available():
        raise ValueError('[!] Cuda not available')
    
    if 'cuda:' in args.device:
        if int(args.device[-1]) >= torch.cuda.device_count():
            raise ValueError('[!] Invalid cuda device. You have lesser cuda devices than the one you specified')


class LIT_TL(pl.LightningModule):
    def __init__(self, model, modelName = "brrr", config: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.modelName = modelName
        self.config = config

        num_filters = model.fc.in_features
        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = config['classes']
        self.classifier = nn.Linear(num_filters, num_target_classes)

        self.ce_loss = nn.CrossEntropyLoss()
        

    def calculate_accuracy(self, yhat, y):
        return 100. * torch.sum(yhat == y) / len(y)


    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x


    def configure_optimizers(self):
        opt = timm.optim.AdamW((param for param in self.classifier.parameters() if param.requires_grad), 
                               lr=self.config['lr'], 
                               weight_decay=self.config['decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 
                                                                         T_0=self.config['T_0'], 
                                                                         eta_min=self.config['eta_min'])
        return [opt], [scheduler]

    
    def training_step(self, batch, batch_idx):
        img = batch['image'].to(self.device)
        # mask = batch['mask'] # Not Needed in classification setting
        y = batch['class_label'].to(self.device)

        y_hat = self.forward(img)
        loss = self.ce_loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        # print(f"y: {y}, y_hat: {y_hat}")

        # DONE_TODO: Log train metrics here
        train_step_acc = self.calculate_accuracy(preds, y) # DONE_TODO: Calculate train accuracy here 
        self.log("train_acc", train_step_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        test_acc = 0.
            
        img = batch['image'].to(self.device)
        # mask = batch['mask'] # Not Needed in classification setting
        y = batch['class_label'].to(self.device)
        # print(f"{i+1} | {img.shape} | {y}")

        y_hat = self.forward(img)
        preds = torch.argmax(y_hat, dim=1)
        test_acc += self.calculate_accuracy(preds, y) # TODO: Calculate acc metric here

        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        test_acc = 0.
            
        img = batch['image'].to(self.device)
        # mask = batch['mask'] # Not Needed in classification setting
        y = batch['class_label'].to(self.device)
        # print(f"{i+1} | {img.shape} | {y}")

        y_hat = self.forward(img)
        preds = torch.argmax(y_hat, dim=1)
        test_acc += self.calculate_accuracy(preds, y) # TODO: Calculate acc metric here

        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)


def get_config(args):
    config = {
        'model': args.model,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device': args.device,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'T_0': 100,
        'eta_min': 1e-4,
        'classes': 3,
        'decay': 1e-3,
    }
    return config


if __name__ == '__main__':
    args = get_args()
    # print("Total devices:", torch.cuda.device_count())

    check_args(args) # will also set args.device properly
    config = get_config(args)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(args.device)
    else:
        device_name = 'cpu'
    print("--------------------------------------------\nSelected device:", device_name,"\n--------------------------------------------")
    print(f"[+] Model Selected: {args.model}")
    
    model = get_model(args.model)
    lit_model = LIT_TL(model, args.model, config)
    
    train_dataloader, test_dataloader = get_dataloaders(batch_size=args.batch_size, num_workers=8)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="test_acc", 
                                          mode="max", 
                                          save_top_k=1, 
                                          dirpath="checkpoints/", 
                                          filename=f"{args.model}" + "_{test_acc:.3f}")
    
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    # early_stop_callback = EarlyStopping(monitor="loss", patience=99)

    wandb.login()
    
    wandb_logger = WandbLogger(project='forgery_detection',
                           name=f'TL_{args.model}_norm_warmRestarts_higherLR',
                           config=config,
                           job_type='finetuning',
                           log_model="all")
    # call trainer
    trainer = Trainer(fast_dev_run=False,
                      inference_mode=False,  # to enable grad enabling during inference
                      max_epochs=args.epochs,
                      accelerator="gpu" if "cuda" in args.device else "cpu",
                      devices=[int(args.device.split(":")[-1])], # GPU ID that you selected
                      precision="16", # automatic mixed precision training
                      deterministic=True,
                      enable_checkpointing=True,
                      callbacks=[checkpoint_callback, lr_monitor],
                      gradient_clip_val=None, 
                      log_every_n_steps=50,
                      logger=wandb_logger, # The absolute best: wandb <3
                      enable_progress_bar=True)

    # fit model
    if args.mode == 'train' or args.mode == 'trainX': # TODO: Implement trainX mode
        trainer.fit(lit_model, train_dataloader, test_dataloader)
    else:
        # DONE_TODO: Load last checkpoint and test
        if not os.exists(args.ckpt_path):
            args.ckpt_path = checkpoint_callback.best_model_path
        lit_model = LIT_TL.load_from_checkpoint(args.ckpt_path)
        lit_model.freeze()
        trainer.test(lit_model, test_dataloader)
    
    wandb.finish()
