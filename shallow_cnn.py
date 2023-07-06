#!/usr/bin/python

# Imports
### System
import os
import argparse

### Python
import numpy as np
import cv2 as cv
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

### Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


### sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

### WandB, Pytorch Lightning & torchsummary 
import wandb
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import DeviceStatsMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torchsummary import summary

### Custom
from custom_dataloader import get_dataloaders
from models.shallow_CNN_arch import ShallowCNN

# DONE_TODO 1: Import pytorch lightning, wandb(logging), timm and use it to load a pretrained model 
# Done_TODO 2: Modify the model's classifier to output 3 classes instead of X (defined by the model)
# Done_TODO 3: Train the model + Logging & Saving best ckpt for 10 epochs and report test accuracy 


def get_args():
    args = argparse.ArgumentParser(description='Transfer Learning')
    args.add_argument('--model', '-m', type=str, default='custom_cnn', required=False, help='Redundant flag (only kept in to ensure compatibility with other scripts)')
    args.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train for')
    args.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args.add_argument('--device', '-d', type=str, default='cuda', required=True, help='Device to use [cpu, cuda:0, cuda:1, cuda]')
    args.add_argument('--mode', '-md', type=str, default='train', help='Mode to run: [train, test]')
    args.add_argument('--ckpt_path', '-cp', type=str, default="", help='Path to checkpoint to load')
    args.add_argument('--lr', '-lr', type=float, default=1e-3, help='Learning rate')
    args.add_argument('--num_workers', '-nw', type=int, default=8, help='Number of workers for dataloader')
    args.add_argument('--exp_name', '-en', type=str, default='generic_exp', help='Experiment name for wandb')
    args.add_argument('--use_cam', action='store_true', help='Use Class Activation Maps Loss')
    # args.print_help()
    return args.parse_args()

def check_args(args):
    if 'cuda' in args.device and not torch.cuda.is_available():
        raise ValueError('[!] Cuda not available')
    
    if 'cuda:' in args.device:
        if int(args.device[-1]) >= torch.cuda.device_count():
            raise ValueError('[!] Invalid cuda device. You have lesser cuda devices than the one you specified')


class LIT_CNN(pl.LightningModule):
    def __init__(self, model, modelName = "brrr", config: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.modelName = modelName
        self.config = config

        self.model = model
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

        self.ce_loss = nn.CrossEntropyLoss()

        # CAM Module
        self.use_cam = config['use_cam']
        if self.use_cam:
            self.gap_fc = nn.utils.spectral_norm(nn.Linear(self.num_filters, 1, bias=False))
            self.gmp_fc = nn.utils.spectral_norm(nn.Linear(self.num_filters, 1, bias=False))
            self.conv1x1 = nn.Conv2d(self.num_filters * 2, self.num_filters, kernel_size=1, stride=1, bias=True)
            self.conv_classifier = nn.utils.spectral_norm(
                nn.Conv2d(self.num_filters, 1, kernel_size=4, stride=1, padding=0, bias=False))
            self.cam_loss = nn.CrossEntropyLoss()

        self.val_preds = []
        self.val_gt = []

        self.test_preds = []
        self.test_gt = []

    
    def init_xavier(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def calculate_accuracy(self, yhat, y):
        return 100. * torch.sum(yhat == y) / len(y)


    def forward(self, x):
        if self.use_cam:
            rep = self.feature_extractor(x)
            gap = torch.nn.functional.adaptive_avg_pool2d(rep, 1)
            gap_logit = self.gap_fc(gap.view(rep.shape[0], -1))
            gap_weight = list(self.gap_fc.parameters())[0]
            gap = rep * gap_weight.unsqueeze(2).unsqueeze(3)

            gmp = torch.nn.functional.adaptive_max_pool2d(rep, 1)
            gmp_logit = self.gmp_fc(gmp.view(rep.shape[0], -1))
            gmp_weight = list(self.gmp_fc.parameters())[0]
            gmp = rep * gmp_weight.unsqueeze(2).unsqueeze(3)

            c_logit = torch.cat([gap_logit, gmp_logit], 1)
            rep = torch.cat([gap, gmp], 1)
            rep = self.leaky_relu(self.conv1x1(rep))

            heatmap = torch.sum(rep, dim=1, keepdim=True)

            rep = self.pad(rep)
            out = self.conv_classifier(rep)

            return out, c_logit, heatmap
        
        else:
            out = self.model(x) 
            return out


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), 
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

        if self.use_cam:
            y_hat, logits, heatmap = self.forward(img)

            if 1 + batch_idx % 100 == 0:
                if not os.exists("heatmaps"):
                    os.makedirs("heatmaps")
                plt.savefig(f"heatmaps/{self.config['model']}_{batch_idx}.png", heatmap.squeeze(0).cpu().numpy())

            cam_loss = self.cam_loss(logits, y)
            loss = self.ce_loss(y_hat, y) + cam_loss
        else:
            y_hat = self.forward(img)
            loss = self.ce_loss(y_hat, y)
            preds = torch.argmax(y_hat, dim=1)

        # DONE_TODO: Log train metrics here
        train_step_acc = self.calculate_accuracy(preds, y) # DONE_TODO: Calculate train accuracy here 
        self.log("train_acc", train_step_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss


    def on_validation_epoch_end(self) -> None:
        class_names = ['authentic', 'copy-moved', 'spliced']
        
        cm = wandb.plot.confusion_matrix(y_true=self.val_gt, preds=self.val_preds, class_names=class_names)
        self.logger.experiment.log({"conf_mat": cm})

        self.log("precision", precision_score(self.val_gt, self.val_preds, average='weighted'))
        self.log("f1-score", f1_score(self.val_gt, self.val_preds, average='weighted'))     
        self.log("recall", recall_score(self.val_gt, self.val_preds, average='weighted'))
        self.log("accuracy", accuracy_score(self.val_gt, self.val_preds))

        self.val_gt = []
        self.val_preds = []

    
    def validation_step(self, batch, batch_idx):
        img = batch['image'].to(self.device)
        # mask = batch['mask'] # Not Needed in classification setting
        y = batch['class_label'].to(self.device)
        # print(f"{i+1} | {img.shape} | {y}")

        y_hat = self.forward(img)
        preds = torch.argmax(y_hat, dim=1)
        test_acc = self.calculate_accuracy(preds, y) 

        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # DONE_TODO: Add misclassified images to the logger for one batch
        if (batch_idx + 1) % 10 == 0:
            id2cat = {'0': 'authentic', '1': 'copy-moved', '2': 'spliced'}
            caption_strs = []
            for i in range(1):
                correct = "Misclassified" if preds[i] != y[i] else "Correct"
                caption_strs.append(f"Pred: {id2cat[str(preds[i].item())]}, Label: {id2cat[str(y[i].item())]} | {correct}")

            self.logger.log_image(
                key=f"Validation Batch: {batch_idx + 1}",
                images=[img[i] for i in range(1)],
                caption=caption_strs,
            )
        self.val_preds.extend(preds.cpu().numpy())
        self.val_gt.extend(y.cpu().numpy())


    def on_test_epoch_end(self) -> None:
        class_names = ['authentic', 'copy-moved', 'spliced']
        cm = confusion_matrix(self.test_gt, self.test_preds)
        print("conf_mat", cm)

        print("precision", precision_score(self.test_gt, self.test_preds, average='weighted'))
        print("f1-score", f1_score(self.test_gt, self.test_preds, average='weighted'))     
        print("recall", recall_score(self.test_gt, self.test_preds, average='weighted'))
        print("accuracy", accuracy_score(self.test_gt, self.test_preds))

        self.test_gt = []
        self.test_preds = []


    def test_step(self, batch, batch_idx):  
        # NOTE: Same as validation loop minus the image logging
        # No image logging so that export can be done easily

        img = batch['image'].to(self.device)
        # mask = batch['mask'] # Not Needed in classification setting
        y = batch['class_label'].to(self.device)
        # print(f"{i+1} | {img.shape} | {y}")

        y_hat = self.forward(img)
        preds = torch.argmax(y_hat, dim=1)
        test_acc = self.calculate_accuracy(preds, y)

        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.test_gt.extend(y.cpu().numpy())
        self.test_preds.extend(preds.cpu().numpy())


def get_config(args):
    config = {
        'seed': 42,
        'model': args.model,
        'mode': args.mode,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device': args.device,
        'num_workers': args.num_workers,
        'T_0': 50,
        'eta_min': 5e-4,
        'classes': 3,
        'decay': 1e-3,
        'exp_name': args.exp_name,
        'use_cam': args.use_cam
    }
    return config


if __name__ == '__main__':
    args = get_args()
    # print("Total devices:", torch.cuda.device_count())

    check_args(args) # will also set args.device properly
    config = get_config(args)
    seed_everything(42)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(args.device)
    else:
        device_name = 'cpu'
    print("--------------------------------------------\nSelected device:", device_name,"\n--------------------------------------------")
    print(f"[+] Model Selected: {config['model']}")
    
    model = ShallowCNN()

    lit_model = LIT_CNN(model, config['model'], config)
    
    train_dataloader, test_dataloader = get_dataloaders(batch_size=config['batch_size'], 
                                                        num_workers=config['num_workers'])

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="test_acc", 
                                          mode="max", 
                                          save_top_k=1, 
                                          dirpath="checkpoints/", 
                                          filename=f"{config['model']}" + "_{test_acc:.3f}")
    
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
    # early_stop_callback = EarlyStopping(monitor="loss", patience=99)

    wandb.login()
    
    if config['exp_name'] == 'generic_exp':
        fnam = f"{config['model']}_GE"
    else:
        fnam = config['exp_name']

    wandb_logger = WandbLogger(project='forgery_detection',
                           name=f"{fnam}",
                           config=config,
                           job_type='finetuning',
                           log_model="all")
    # call trainer
    trainer = Trainer(fast_dev_run=False,
                      inference_mode=False,  # to enable grad enabling during inference
                      max_epochs=config['epochs'],
                      accelerator="gpu" if "cuda" in config['device'] else "cpu",
                      devices=[int(config['device'].split(":")[-1])], # GPU ID that you selected
                      precision="16", # automatic mixed precision training
                      deterministic=True,
                      enable_checkpointing=True,
                      callbacks=[checkpoint_callback, lr_monitor],
                      gradient_clip_val=None, 
                      log_every_n_steps=50,
                      logger=wandb_logger, # The absolute best: wandb <3
                      enable_progress_bar=True)

    # fit model
    if config['mode'] == 'train' or config['mode'] == 'trainX': # TODO: Implement trainX mode
        trainer.fit(lit_model, train_dataloader, test_dataloader)
    else:
        # DONE_TODO: Load last checkpoint and test
        if not os.exists(args.ckpt_path):
            args.ckpt_path = checkpoint_callback.best_model_path
        lit_model = LIT_CNN.load_from_checkpoint(args.ckpt_path)
        lit_model.freeze()
        trainer.test(lit_model, test_dataloader)
    
    wandb.finish()
