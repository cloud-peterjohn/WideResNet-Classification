import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl


class ResNetModel(pl.LightningModule):
    """
    Wide ResNet model for image classification with PyTorch Lightning.
    """
    def __init__(self, num_classes=100, label_smoothing=0.1):
        super().__init__()
        self.model = torchvision.models.wide_resnet50_2(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )
        return [optimizer], [scheduler]
