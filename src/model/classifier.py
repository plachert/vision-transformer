from typing import Any
import lightning as L
import torch.nn as nn
import torch


class ImageClassifier(L.LightningModule):
    def __init__(
            self,
            feature_extractor: nn.Module, 
            classification_head: nn.Module,
            ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        self.loss = nn.CrossEntropyLoss()

    def forward(self, image):
        features = self.feature_extractor(image)
        preds = self.classification_head(features)
        return preds
    
    def _step(self, batch):
        image, targets = batch['image'], batch['class_idx']
        preds = self(image)
        loss = self.loss(preds, targets)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch)
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch)
        self.log('val/loss', loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
