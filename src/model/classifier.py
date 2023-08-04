from typing import Any
import lightning as L
import torch.nn as nn
import torch
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric, MeanMetric


class ImageClassifier(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            optimizer_factory: torch.optim.Optimizer,
            num_classes: int, 
            ):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer_factory(self.parameters())
        self._setup_metrics(num_classes)

    def _setup_metrics(self, num_classes):
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, image):
        logits = self.model(image)
        return logits
    
    def _step(self, batch):
        image, targets = batch['image'], batch['label']
        logits = self(image)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, targets)
        return loss, preds, targets
    
    def on_train_start(self):
        """Reset after sanity checks."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch)
        self.train_acc(preds, targets)
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_epoch=True, on_step=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch)
        self.val_acc(preds, targets)
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, on_epoch=True, on_step=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer
