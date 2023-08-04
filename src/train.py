from model.classifier import ImageClassifier
from src.data.datamodules import ImageNet100DataModule
from torchvision import models, transforms
import cv2
from functools import partial
import lightning as L
from torch.optim import Adam
import torch.nn as nn
from lightning.pytorch import loggers as pl_loggers
import numpy as np


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
inference_transform = train_transform

def main():    
    datamodule = ImageNet100DataModule(
        train_transform=train_transform, inference_transform=inference_transform,
    )
    m = models.mobilenet_v2(pretrained=True)
    for param in list(m.features.parameters()):
        param.requires_grad = False
    m.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 100)
)
    model = ImageClassifier(
        model=m,
        optimizer_factory=partial(Adam, lr=0.001),
        num_classes=100,
        )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/')
    trainer = L.Trainer(max_epochs=100, accelerator='gpu', logger=tb_logger)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()