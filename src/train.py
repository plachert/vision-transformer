from model.classifier import ImageClassifier
from src.data.datamodules import ImageNet100DataModule, CIFAR10DataModule
from torchvision import models, transforms
import cv2
from functools import partial
import lightning as L
from torch.optim import Adam
import torch.nn as nn
from lightning.pytorch import loggers as pl_loggers
import numpy as np
from src.data.transform import Patchify
from src.model.architectures.visual_transformer import VisionTransformer


inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Patchify((16, 16)),
    ]
    )
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((224, 224), scale=(0.8,1.0), ratio=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Patchify((16, 16)),
    ]
    )


def main():    
    # datamodule = CIFAR10DataModule(
    #     train_batch_size=128,
    #     val_batch_size=128,
    #     train_transform=train_transform, 
    #     inference_transform=inference_transform
    # )
    datamodule = ImageNet100DataModule(
        train_batch_size=128,
        val_batch_size=128,
        train_transform=train_transform, 
        inference_transform=inference_transform
    )
    
    #####
#     m = models.mobilenet_v2(pretrained=True)
#     for param in list(m.features.parameters()):
#         param.requires_grad = False
#     m.classifier = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(1280, 100)
# )
    m = VisionTransformer(
        emb_dim=256, 
        flatten_patch_dim=16*16*3, 
        num_blocks=6, 
        num_heads=8, 
        num_patches=196, 
        num_classes=100, 
        dim_feedforward=512,
        dropout=0.2,
        )
    #####
    model = ImageClassifier(
        model=m,
        optimizer_factory=partial(Adam, lr=0.001),
        num_classes=100,
        )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/imagenet100_vit_16')
    trainer = L.Trainer(max_epochs=100, accelerator='gpu', logger=tb_logger)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
