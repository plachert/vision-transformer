from src.data.base_datamodule import BaseDataModule
from src.data.datasets import ImageNet100, CIFAR10
import torch
import lightning as L


class ImageNet100DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def setup(self, stage: str):
        self.train_ds = ImageNet100(
            transform=self.train_transform,
            is_train=True,
            )
        self.val_ds = ImageNet100(
            transform=self.inference_transform,
            is_train=False,
            )
        self.test_ds = None
        
        
class CIFAR10DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def setup(self, stage: str):
        """Split based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html"""
        if stage == "fit":
            train_ds = CIFAR10(train=True, transform=self.train_transform)
            val_ds = CIFAR10(train=True, transform=self.inference_transform)
            L.seed_everything(42)
            self.train_ds, _ = torch.utils.data.random_split(train_ds, [45000, 5000])
            L.seed_everything(42)
            _, self.val_ds = torch.utils.data.random_split(val_ds, [45000, 5000])
        else:
            self.test_ds = CIFAR10(train=False, transform=self.inference_transform)

