from src.data.base_datamodule import BaseDataModule
from src.data.datasets import ImageNet100


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
