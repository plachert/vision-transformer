from typing import Callable

import lightning as L
from torch.utils.data import DataLoader


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_transform: Callable | None = None,
        inference_transform: Callable | None = None,
        train_batch_size: int = 256,
        val_batch_size: int = 32,
        test_batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.train_transform = train_transform
        self.inference_transform = inference_transform
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
