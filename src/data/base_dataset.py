from typing import Callable
from torch.utils.data import Dataset
from abc import ABC
from abc import abstractproperty
import pathlib
import numpy as np
from PIL import Image


class ImageClassificationDataset(Dataset, ABC):

    def __init__(self, transform: Callable | None = None) -> None:
        super().__init__()
        self.transform = transform

    @abstractproperty
    def images(self) -> list[pathlib.Path]:
        """Return paths of images."""
        
    @abstractproperty
    def labels(self) -> np.ndarray:
        """Return labels in range 0:no_classes-1"""

    @property
    def no_classes(self):
        return np.max(self.labels) + 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self._load_image(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': label}

    def _load_image(self, path: pathlib.Path) -> np.ndarray:
        """Load image from path."""
        image = Image.open(path).convert('RGB')
        return image
