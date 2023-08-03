from typing import Callable
from torch.utils.data import Dataset
from abc import ABC
from abc import abstractproperty
import pathlib
import numpy as np
import cv2


class ImageClassificationDataset(Dataset, ABC):

    def __init__(self, transform: Callable | None = None) -> None:
        super().__init__()
        self.transform = transform

    @abstractproperty
    def images(self) -> list[pathlib.Path]:
        """Return paths of images."""

    @abstractproperty
    def class_mapping(self) -> list[str]:
        """Return classes of objects."""

    @property
    def no_classes(self):
        return len(self.classes)

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
        image_bgr = cv2.imread(path)[:3]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
