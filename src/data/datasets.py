from src.data.base_dataset import ImageClassificationDataset
import pathlib
import json
import numpy as np
from typing import Callable
import torchvision


PATHS = {
    "ImageNet100": pathlib.Path('/home/piotr/datasets/vision/imagenet_100'),
    "CIFAR10": pathlib.Path('/home/piotr/datasets/vision/cifar10'),
}


class ImageNet100(ImageClassificationDataset):
    
    """https://www.kaggle.com/datasets/ambityga/imagenet100"""
    
    def __init__(
        self, 
        is_train: bool,
        transform: Callable | None = None,
        path: pathlib.Path = PATHS["ImageNet100"], 
        ):
        super().__init__(transform)
        self.path = path
        self.train_or_val = "train" if is_train else "val"
        self._images = self._get_image_paths()
        self._labels = self._get_labels() # maps original id to a number (0-99)
        self.transform = transform
        
    def _get_image_paths(self):
        return list(self.path.glob(f'{self.train_or_val}*/*/*.JPEG'))
    
    def _get_labels(self):
        mapping = {}
        with open(self.path.joinpath("Labels.json"), "r") as file:
            label_mapping = json.load(file)
        for idx, (id_, class_name) in enumerate(label_mapping.items()):
            mapping[id_] = (idx, class_name)
        labels = np.zeros(len(self), dtype=int)
        for idx, path in enumerate(self.images):
            id_ = path.parent.name
            labels[idx] = mapping[id_][0]           
        return labels        
        
    @property
    def images(self) -> list[pathlib.Path]:
        return self._images
    
    @property
    def labels(self) -> np.ndarray:
        return self._labels


class CIFAR100(torchvision.datasets.CIFAR10):
    def __init__(self, root=PATHS["CIFAR10"], download=True, *args, **kwargs):
        super().__init__(root=root, download=download, *args, **kwargs)
        
    @property
    def no_classes(self):
        return 10
    
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return {"image": image, "label": label}
