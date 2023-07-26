from torch.utils.data import Dataset
import pathlib
from PIL import Image
import json


class ImageNet100(Dataset):
    
    """https://www.kaggle.com/datasets/ambityga/imagenet100"""
    
    def __init__(self, path: pathlib.Path, is_train: bool):
        super().__init__()
        self.path = path
        self.is_train = is_train
        self.labels = self._load_labels()
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self):
        if self.is_train:
            pass
        else:
            path = self.path.joinpath('val.X')
            image_paths = path.glob('*/*.JPEG')
        return image_paths
            
    def _load_labels(self):
        with open(self.path.joinpath("Labels.json"), "r") as file:
            labels = json.load(file)
        return labels
        
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
        
if __name__ == "__main__":
    ds = ImageNet100(pathlib.Path('/home/piotr/datasets/vision/imagenet_100'), is_train=False)