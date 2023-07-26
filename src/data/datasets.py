from torch.utils.data import Dataset
import pathlib
from PIL import Image
import json
import numpy as np


class ImageNet100(Dataset):
    
    """https://www.kaggle.com/datasets/ambityga/imagenet100"""
    
    def __init__(self, path: pathlib.Path, is_train: bool):
        super().__init__()
        self.path = path
        self.train_or_val = "train" if is_train else "val"
        self.image_paths = self._get_image_paths()
        self.labels = self._create_mapping() # maps original id to a number (0-99)   
        
    def _get_image_paths(self):
        image_paths = list(self.path.glob(f'{self.train_or_val}*/*/*.JPEG'))
        return image_paths
            
    def _create_mapping(self):
        mapping = {}
        with open(self.path.joinpath("Labels.json"), "r") as file:
            labels = json.load(file)
        for idx, (id_, class_name) in enumerate(labels.items()):
            mapping[id_] = (idx, class_name)
        return mapping        
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        id_ = image_path.parent.name
        class_idx, class_name = self.labels[id_]
        return {"image": image, "class_idx": class_idx, "class_name": class_name}
        
if __name__ == "__main__":
    ds = ImageNet100(pathlib.Path('/home/piotr/datasets/vision/imagenet_100'), is_train=False)
    print(ds[0])