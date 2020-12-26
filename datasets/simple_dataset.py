from os.path import join as ospj

import torch.utils.data as data
from PIL import Image


class Multipiece(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform) -> None:
        super().__init__()
        self.images_dir = ospj(data_root, dataset_name, 'images')
        self.transform = transform

    def __getitem__(self, index: int):
        image_path = self.images_dir[index]
        image = Image.open(image_path).convert('RGB')
        return {
            'image': self.transform(image)
        }
