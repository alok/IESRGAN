import glob
import random
from pathlib import Path
from typing import List

import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image


def expand_path(s: str) -> Path:
    return Path(s).expanduser().resolve(strict=True)


def get_images(s: str) -> List[Path]:
    return list(expand_path(s).glob("*.png"))


class DatasetFromFolder(data.Dataset):
    def __init__(self, mode, config):

        super().__init__()
        if mode == "train":
            self.hr_images = get_images(config["PATH_TO_TRAIN_HR_DATA"])
            self.lr_images = get_images(config["PATH_TO_TRAIN_LR_DATA"])
        else:
            self.hr_images = get_images(config["PATH_TO_VALID_HR_DATA"])
            self.lr_images = get_images(config["PATH_TO_VALID_LR_DATA"])

        assert len(self.hr_images) == len(
            self.lr_images
        ), "Count HR images must be equal count LR images!"
        assert [x.split("/")[-1] for x in self.hr_images] == [
            x.split("/")[-1] for x in self.hr_images
        ], "List HR images must be equal List LR images!"

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_images[index])
        lr_image = Image.open(self.lr_images[index])
        lr_image, hr_image = self.transform(lr_image, hr_image)
        return lr_image, hr_image

    @staticmethod
    def transform(image, mask):
        # Random horizontal flipping
        if random.random() > 0.5:

            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self) -> int:
        return len(self.hr_images)
