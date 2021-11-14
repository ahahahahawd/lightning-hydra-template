import os
from typing import Tuple
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader, random_split, Dataset

import pytorch_lightning as pl
from torchvision import transforms

from .tsfm import tsfmTrainFakeSnow, tsfmValFakeSnow, tsfmTestFakeSnow
from .utils import get_image_path_list


class PairedSnowDataset(Dataset):
    """
    Paired fake snow dataset for train and test.

    Args:
        data_dir: the root dir of 'synthetic', 'gt', 'mask' images
        subdirs: the name of subdirs in data_dir like ['synthetic', 'gt', 'mask'].
        transform: the same image transform to the 'synthetic', 'gt', 'mask' images

    """

    def __init__(
        self,
        data_dir: str = 'data/',
        subdirs: Tuple[str, str, str] = ('synthetic', 'gt', 'mask'),
        transform=tsfmTrainFakeSnow(),
    ):
        self.transform = transform
        self.images_synthetic, self.images_gt, self.images_mask = map(
            lambda x: get_image_path_list(os.path.join(data_dir, x)), subdirs)
        assert len(self.images_synthetic) == len(self.images_gt) == len(self.images_mask)

    def __getitem__(self, idx):
        # open image
        # img_synthetic, img_gt, img_mask = map(
        #     lambda x: np.array(Image.open(x[idx])), [self.images_synthetic, self.images_gt, self.images_mask])

        # # transform using albumentations
        # img_synthetic, img_gt, img_mask = self.transform([img_synthetic, img_gt, img_mask])

        # transform using albumentations
        img_synthetic, img_gt = self.transform([
            map(
                lambda x: np.array(Image.open(x[idx])), [self.images_synthetic, self.images_gt]
                )
            ])

        return img_synthetic, img_gt, None

    def __len__(self):
        return len(self.images_synthetic)


class SingleSnowDataset(Dataset):
    """
    real or single dir snow images dataset

    Args::
        data_dir: the root dir of snow images
        transform: the transform to the snow images

    """

    def __init__(
        self,
        data_dir: str = 'data/',
        transform=transforms.ToTensor(),
    ):
        self.transform = transform
        self.images_real = get_image_path_list(data_dir)

    def __getitem__(self, idx):
        img_real = Image.open(self.images_real[idx])
        img_real = self.transform(img_real)
        return img_real

    def __len__(self):
        return len(self.images_real)


if __name__ == '__main__':
    pass
    # dm = PairedDataModule(
    #     train_dir=get_path_by_name('all'),
    #     test_dir=get_path_by_name('S'),
    #     batch_size=4,
    # )
    # dm.prepare_data()
    # dm.setup('fit')
    # assert len(dm.train_dt) > 0
    # print(len(dm.train_dt))
