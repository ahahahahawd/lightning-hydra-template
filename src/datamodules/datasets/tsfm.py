import pathlib
import albumentations as A
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class tsfmTrainFakeSnow(object):
    def __init__(self,
                 resize_size=None,
                 crop_size=None,
                ):
#         self.resize_size = resize_size
#         self.crop_size = crop_size
        pass

    def __call__(self, x1, x2, x3):
        tsfm = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale((0.3, 1.75), p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.RandomCrop(width=128, height=128),
        ])

        tsfm_res = tsfm(image=x1, masks=[x2, x3])
        tsfm_synthetic = tsfm_res['image']
        tsfm_gt = tsfm_res['masks'][0]
        tsfm_mask = tsfm_res['masks'][1]

        tsfm_synthetic, tsfm_gt, tsfm_mask = map(
            lambda x: transforms.ToTensor()(x), [tsfm_synthetic, tsfm_gt, tsfm_mask]
        )
        return tsfm_synthetic, tsfm_gt, tsfm_mask


class tsfmTrainNoiseFakeSnow(object):
    def __init__(self,
                 resize_size=None,
                 crop_size=None,
                ):
#         self.resize_size = resize_size
#         self.crop_size = crop_size
        pass

    def __call__(self, x1, x2, x3):
        tsfm_noise = A.Compose([
            A.GaussNoise(),
            #A.GaussianBlur(),
            #A.RandomBrightnessContrast(),
        ])
        tsfm_res = tsfm_noise(image=x1)
        x1 = tsfm_res['image']

        tsfm = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale((0.3, 1.75), p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.RandomCrop(width=128, height=128),
        ])

        tsfm_res = tsfm(image=x1, masks=[x2, x3])
        tsfm_synthetic = tsfm_res['image']
        tsfm_gt = tsfm_res['masks'][0]
        tsfm_mask = tsfm_res['masks'][1]

        tsfm_synthetic, tsfm_gt, tsfm_mask = map(
            lambda x: transforms.ToTensor()(x), [tsfm_synthetic, tsfm_gt, tsfm_mask]
        )
        return tsfm_synthetic, tsfm_gt, tsfm_mask


class tsfmTestFakeSnow(object):
    def __init__(self, resize_size=None, crop_size=None):
        self.resize_size = resize_size
        self.crop_size = crop_size
        pass

    def __call__(self, x1, x2, x3):
        # Resize
        if self.resize_size is not None:
            resize = transforms.Resize(size=self.resize_size)
            x1, x2, x3 = map(lambda x: resize(x), [x1, x2, x3])

        # Transform to tensor
        x1, x2, x3 = map(lambda x: TF.to_tensor(x), [x1, x2, x3])

        return x1, x2, x3


class tsfmTestRealSnow(object):
    def __init__(self, resize_size=None, crop_size=None):
        self.resize_size = resize_size
        self.crop_size = crop_size
        pass

    def __call__(self, x):
        # Resize
        if self.resize_size is not None:
            x = transforms.Resize(self.resize_size)(x)

        # Random crop
        if self.crop_size is not None:
            x = transforms.RandomCrop(self.crop_size)(x)

        # Transform to tensor
        x = TF.to_tensor(x)

        return x

