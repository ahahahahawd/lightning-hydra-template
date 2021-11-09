import os
import pathlib
from typing import Tuple
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split, Dataset

import pytorch_lightning as pl

from src.datamodules.datasets.dataset import PairedSnowDataset
from src.datamodules.datasets.tsfm import tsfmTrainFakeSnow, tsfmTestFakeSnow, tsfmTrainNoiseFakeSnow
from src.datamodules.datasets.utils import get_path_by_name




class PairedDataModule(pl.LightningDataModule):
    """
    PairedDataModule for image snow remove

    Args::
        data_dir: 'data/', short name for Snow100K dataset or root dir of the images
        batch_size: 16, 
        test_ds: 'fake', choose fake or real dataset when using dm.setup('test')
    Example::
        from src.data.snow import SnowDataModule
        dm = SnowDataModule(
            batch_size=16,
            data_dir='all')
        dm.setup('fit')
    """

    def __init__(
        self,
        # data_dir: str = 'data/train',
        train_dir: str = 'data/train',
        test_dir: str = 'data/test',
        subdirs: list = ['synthetic', 'gt', 'mask'],
        train_val_split: float = 0.7,
        seed: int = 2020,
        batch_size: int = 16,
        # test_ds='fake',
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        #self.data_dir = get_path_by_name(data_dir)

        self.tsfm_train = tsfmTrainFakeSnow()
        self.tsfm_val = tsfmTrainFakeSnow()
        self.tsfm_test = tsfmTestFakeSnow()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            dt = PairedSnowDataset(
                self.hparams.train_dir,
                self.hparams.subdirs,
                self.tsfm_train,
            )
            dt_len = len(dt)
            train_dt_len = int(self.hparams.train_val_split * dt_len)
            val_dt_len = dt_len - train_dt_len
            print('all   dt length: {}'.format(dt_len))
            print('train dt length: {}'.format(train_dt_len))
            print('val   dt length: {}'.format(val_dt_len))
            self.train_dt, self.val_dt = random_split(
                dt, [train_dt_len, val_dt_len],
                generator=torch.Generator().manual_seed(self.hparams.seed)
            )
        if stage == 'test' or stage is None:
            dt = PairedSnowDataset(
                self.hparams.test_dir,
                self.hparams.subdirs,
                self.tsfm_test,
            )
            test_dt_len = len(dt)
            print('test dt length: {}'.format(test_dt_len))
            self.test_dt = dt

    def train_dataloader(self):
        return DataLoader(
            self.train_dt,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dt, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False, 
            )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dt,
            batch_size=self.hparams.batch_size, 
            num_workers=0, 
            pin_memory=self.hparams.pin_memory,
            shuffle=False, 
            )


if __name__ == '__main__':
    dm = PairedDataModule(
        train_dir=get_path_by_name('all'),
        test_dir=get_path_by_name('S'),
        batch_size=4,
    )
    dm.prepare_data()
    dm.setup('fit')
    assert len(dm.train_dt) > 0
    print(len(dm.train_dt))