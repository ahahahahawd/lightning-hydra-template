import os

import pytest
import torch

from src.datamodules.snow_datamodule import PairedDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_csd_datamodule(batch_size):
    datamodule = PairedDataModule(batch_size=batch_size)
    datamodule.prepare_data()

    assert not datamodule.data_train and not datamodule.data_val and not datamodule.data_test

    assert os.path.exists(os.path.join("data", "MNIST"))
    assert os.path.exists(os.path.join("data", "MNIST", "raw"))

    datamodule.setup()

    assert datamodule.data_train and datamodule.data_val and datamodule.data_test
    assert (
        len(datamodule.data_train) + len(datamodule.data_val) + len(datamodule.data_test) == 70_000
    )

    assert datamodule.train_dataloader()
    assert datamodule.val_dataloader()
    assert datamodule.test_dataloader()

    batch = next(iter(datamodule.train_dataloader()))
    x, y = batch

    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
