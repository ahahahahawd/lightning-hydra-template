import os
from PIL.Image import Image
import numpy as np

import pytest
import torch

from torchvision import transforms
from src.datamodules.datasets.tsfm import tsfmTrainSnow, tsfmValSnow, tsfmTestSnow


@pytest.mark.parametrize("crop_size", [64,128,256])
def test_train_snow_tsfm(crop_size):
    tsfm_train_snow = tsfmTrainSnow(crop_size=crop_size)

    x = np.random.rand(480,640,3)
    inputs = [x*3]
    outputs = tsfm_train_snow(inputs)

    assert len(outputs) == len(inputs)
    assert outputs[0].shape == (3, crop_size, crop_size)
    assert type(outputs[0]) == torch.Tensor

@pytest.mark.parametrize("crop_size", [64,128,256])
def test_val_snow_tsfm(crop_size):
    tsfm_val_snow = tsfmValSnow(crop_size=crop_size)

    x = np.random.rand(480,640,3)
    inputs = [x*3]
    outputs = tsfm_val_snow(inputs)

    assert len(outputs) == len(inputs)
    assert outputs[0].shape == (3, crop_size, crop_size)
    assert type(outputs[0]) == torch.Tensor

# @pytest.mark.parametrize("resize_size", [64,128,256])
# def test_test_snow_tsfm(resize_size):
#     tsfm_test_snow = tsfmTestSnow(resize_size=resize_size)

#     x = np.random.rand(480,640,3)
#     img = Image. fromarray(x.astype(np.uint8))
#     outputs = tsfm_test_snow(img)

#     assert outputs.shape == (3, resize_size, resize_size)
#     assert type(outputs) == torch.Tensor