from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser

from src.models.unet_shuffle_model import UNetShuffleLitModel

from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)

def predict(args):
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = args.ckpt

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    # trained_model = UNetShuffleLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)
    trained_model: LightningModule = hydra.utils.instantiate(config.model)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    # img = Image.open(args.img_path).convert("RGB")  # convert to black and white
    img = Image.open(args.img_path).convert("RGB")  # convert to RGB

    # preprocess
    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    img = img_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img).clamp(0, 1)  # clamp to [0, 1]
    img_out = transforms.ToPILImage()(output.squeeze(0))
    img_out.save(args.out_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--img_path', default=None, type=str, help="image path to predict")
    parser.add_argument('--ckpt', default=None, type=str, help="model ckpt to load")
    parser.add_argument('--out_path', default=None, type=str, help="image path to save")
    
    args = parser.parse_args()
    predict(args)