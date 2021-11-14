from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from pytorch_lightning import LightningModule

from torchmetrics.image.ssim import SSIM
from torchmetrics.functional import psnr, ssim

from src.models.modules.unet_parts import DoubleConv, Down, Up, OutConv, ResBlock


class UNetResRLitModel(LightningModule):
    """
    UNetResR
    """

    def __init__(
        self,
        in_c: int = 3,
        out_c: int = 3,
        # num_feat: int = 64,
        bilinear: bool = True,

        lr: float = 1e-4,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_c)

        self.res_block_1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
        )
        self.res_block_2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
        )
        self.res_block_3 = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        self.res_block_4 = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.res_block_1(x1)
        x2 = self.down1(x1)
        x2 = self.res_block_2(x2)
        x3 = self.down2(x2)
        x3 = self.res_block_3(x3)
        x4 = self.down3(x3)
        x4 = self.res_block_4(x4)
        x5 = self.down4(x4)
        x_ = self.up1(x5, x4)
        x_ = self.up2(x_, x3)
        x_ = self.up3(x_, x2)
        x_ = self.up4(x_, x1)
        x_ = self.outc(x_)
        logits = x + x_
        return logits

    def _shared_step(self, batch:Any, batch_idx:int, prefix:str):
        s, g, m = batch
        d = self(s)
        if prefix == 'test':
            d = d.clamp(0., 1.)

        # Losses
        mse_loss = nn.MSELoss()(d, g)
        ssim_loss = 1 - SSIM()(d, g)
        loss = ssim_loss + mse_loss

        # Metrics
        psnr_metric = psnr(d, g)
        ssim_metric = ssim(d, g)

        # Log losses
        self.log(f'{prefix}/loss/mse', mse_loss, on_epoch=True)
        self.log(f'{prefix}/loss/ssim', ssim_loss, on_epoch=True)
        self.log(f'{prefix}/loss/all', loss, on_epoch=True, prog_bar=True)

        # Log metrics
        self.log(f'{prefix}/metric/psnr', psnr_metric, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}/metric/ssim', ssim_metric, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=[200,300,350], gamma=0.1),
            'monitor': 'valid/loss/all',
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

