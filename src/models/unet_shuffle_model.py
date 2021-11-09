from typing import Any, List

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from pytorch_lightning import LightningModule

# from torchmetrics.image.psnr import PSNR
from torchmetrics.image.ssim import SSIM
from torchmetrics.functional import psnr, ssim

# from src.losses import build_losses

class UNetShuffleLitModel(LightningModule):
    """
    UNet shuffle model
    """

    def __init__(
        self,
        in_c: int = 3,
        out_c: int = 3,
        num_feat: int = 16,
        scale: int = 2,

        lr: float = 1e-4,
        weight_decay: float = 0.0005,
        

    ):
        super().__init__()

        self.save_hyperparameters()

        feat_list = [
            num_feat,
            num_feat*(scale*scale),
            num_feat*(scale*scale)*(scale*scale),
            num_feat*(scale*scale)*(scale*scale)*(scale*scale),
            num_feat*(scale*scale)*(scale*scale)*(scale*scale)*(scale*scale),
            ]
        self.conv_in = nn.Conv2d(in_channels=in_c, out_channels=num_feat, kernel_size=3, stride=1, padding=1)
        

        self.down1 = nn.PixelUnshuffle(scale)
        self.conv1 = nn.Conv2d(in_channels=feat_list[1], out_channels=feat_list[1], kernel_size=3, stride=1, padding=1)
        
        self.down2 = nn.PixelUnshuffle(scale)
        self.conv2 = nn.Conv2d(in_channels=feat_list[2], out_channels=feat_list[2], kernel_size=3, stride=1, padding=1)
        
        self.down3 = nn.PixelUnshuffle(scale)
        self.conv3 = nn.Conv2d(in_channels=feat_list[3], out_channels=feat_list[3], kernel_size=3, stride=1, padding=1)
        
        self.down4 = nn.PixelUnshuffle(scale)
        self.conv4 = nn.Conv2d(in_channels=feat_list[4], out_channels=feat_list[4], kernel_size=3, stride=1, padding=1)
        
        self.up1 = nn.PixelShuffle(scale)
        self.up2 = nn.PixelShuffle(scale)
        self.up3 = nn.PixelShuffle(scale)
        self.up4 = nn.PixelShuffle(scale)
        
        self.conv_out = nn.Conv2d(in_channels=num_feat, out_channels=out_c, kernel_size=3, stride=1, padding=1)

        self.loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.conv1(x)
        x = self.down2(x)
        x = self.conv2(x)
        x = self.down3(x)
        x = self.conv3(x)
        x = self.down4(x)
        x = self.conv4(x)
        x = self.up4(x)
        x = self.conv3(x)
        x = self.up3(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv_out(x)
        return x

    def _shared_step(self, batch:Any, batch_idx:int, prefix:str):
        s, g, m = batch
        d = self(s)
        if prefix == 'test':
            d = d.clamp(0., 1.)

        # Losses
        mse_loss = self.loss(d, g)
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
