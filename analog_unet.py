""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

import spikingjelly
from spikingjelly.activation_based import layer, neuron, functional
from core.models import SpCBAMmodule
from core.models.analog_unet_cbam import *
from core.models.spViT_block import ViTBlock

__all__ = ['AnalogUNet', 'get_analog_unet']


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            neuron.IFNode(),
            layer.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(mid_channels),
            neuron.IFNode(),
            layer.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            layer.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            from functools import partial
            # self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='bilinear', align_corners=True)
            self.up = partial(nn.functional.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
            # self.up = layer.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # self.act = neuron.IFNode()
            self.up = layer.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x2_shape = x2.shape
        x2 = torch.reshape(x2, shape=(-1, *x2_shape[-3:]))
        # x1 = self.act(x1)
        x1 = self.up(x1)

        x1_shape = x1.shape
        x1 = torch.reshape(x1, shape=(-1, *x1_shape[-3:]))
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x_shape = x.shape
        x = torch.reshape(x, shape=(*x1_shape[:-3], *x_shape[-3:]))
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.sn = neuron.IFNode()
        self.conv = layer.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(self.sn(x))


class AnalogUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False, time_step=4, **kwargs):
        super(AnalogUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.T = time_step
        self.inc = nn.Sequential(
            layer.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.IFNode(),
            layer.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64)
        )
        # self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.module1 = AnalogCBAM(128)
        self.down2 = Down(128, 256)
        # self.module3 = ViTBlock(in_channels=256, out_channels=256, patch_size=1, num_heads=4, mlp_ratio=2)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.module2 = ViTBlock(in_channels=1024 // factor, out_channels=1024 // factor, patch_size=1, num_heads=8, mlp_ratio=2)
        # self.module3 = ViTBlock(in_channels=512, out_channels=1024 // factor, patch_size=1, num_heads=8, mlp_ratio=2)
        self.up1 = Up(1024 // factor, 512 // factor, bilinear)
        self.up2 = Up(512 // factor, 256 // factor, bilinear)
        self.up3 = Up(256 // factor, 128 // factor, bilinear)
        self.up4 = Up(128 // factor, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # self.__setattr__('exclusive', ['head', 'auxlayer'])
        self.__setattr__('exclusive', [
            'inc', 'down1',
            'module1',
            'down2', 'down3', 'down4',
            'module2',
            'up1', 'up2', 'up3', 'up4', 'outc'
        ])

    def forward(self, x):
        functional.reset_net(self)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.module1(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.module2(x5)
        # x5 = self.module3(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # print(logits.shape)
        logits = torch.mean(logits, 0)
        # print(logits.shape)
        return tuple([logits])


def get_analog_unet(dataset='citys', **kwargs):
    from ..data.dataloader import datasets
    model = AnalogUNet(n_classes=datasets[dataset].NUM_CLASS, step_mode=True, **kwargs)
    return model
