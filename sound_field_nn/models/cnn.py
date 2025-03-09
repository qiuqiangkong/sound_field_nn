import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
import numpy as np
from dataclasses import dataclass


class Cnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.pre_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv_block1 = ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5))
        self.conv_block2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5))
        self.conv_block3 = ConvBlock(in_channels=32, out_channels=32, kernel_size=(5, 5))
        self.post_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        
    def forward(self, x):
        """Separation model.

        Args:
            x: (b, t, x, y)

        Outputs:
            output: (b, t, x, y)
        """

        B = x.shape[0]

        x = rearrange(x, 'b t w h -> (b t) 1 w h')  # shape: (b*t, w, h)
        x = self.pre_conv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.post_conv(x)
        x = rearrange(x, '(b t) 1 w h -> b t w h', b=B)  # shape: (b, t, w, h)

        return x



class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size):
        r"""Residual block."""
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn = nn.BatchNorm2d(in_channels)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, freq_bins)

        Returns:
            output: (batch_size, out_channels, time_steps, freq_bins)
        """
        x = self.conv(F.leaky_relu_(self.bn(x)))
        return x