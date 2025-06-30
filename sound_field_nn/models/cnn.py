import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.pre_layer = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=2)
        self.conv1 = ConvBlock(in_channels=32, out_channels=32, kernel_size=5)
        self.conv2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=5)
        self.conv3 = ConvBlock(in_channels=32, out_channels=32, kernel_size=5)
        self.post_layer = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)
        
    def forward(self, x: Tensor) -> Tensor:
        """Model.

        b: batch_size
        c: channels_num
        h: height
        w: weight

        Args:
            x: (b, c, h, w)

        Outputs:
            output: (b, c, h, w)
        """

        x = self.pre_layer(x)  # (b, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.post_layer(x)  # (b, c, h, w)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        r"""Conv block"""
        super(ConvBlock, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        b: batch_size
        c: channels_num
        h: height
        w: weight

        Args:
            x: (b, c, h, w)

        Returns:
            output: (b, c, h, w)
        """
        out = self.conv(F.leaky_relu_(self.bn(x)))
        return out