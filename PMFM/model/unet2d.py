from typing import Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import nn
from utils.tensor_to_png import TensorToPng


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, isGroupNorm=False, num_gropus=32):

        super().__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.num_gropus, mid_channels) if self.isGroupNorm else nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.num_gropus, out_channels) if self.isGroupNorm else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, isGroupNorm=False, num_gropus=32):
        '''
            Down模块包括下采样(MaxPool) + 2个Conv2d,因为在网络左侧encoder部分基本上都是通过下采样(MaxPool) + 2个Conv2d搭建的。
        :param in_channels:
        :param out_channels:
        '''
        super().__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, isGroupNorm=False, num_gropus=32):


        super(Up, self).__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)

    def forward(self, x1, x2):

 
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)




class PointNet(nn.Module):
    def __init__(self, conv_inch=512, ext=False, isGroupNorm=False, num_gropus=32):
        super().__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        if ext:
            self.conv1 = nn.Sequential(
                nn.Conv2d(conv_inch, conv_inch * 2, kernel_size=3, padding=1, stride=1, bias=False),
                nn.GroupNorm(self.num_gropus, conv_inch * 2) if self.isGroupNorm else nn.BatchNorm2d(conv_inch * 2),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(conv_inch * 2, conv_inch, kernel_size=3, padding=1, stride=1, bias=False),
                nn.GroupNorm(self.num_gropus, conv_inch) if self.isGroupNorm else nn.BatchNorm2d(conv_inch),
                nn.ReLU(inplace=True)
            )
        self.final_conv = nn.Sequential(
            nn.Conv2d(conv_inch, 3, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self._ext = ext
        self.dwconv = nn.Conv2d(conv_inch, conv_inch, kernel_size=1, stride=1, padding=1)



    def forward(self, input):
        """
        :param input: (batch_size, channel, w, h)
        :return: (batch_size, channel, 3)
        """
        x = input
        if self._ext:
            x = self.conv1(x)
            x = self.conv2(x)
        x = self.final_conv(x)  # (batch_size, 3, w, h)
        x = x.view(x.size(0), -1, x.size(1))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, bilinear: bool = True, base_c: int = 64, pointnet=False, extpn=False, is_point=False, isGroupNorm=False, num_gropus=32):

        super(UNet, self).__init__()
        self.is_point = is_point
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self._pointnet = pointnet
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        self.in_conv = DoubleConv(in_channels, base_c, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down1 = Down(base_c, base_c * 2, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down2 = Down(base_c * 2, base_c * 4, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down3 = Down(base_c * 4, base_c * 8, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        factor = 2 if bilinear else 1
        if pointnet:
            self.pointNet = PointNet(conv_inch=base_c, ext=extpn, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down4 = Down(base_c * 8, base_c * 16 // factor, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.up4 = Up(base_c * 2, base_c, bilinear, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        point_x = x
        out_point = None
        if self._pointnet:
            out_point = self.pointNet(point_x)
        logits = self.out_conv(x)
        if self.is_point:
            return logits, out_point, point_x, logits   # 最后一个point_x用于PFF
        else:
            return logits

    def use_checkpointing(self):
        self.in_conv = torch.utils.checkpoint(self.in_conv)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.out_conv = torch.utils.checkpoint(self.out_conv)

