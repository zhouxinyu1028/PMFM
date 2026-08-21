import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, isGroupNorm=False, num_gropus=32):
        '''
            在网络结构中，卷积基本上都是成对使用的，所以就定义了一个DoubleConv类
            1. 通过父类的构造函数搭建DoubleConv,其中Conv2d的Kernel=3,padding=1,设置padding=1经过卷积后不会改变特征层的大小，这也是现在主流的实现方式
            2. 由于我们会使用到BN，因此将bias设置为Flase
        :param in_channels:指的是输入特征层的channels
        :param out_channels:指的是经过DoubleConv层后输出特征层的channels
        :param mid_channels:指第一个卷积层输出的channels
        '''
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
        '''
        上采样(w,h翻倍)+concat拼接+2个Conv2d, 在UNet网络的右半部分decoder也就是解码器部分，基本上都是有各个上采样+concat拼接+2个Conv2d模块组成，因此这里定义了一个UP模块。
        :param in_channels:指的是concat拼接后的channels,或者是up这个模块第一个卷积输入的channels :param out_channels: :param
        bilinear:输入参数bilinear=True,表示默认情况下是会使用双线性差值,如果bilinear=False,
        则使用论文中提到的转置卷积进行上采样，长宽翻倍，channels会减半，此时DoubleConv(in_channels,out_channels)
        中mid_channels和out_channels是一样的。因为在原论文中这两个卷积的channels是一样的.如果采用双线性插值进行上采样的话（经过双线性插值自身不会改变channels
        ），上采样后面跟着的两个卷积的channels是不一样的，比如通过第一个卷积后channels会减半，通过第二个卷积后，channels又会减半。这样做的目的是为了经过双线性插值后得到的channels
        和我们要concat拼接的特征层的channels保持一致。
        '''
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
        '''
        :param x1: 指的是需要上采样的特征层
        :param x2: 指的是要concat拼接的特征层
        :return:
        '''
        x1 = self.up(x1)
        # -----------------------------------------
        # 但是这里作者有多做了一步，对我们上采样之后的x1进行了padding,目的是为
        # 了防止我们输入的图片如果不是16的整数倍的话，通过下采样得到的x1与我们要拼接的x2的高度和宽度是不一致的
        # [N,C,H,W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # padding_left,padding_right,padding_top,padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # ------------------------------------------
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
            OutConv对应的是最后一个1x1的卷积层，通过这个1x1的卷积层之后就得到我们最终的输出了。这个1x1的卷积它是没有BN和ReLU激活函数的。
        :param in_channels:
        :param num_classes:
        '''
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
        """

        :param in_channels: 输入的图片如果是彩色的,in_channels=3,如果使用的是黑白的`in_channels=1 :param num_classes: :param
        bilinear:bilinear 默认是True,根据测试无论采用双线性插值还是采用转置卷积计算，他们的结果其实是差不多的。那么你采用双线性差值其实会更高效点。 :param
        base_c:就是网络中第一个卷积层输出的channels,在unet网络中各层的channels都是翻倍的，比如64,128,256,512,所以就定义了一个base
            channel，默认为64，也可以根据自己的想法去调整channels的大小。我这边训练的时候将base_c设置为32，发现得到的结果也没啥区别，但设置为32网络的参数会降低、训练速度会得到提升。
        """
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

