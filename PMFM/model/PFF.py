
import numpy as np
import torch
import torch.nn as nn
from .pointnet import PointNetDenseCls
from utils.tensor_to_png import TensorToPng
import torch.nn.functional as F
from utils.show_pointCloud import show_pointCloud, point_distribution
from utils.image_to_pointCloud import ImageToPointCloud
import time



class PFFConv(nn.Module):
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


class PFFDown(nn.Module):
    def __init__(self, in_channels, out_channels, isGroupNorm=False, num_gropus=32):
        super().__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            PFFConv(in_channels, out_channels, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class PFFUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, isGroupNorm=False, num_gropus=32):
        super(PFFUp, self).__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = PFFConv(in_channels, out_channels, in_channels // 2, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = PFFConv(in_channels, out_channels, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class PFFOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PFFOutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class PFFBloack(nn.Module):
    """
        小型UNet结构
    """
    def __init__(self, in_channels: int = 30, num_classes: int = 2, bilinear: bool = True, base_c: int = 64, extpn=False, isGroupNorm=False, num_gropus=32):
        super(PFFBloack, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus

        self.in_conv = PFFConv(in_channels, base_c, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down1 = PFFDown(base_c, base_c * 2, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down2 = PFFDown(base_c * 2, base_c * 4, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.down3 = PFFDown(base_c * 4, base_c * 8, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        factor = 2 if bilinear else 1
        self.down4 = PFFDown(base_c * 8, base_c * 16 // factor, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.up1 = PFFUp(base_c * 16, base_c * 8 // factor, bilinear, isGroupNorm=self.isGroupNorm,
                      num_gropus=self.num_gropus)
        self.up2 = PFFUp(base_c * 8, base_c * 4 // factor, bilinear, isGroupNorm=self.isGroupNorm,
                      num_gropus=self.num_gropus)
        self.up3 = PFFUp(base_c * 4, base_c * 2 // factor, bilinear, isGroupNorm=self.isGroupNorm,
                      num_gropus=self.num_gropus)
        self.up4 = PFFUp(base_c * 2, base_c, bilinear, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.out_conv = PFFOutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits


class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
        # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out


# 版本二
class PFF(nn.Module):

    def __init__(self, num_classes: int = 30, isGroupNorm=False, num_gropus=32):
        super(PFF, self).__init__()
        self.pointNet_num_classes = 30
        self.num_classes = num_classes
        self.refining_point_factor = 0.25  # 细化因子
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        self.pointcls = PointNetDenseCls(k=self.pointNet_num_classes, isGroupNorm=self.isGroupNorm,
                                         num_gropus=self.num_gropus)
        self.start = nn.Sequential(
            # nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )
        self.fusionConv = nn.Sequential(
            nn.Conv2d(num_classes * 2, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(self.num_gropus, 32) if self.isGroupNorm else nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )
        self.lastConv = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        self.pointnet_result = nn.Sequential(
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )
        self.channel_shuffle = Channel_Shuffle(2)  # 将通道打乱
        # 二维图像转点云
        # show_point_cloud 必须保持 False：True 会在每次前向时调用
        # open3d.visualization.draw_geometries()，弹出阻塞窗口并让训练慢几个数量级
        self.transverter = ImageToPointCloud(show_point_cloud=False, is_save_point_cloud=False,
                                             save_path=r"E:\datesets\HaNSeg\fusion_point_image2", png_size=(256, 256))
        # self.pffBloack = PFFBloack(in_channels=num_classes, num_classes=num_classes, isGroupNorm=self.isGroupNorm,
        #                                  num_gropus=self.num_gropus)

    def deal_pointnet_result(self, orig_2d_point, result, ct_pred):
        """
            处理结果
        :param orig_2d_point: numpy类型，orig_2d_point[i]表示在原始图像中的[x,y]
        :param result: tensor类型, (batch_size, point_number, 30)
        :param ct_pred: 用于创建相同大小的tensor,ct_pred的shape(batch_size, 30, W, H)
        :return:
        """
        # 创建一个大小为
        batch_size, C, W, H = ct_pred.shape
        new_tensor = torch.zeros_like(ct_pred)

        # for b in range(batch_size):
        #     score, index = torch.max(result[b], dim=2)  # 得到每个点属于那个类别
        #     coords = torch.from_numpy(orig_2d_point[b])
        #     coords = coords.to(ct_pred.device)
        #     x_indices = coords[:, 1].view(1, -1)  # x 坐标对应列
        #     y_indices = coords[:, 0].view(1, -1)  # y 坐标对应行
        #     new_tensor[b, index, x_indices, y_indices] = score

        score, index = torch.max(result, dim=2)  # 得到每个点属于那个类别
        # threshold = (torch.min(score).item() + torch.max(score).item()) / 2.
        batch_indices = torch.zeros_like(index)
        coords = torch.from_numpy(orig_2d_point)
        coords = coords.to(ct_pred.device)
        x_indices = coords[:, 0].view(1, -1)  # x 坐标对应列
        y_indices = coords[:, 1].view(1, -1)  # y 坐标对应行
        new_tensor[batch_indices, index, x_indices, y_indices] = score
        return new_tensor

    def deal_point(self, ct_pred, mr_pred):
        """
        :param ct_pred: (1, 30, w, h)
        :param mr_pred: (1, 30, w, h)
        :return:
        """
        batch_size, C, W, H = ct_pred.shape
        ct_pred = (ct_pred > 0.7).float()
        mr_pred = (mr_pred > 0.7).float()


        points, orig_2d_points, index_C = self.transverter.combine(ct_pred[0], mr_pred[0])

        if len(points) == 0:
            return torch.zeros_like(ct_pred)
   

    
        coordinates = np.column_stack((np.array([*index_C]), np.array([*orig_2d_points])))
        _, unique_indices = np.unique(coordinates, axis=0, return_index=True)
        count = (np.unique(coordinates, axis=0, return_counts=True)[1] > 1)
        intersection_indices = unique_indices[count]

        if len(intersection_indices) == 0:
            return torch.zeros_like(ct_pred)
        numpy_combined = points[intersection_indices]
        combined_orig_2d_points = orig_2d_points[intersection_indices]
        combined_tensor = torch.from_numpy(numpy_combined)
        combined_tensor = combined_tensor.unsqueeze(dim=0).permute(0, 2, 1).to(ct_pred.device)
        combined_tensor = combined_tensor.to(dtype=ct_pred.dtype)

        result, _, _ = self.pointcls(combined_tensor)  # (1, point_number, 30)
        fusion_result = self.deal_pointnet_result(combined_orig_2d_points, result, ct_pred)
        fusion_result = fusion_result.to(device=ct_pred.device).to(dtype=ct_pred.dtype)
        return fusion_result

    def deal_fusion(self, probability_matrix, ct_pred, mr_pred):
  


        ct_outConv = ct_pred
        mr_outConv = mr_pred

        mr_weight = mr_outConv + probability_matrix

        ct_weight = ct_outConv + probability_matrix

        ct_highlight = mr_weight
        mr_highlight = ct_weight

        fusion_highlight = torch.cat([ct_highlight, mr_highlight], dim=1)
        fusion_highlight = self.channel_shuffle(fusion_highlight)
        fusion_highlight = self.fusionConv(fusion_highlight)

        fusion_common = torch.cat([ct_outConv, mr_outConv], dim=1)
        fusion_common = self.channel_shuffle(fusion_common)
        fusion_common = self.fusionConv(fusion_common)

        final_fusion = fusion_highlight + fusion_common
        final_fusion = self.lastConv(final_fusion)
        return final_fusion

    def forward(self, point1, point2, ct_last, mr_last, ct_pred, mr_pred):
        """
            ct_pred:(1, 30, w, h)
            mr_pred:(1, 30, w, h)
        """
        # s = time.time()
        ct_point = self.start(ct_pred)
        mr_point = self.start(mr_pred)

        fusion_result = self.deal_point(ct_point, mr_point)
 
        final_result = self.deal_fusion(fusion_result, ct_pred, mr_pred)

        return final_result
