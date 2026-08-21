# 部分特征融合
import numpy as np
import torch
import torch.nn as nn
from .pointnet import PointNetDenseCls
import torch.nn.functional as F
from utils.image_to_pointCloud import ImageToPointCloud


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
        self.transverter = ImageToPointCloud()
        self.focus_directions = np.asarray([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=np.int64)
        # self.pffBloack = PFFBloack(in_channels=num_classes, num_classes=num_classes, isGroupNorm=self.isGroupNorm,
        #                                  num_gropus=self.num_gropus)

    def focus_filter(self, points, orig_2d_points, index_C):
        if len(points) == 0:
            return points, orig_2d_points

        coordinates = np.column_stack((
            index_C.astype(np.int64),
            orig_2d_points.astype(np.int64),
        ))
        unique_coordinates, unique_indices, counts = np.unique(
            coordinates,
            axis=0,
            return_index=True,
            return_counts=True,
        )

        intersection_coordinates = unique_coordinates[counts > 1]
        if len(intersection_coordinates) == 0:
            return np.empty((0, 3), dtype=points.dtype), np.empty((0, 2), dtype=orig_2d_points.dtype)

        candidate_set = {tuple(coord) for coord in unique_coordinates}
        selected_set = {tuple(coord) for coord in intersection_coordinates}
        for coord in intersection_coordinates:
            for direction in self.focus_directions:
                neighbor = tuple(coord + direction)
                if neighbor in candidate_set:
                    selected_set.add(neighbor)

        selected_indices = [
            unique_indices[idx]
            for idx, coord in enumerate(unique_coordinates)
            if tuple(coord) in selected_set
        ]
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        return points[selected_indices], orig_2d_points[selected_indices]

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
        # 他们都是list类型,例如ct_point[i]，都是numpy类型,尺寸为(point_number, 3)
        # pointNet_results = []
        # orig_2d = []
        # for b in range(batch_size):
        #     ct_points, ct_orig_2d_points = self.transverter.Change(ct_pred[b])  # 他们的长度都是30
        #     mr_points, mr_orig_2d_points = self.transverter.Change(mr_pred[b])
        #     # 对齐每个类别
        #     points = np.array([*ct_points, *mr_points])
        #     orig_2d_points = np.array([*ct_orig_2d_points, *mr_orig_2d_points])
        #     _, unique_indices = np.unique(orig_2d_points, axis=0, return_index=True)
        #     numpy_combined = points[unique_indices]
        #     combined_orig_2d_points = orig_2d_points[unique_indices]
        #
        #     combined_tensor = torch.from_numpy(numpy_combined)
        #     combined_tensor = combined_tensor.unsqueeze(dim=0).permute(0, 2, 1).to(ct_pred.device)
        #     combined_tensor = combined_tensor.to(dtype=ct_pred.dtype)
        #
        #     result, _, _ = self.pointcls(combined_tensor)  # (1, point_number, 30)
        #     pointNet_results.append(result)
        #     orig_2d.append(combined_orig_2d_points)

        # 他们都是list类型,例如ct_point[i]，都是numpy类型,尺寸为(point_number, 3)
        # ct_points, ct_orig_2d_points, ct_C = self.transverter.Change(ct_pred[0])  # 他们的长度都是30
        # mr_points, mr_orig_2d_points, mr_C = self.transverter.Change(mr_pred[0])
        points, orig_2d_points, index_C = self.transverter.combine(ct_pred[0], mr_pred[0])
        # # 某一个点同时含有30个分类的点才能送入pointnet中
        # # 1. 首先处理CT
        # # 由于二维坐标每一个类别不可能有相同点，所以直接求出每个坐标出现的次数，从而可以确定出这个点出现在了多少个类别中
        # value_ct, count_ct = np.unique(ct_orig_2d_points, axis=0, return_counts=True)
        # ids_ct = np.where((ct_orig_2d_points == value_ct[count_ct == C][:, None]).all(-1))[1]
        # points_ct = ct_points[ids_ct]
        # orig_2d_points_ct = ct_orig_2d_points[ids_ct]
        #
        # # 2. 处理MR
        # value_mr, count_mr = np.unique(mr_orig_2d_points, axis=0, return_counts=True)
        # ids_mr = np.where((mr_orig_2d_points == value_mr[count_mr == C][:, None]).all(-1))[1]
        # points_mr = ct_points[ids_mr]
        # orig_2d_points_mr = mr_orig_2d_points[ids_mr]
        # # 得到每一层都拥有的点
        # points = np.array([*points_ct, *points_mr])
        # orig_2d_points = np.array([*orig_2d_points_ct, *orig_2d_points_mr])
        # # 然后在给他们去重(MR中有的点，CT也有)
        # _, unique_index = np.unique(orig_2d_points, axis=0, return_index=True)
        # final_points = points[unique_index]
        # final_2d_points = orig_2d_points[unique_index]

        numpy_combined, combined_orig_2d_points = self.focus_filter(points, orig_2d_points, index_C)
        if len(numpy_combined) == 0:
            return torch.zeros_like(ct_pred)

        combined_tensor = torch.from_numpy(numpy_combined)
        combined_tensor = combined_tensor.unsqueeze(dim=0).permute(0, 2, 1).to(ct_pred.device)
        combined_tensor = combined_tensor.to(dtype=ct_pred.dtype)

        result, _, _ = self.pointcls(combined_tensor)  # (1, point_number, 30)
        fusion_result = self.deal_pointnet_result(combined_orig_2d_points, result, ct_pred)
        fusion_result = fusion_result.to(device=ct_pred.device).to(dtype=ct_pred.dtype)
        return fusion_result

    def deal_fusion(self, probability_matrix, ct_pred, mr_pred):
        """
            融合概率矩阵，同时尽可能保留更多的Unet分割的结果，以免丢失更多的信息
        :param mr_last: UNet中的倒数第二层(batch_size, 64, w, h)
        :param ct_last: UNet中的倒数第二层(batch_size, 64, w, h)
        :param probability_matrix: 是用pointent分割后的矩阵
        :return:
        """

        # 对Unet中倒数第二层重新提取一次特征，用于后面更加细致的融合
        ct_outConv = ct_pred
        mr_outConv = mr_pred
        # 通过概率矩阵，对应到mr_pred,挑出还有概率的部分
        mr_weight = mr_outConv + probability_matrix
        # 同理ct_pred一样
        ct_weight = ct_outConv + probability_matrix
        # 交叉加上对应的权重(使得分割目标处更加突出<CT分割的结果需要在MR结果中突出,同理MR也是>)
        ct_highlight = mr_weight
        mr_highlight = ct_weight
        # 使用通道洗牌操作并且结合Fusion卷积融合,主要融合突出部分
        fusion_highlight = torch.cat([ct_highlight, mr_highlight], dim=1)
        fusion_highlight = self.channel_shuffle(fusion_highlight)
        fusion_highlight = self.fusionConv(fusion_highlight)
        # 同理还需对Unet分割后的结果进行融合，（称之为"普通融合"）
        fusion_common = torch.cat([ct_outConv, mr_outConv], dim=1)
        fusion_common = self.channel_shuffle(fusion_common)
        fusion_common = self.fusionConv(fusion_common)
        # 最后在把两者进行融合，得到最终分割结果
        final_fusion = fusion_highlight + fusion_common
        final_fusion = self.lastConv(final_fusion)
        return final_fusion

    def forward(self, point1, point2, ct_last, mr_last, ct_pred, mr_pred):
        """
            ct_pred:(1, 30, w, h)
            mr_pred:(1, 30, w, h)
        """
        ct_point = self.start(ct_pred)
        mr_point = self.start(mr_pred)
        # 将Unet分割后的结果转换为点云数据，然后在使用pointnet分割
        fusion_result = self.deal_point(ct_point, mr_point)
        # fusion_result是一个两种图像通过点云的方式去融合，然后在使用pointnet分割，得到每个点的属于那个类别的概率分数
        # 通过筛选出概率分数大于阈值(1/C)的点，组合成新的矩阵(funsion_result)，这个矩阵只有0和大于阈值的浮点数（每个点的概率分数）
        final_result = self.deal_fusion(fusion_result, ct_pred, mr_pred)
        return final_result
