import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet2d import UNet
from .PFF import PFF


class PFFUNet(nn.Module):

    def __init__(self, in_channels: int = 1, num_classes: int = 2, bilinear: bool = True, base_c: int = 64, pointnet=False, extpn=False, isGroupNorm=False, num_gropus=32):
        super(PFFUNet, self).__init__()
        self.isGroupNorm = isGroupNorm
        self.num_gropus = num_gropus
        self.unet = UNet(in_channels, num_classes, bilinear, base_c, pointnet, extpn, is_point=True, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)
        self.pff = PFF(num_classes, isGroupNorm=self.isGroupNorm, num_gropus=self.num_gropus)

    def forward(self, ct, mr):
        target_ct_pred, point_ct, ct_last, ct_pred = self.unet(ct)
        target_mr_pred, point_mr, mr_last, mr_pred = self.unet(mr)
        fusion_target = self.pff(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)
        return target_ct_pred, target_mr_pred, fusion_target
