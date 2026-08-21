import torch.nn as nn
from .unet2d import UNet
from .CFM import CFM


class PMFM(nn.Module):
    def __init__(self, in_channels=1, num_classes=30, bilinear=True, base_c=64, pointnet=True, extpn=False, isGroupNorm=False, num_gropus=32):
        super(PMFM, self).__init__()
        self.unet = UNet(in_channels, num_classes, bilinear, base_c, pointnet, extpn, is_point=True, isGroupNorm=isGroupNorm, num_gropus=num_gropus)
        self.cfm = CFM(num_classes, isGroupNorm=isGroupNorm, num_gropus=num_gropus)

    def forward(self, ct, mr):
        target_ct_pred, point_ct, ct_last, ct_pred = self.unet(ct)
        target_mr_pred, point_mr, mr_last, mr_pred = self.unet(mr)
        fusion_target = self.cfm(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)
        return target_ct_pred, target_mr_pred, fusion_target
