import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_num=None, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.ones(2) if alpha is None else alpha.float()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, labels, is_softmax=False):
        eps = 1e-7
        batch_size, channels, height, width = preds.shape
        inputs = preds if is_softmax else F.softmax(preds, dim=1)
        probs = inputs.view(batch_size, channels, -1)
        targets = labels.view(probs.size()).long()

        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        log_probs = -torch.log(probs + eps) * targets
        focal = torch.pow(1 - probs, self.gamma) * log_probs
        weights = self.alpha[targets]
        focal = focal * weights
        focal = torch.sum(focal, dim=1)
        return focal.mean() if self.size_average else focal.sum()


class DiceLoss2D(nn.Module):
    def __init__(self):
        super(DiceLoss2D, self).__init__()

    def forward(self, pred, target):
        return 1 - self.dice_coeff(pred, target)

    @staticmethod
    def dice_coeff(input, target, smooth=1.0):
        input, target = input.flatten(0, 1), target.flatten(0, 1)
        sum_dim = (-1, -2)
        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        dice = (inter + smooth) / (sets_sum + smooth)
        return dice.mean()
