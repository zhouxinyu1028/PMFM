"""
    基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
import numpy as np


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(
                    dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        smooth + target[:, i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        smooth + (1 - target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, pred, target):

        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


# class TverskyLoss2D(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred, target, alpha=0.7):
#         smooth = 1
#         dice = 0.
#         valid_label_number = 0.  # 有效标签数量
#         for i in range(pred.size(1)):
#             # if torch.max(target[:, i]).item() == 0:
#             #     continue
#             valid_label_number += 1.
#             dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
#                     (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) +
#                     0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1) + alpha * (
#                             (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1) + smooth)
#
#         dice = dice / valid_label_number
#         return torch.clamp((1 - dice).mean(), 0, 2)

class TverskyLoss2D(nn.Module):
    '''
    Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    '''

    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss2D, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:  # alpha 是平衡因子
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def forward(self, preds, labels, is_softmax=False):
        """
        :param is_softmax:
        :param preds: (batch_size, num_classes, w, h)
        :param labels: (batch_size, num_classes, w, h), one_hot
        :return:
        """
        eps = 1e-7
        B, C, H, W = preds.shape
        if is_softmax:
            inputs = preds
        else:
            inputs = F.softmax(preds, dim=1)
        pt = inputs.view((B, C, -1))  # (B, C, H, W) -> (B, C, H * W)
        target = labels.view(pt.size())  # (B, C, H, W) -> (B, C, H * W)
        log_pt = -1 * torch.log(pt + eps) * target
        focal_loss = torch.pow((1 - pt), self.gamma) * log_pt  # (B, C, H * W)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外

        alpha = self.alpha[target]  # (B, C, H, W) -> (B, C, H * W)
        focal_loss = torch.mul(focal_loss, alpha)
        focal_loss = torch.sum(focal_loss, dim=1)  # 所有类别加起来
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceLoss2D(nn.Module):
    def __init__(self):
        super(DiceLoss2D, self).__init__()

    def forward(self, pred, target):
        dice = DiceLoss2D.diceCoeff(pred, target)
        return 1 - dice

    @staticmethod
    def diceCoeff(input, target, smooth=1.):
        input, target = input.flatten(0, 1), target.flatten(0, 1)
        # 所有batches或单个mask的Dice系数的平均值
        sum_dim = (-1, -2)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + smooth) / (sets_sum + smooth)
        return dice.mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss


class DiceLoss2Dv2(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss2Dv2, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, input, target):
        """
            input tesor of shape = (N, C, H, W)
            target tensor of shape = (N, C, H, W)
        """

        assert input.shape == target.shape, "predict & target shape do not match"

        binaryDiceLoss = BinaryDiceLoss()
        total_loss = 0

        # 归一化输出
        logits = F.softmax(input, dim=1)
        C = target.shape[1]

        # 遍历 channel，得到每个类别的二分类 DiceLoss
        for i in range(C):
            dice_loss = binaryDiceLoss(logits[:, i], target[:, i])
            total_loss += dice_loss

        # 每个类别的平均 dice_loss
        return total_loss / C


class GDiceLoss(nn.Module):
    """
        Generalized Dice Loss
    """
    def __init__(self):
        super(GDiceLoss, self).__init__()

    def forward(self, pred, target, epsilon=1e-6):
        """compute the weighted dice_loss
    Args:
        :param epsilon:
        pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
        target (tensor): gt, shape(bath_size, channels, height, width)
    Returns:
        gldice_loss: loss value
    """
        wei = torch.sum(target, axis=[0, 2, 3])      # (n_class,)
        wei = 1 / (wei ** 2 + epsilon)
        intersection = torch.sum(wei * torch.sum(pred * target, axis=[0, 2, 3]))
        union = torch.sum(wei * torch.sum(pred + target, axis=[0, 2, 3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss


class SSLoss2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=1).sum(dim=1) / (
                        smooth + target[:, i].sum(dim=1).sum(dim=1))

            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=1).sum(dim=1) / (
                        smooth + (1 - target[:, i]).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


# class JaccardLoss2D(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred, target):
#         smooth = 1
#
#         # jaccard系数的定义
#         jaccard = 0.
#
#         for i in range(pred.size(1)):
#             jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
#                         pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
#                         target[:, i].pow(2).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(
#                     dim=1).sum(dim=1) + smooth)
#
#         # 返回的是jaccard距离
#         jaccard = jaccard / pred.size(1)
#         return torch.clamp((1 - jaccard).mean(), 0, 1)

class JaccardLoss2D(nn.Module):
    def __init__(self):
        super(JaccardLoss2D, self).__init__()

    def forward(self, true, logits, eps=1e-7, activation=True):
        """
        Computes the Jaccard loss, a.k.a the IoU loss.
        :param true: a tensor of shape [B, H, W] or [B, C, H, W] or [B, 1, H, W].
        :param logits: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        :param eps: added to the denominator for numerical stability.
        :param activation: if apply the activation function before calculating the loss.
        :return: the Jaccard loss.
        """
        num_classes = logits.shape[1]
        # if num_classes == 1:
        #     true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        #     true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        #     true_1_hot_f = true_1_hot[:, 0:1, :, :]
        #     true_1_hot_s = true_1_hot[:, 1:2, :, :]
        #     true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        #     pos_prob = torch.sigmoid(logits)
        #     neg_prob = 1 - pos_prob
        #     probas = torch.cat([pos_prob, neg_prob], dim=1)
        # else:

        probas = F.softmax(logits, dim=1) if activation else logits

        true_1_hot = true.type(probas.type())
        dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
        probas = probas.contiguous()
        true_1_hot = true_1_hot.contiguous()
        intersection = probas * true_1_hot
        intersection = torch.sum(intersection, dims)
        cardinality = probas + true_1_hot
        cardinality = torch.sum(cardinality, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return 1 - jacc_loss


class HybridLoss2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class ELDiceLoss2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLossv2(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLossv2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, target, is_one_hot=False, is_softmax=False):
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1, 2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))   # N,H*W,C => N*H*W,C
        if is_one_hot:
            target = target.view(-1, 1)
        else:
            target = target.view(-1, target.size(1))
        if is_softmax:
            logpt = torch.log(pred)
        else:
            logpt = F.log_softmax(pred, dim=1)  # 这里转成log(pt)
        if is_one_hot:
            logpt = logpt.gather(1, target)
        else:
            logpt = logpt * target
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))

            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalCosineLoss(nn.Module):
    """Implementation Focal cosine loss.
    [Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble](https://arxiv.org/abs/2007.07805).
    Source : <https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271>
    """

    def __init__(self, alpha=1, gamma=2, xent=0.1, reduction="mean"):
        """Constructor for FocalCosineLoss.
        """
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, preds, target):
        """Forward Method."""
        cosine_loss = F.cosine_embedding_loss(
            preds,
            torch.nn.functional.one_hot(target, num_classes=preds.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )
        cent_loss = F.cross_entropy(F.normalize(preds), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss
        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss