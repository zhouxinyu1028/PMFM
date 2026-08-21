import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils.dice_score import dice_loss, multiclass_dice_coeff


class LossAverage(object):
    """
        计算并存储平均值和当前值，用于计算平均损耗
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


class DiceAverage(object):
    """
        计算并存储平均值和当前值，用于计算平均损耗
    """

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0.

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += logits.shape[0]
        self.avg = self.sum / self.count

    @staticmethod
    def get_dices(logits, targets, epsilon: float = 1e-6):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
            union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, :])
            dice = (2. * inter + epsilon) / (union + epsilon)
            dices.append(dice.item())
        return np.asarray(dices)


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


class DiceAverage2D(object):
    """
        计算并存储平均值和当前值，用于计算平均损耗
    """

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.avg = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.count = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.total_avg = 0.

    def update(self, logits, new_targets, targets):
        tempDice = DiceAverage2D.get_dices(logits, new_targets)
        step = 0
        for classes_id in range(targets.shape[1]):
            if torch.max(targets[:, classes_id, :, :]).item() == 1:
                self.value[classes_id] = tempDice[step]
                self.count[classes_id] += 1.
                self.sum[classes_id] += tempDice[step]
                self.avg[classes_id] = self.sum[classes_id] / self.count[classes_id]
                step += 1
        self.total_avg = self.avg.mean().item()

    @staticmethod
    def get_dices(input, target, epsilon: float = 1e-6):
        input, target = input.flatten(0, 1), target.flatten(0, 1)
        # 所有batches或单个mask的Dice系数的平均值
        sum_dim = (-1, -2)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice


class DiceAverage2D_single_label(object):
    """
        计算并存储平均值和当前值，用于计算平均损耗
    """

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.avg = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.count = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.total_avg = 0.

    def update(self, logits, targets, label_id):
        dice = DiceAverage2D_single_label.get_dices(logits, targets).item()
        self.value[label_id] = dice
        self.count[label_id] += 1.
        self.sum[label_id] += dice
        self.avg[label_id] = self.sum[label_id] / self.count[label_id]
        self.total_avg = self.avg.mean().item()

    @staticmethod
    def get_dices(input, target, epsilon: float = 1e-6):
        input, target = input.flatten(0, 1), target.flatten(0, 1)
        # 所有batches或单个mask的Dice系数的平均值
        sum_dim = (-1, -2)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()


class BuiltInEvaluationMetrics(object):
    # 内置评价指标
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.dice_sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.dice_avg = torch.asarray([0] * self.class_num, dtype=torch.float64)

        self.jc_sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.jc_avg = torch.asarray([0] * self.class_num, dtype=torch.float64)

        self.hd_sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.hd_avg = torch.asarray([0] * self.class_num, dtype=torch.float64)

        self.asd_sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.asd_avg = torch.asarray([0] * self.class_num, dtype=torch.float64)

        self.count = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.total_dice_avg = 0.
        self.total_jc_avg = 0.
        self.total_hd_avg = 0.
        self.total_asd_avg = 0.

    def update(self, dice, jc, hd, asd, label_id):
        self.count[label_id] += 1.
        self.dice_sum[label_id] += dice
        self.jc_sum[label_id] += jc

        self.dice_avg[label_id] = self.dice_sum[label_id] / self.count[label_id]
        self.jc_avg[label_id] = self.jc_sum[label_id] / self.count[label_id]

        self.total_dice_avg = self.dice_avg.mean().item()
        self.total_jc_avg = self.jc_avg.mean().item()

        if hd != float('inf'):
            self.hd_sum[label_id] += hd
            self.asd_sum[label_id] += asd
            self.hd_avg[label_id] = self.hd_sum[label_id] / self.count[label_id]
            self.asd_avg[label_id] = self.asd_sum[label_id] / self.count[label_id]

            self.total_hd_avg = self.hd_avg.mean().item()
            self.total_asd_avg = self.asd_avg.mean().item()


# class DiceAverage2D_v2(object):
#     """
#         计算并存储平均值和当前值，用于计算平均损耗
#     """
#
#     def __init__(self, class_num):
#         self.class_num = class_num
#         self.reset()
#
#     def reset(self):
#         self.value = torch.asarray([0] * self.class_num, dtype=torch.float64)
#         self.avg = torch.asarray([0] * self.class_num, dtype=torch.float64)
#         self.sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
#         self.count = torch.asarray([0] * self.class_num, dtype=torch.float64)
#         self.total_avg = 0.
#
#     def update(self, logits, new_targets, targets):
#         tempDice = DiceAverage2D_v2.get_dices(logits, new_targets)
#         step = 0
#         for classes_id in range(targets.shape[1]):
#             if torch.max(targets[:, classes_id, :, :]).item() == 1:
#                 self.value[classes_id] = tempDice[step]
#                 self.count[classes_id] += 1.
#                 self.sum[classes_id] += tempDice[step]
#                 self.avg[classes_id] = self.sum[classes_id] / self.count[classes_id]
#                 step += 1
#         self.total_avg = self.avg.mean()
#
#     @staticmethod
#     def get_dices(input, target, epsilon: float = 1e-6):
#         input, target = input.flatten(0, 1), target.flatten(0, 1)
#         # 所有batches或单个mask的Dice系数的平均值
#         sum_dim = (-1, -2)
#
#         inter = 2 * (input * target).sum(dim=sum_dim)
#         sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
#         sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
#         dice = (inter + epsilon) / (sets_sum + epsilon)
#         return dice