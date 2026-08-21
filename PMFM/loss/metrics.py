import torch


class LossAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


class DiceAverage2D_single_label(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.avg = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.sum = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.count = torch.asarray([0] * self.class_num, dtype=torch.float64)
        self.total_avg = 0.0

    def update(self, logits, targets, label_id):
        dice = self.get_dices(logits, targets).item()
        self.value[label_id] = dice
        self.count[label_id] += 1.0
        self.sum[label_id] += dice
        self.avg[label_id] = self.sum[label_id] / self.count[label_id]
        self.total_avg = self.avg.mean().item()

    @staticmethod
    def get_dices(input, target, epsilon=1e-6):
        input, target = input.flatten(0, 1), target.flatten(0, 1)
        sum_dim = (-1, -2)
        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()


class BuiltInEvaluationMetrics(object):
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

    def update(self, dice, jc, hd, asd, label_id):
        self.count[label_id] += 1.0
        self.dice_sum[label_id] += dice
        self.jc_sum[label_id] += jc
        self.dice_avg[label_id] = self.dice_sum[label_id] / self.count[label_id]
        self.jc_avg[label_id] = self.jc_sum[label_id] / self.count[label_id]

        if hd != float('inf'):
            self.hd_sum[label_id] += hd
            self.asd_sum[label_id] += asd
            self.hd_avg[label_id] = self.hd_sum[label_id] / self.count[label_id]
            self.asd_avg[label_id] = self.asd_sum[label_id] / self.count[label_id]
