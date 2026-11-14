
import torch
import numpy as np
import math
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize


class Resize:

    def __init__(self, size_w, size_h):
        self.size_w = size_w
        self.size_h = size_h

    def __call__(self, ct, mr, labels):
        ct, mr, labels = ct.unsqueeze(0), mr.unsqueeze(0), labels.unsqueeze(0)

        ct = F.interpolate(ct, size=(1, self.size_w, self.size_h), mode='trilinear', align_corners=False)

        mr = F.interpolate(mr, size=(1, self.size_w, self.size_h), mode='trilinear', align_corners=False)

        labels = F.interpolate(labels, size=(1, self.size_w, self.size_h), mode="nearest")
        return ct[0], mr[0], labels[0]


class RandomFlip_LR:
 

    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, ct, mr, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))  # 随机生成两个概率，介于[0,1)之间
        return self._flip(ct, prob), self._flip(mr, prob), self._flip(mask, prob)


class RandomFlip_UD:


    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, ct, mr, labels):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(ct, prob), self._flip(mr, prob), self._flip(labels, prob)


class cropSize:
    def __init__(self, w=None, h=None):
        super(cropSize, self).__init__()
        self.w = w
        self.h = h

    def __call__(self, ct, mr, labels):
        w_len = self.w
        h_len = self.h
        B, C, W, H = ct.shape
        surplus_w = W - w_len
        surplus_h = H - h_len
        w_start = math.floor(surplus_w / 2)
        w_end = w_start + w_len
        h_start = math.floor(surplus_h / 2)
        h_end = h_start + h_len
        new_ct = ct[:, :, w_start:w_end, h_start:h_end]
        new_mr = mr[:, :, w_start:w_end, h_start:h_end]
        new_labels = labels[:, :, w_start:w_end, h_start:h_end]
        return new_ct, new_mr, new_labels


class Crop:
    """
        随机裁剪
    """

    def __init__(self, W_size, H_size, step=1, eq_cnt=20):
        self.W_size = W_size
        self.H_size = H_size
        self.step = step
        self.eq_cnt = eq_cnt

    def get_reange(self, label):
        B, C, W, H = label.shape
        start_w = W
        end_w = 0
        label_sum = label.sum(dim=1).sum(dim=0)
        w_sum = label_sum.sum(dim=1)
        w_list = []
        for w in range(W):
            if w_sum[w].item():
                w_list.append(w)
                start_w = min(start_w, w)
                end_w = max(end_w, w)
        start_h = H
        end_h = 0
        h_sum = label_sum.sum(dim=0)
        h_list = []
        for h in range(H):
            if h_sum[h].item():
                h_list.append(h)
                start_h = min(start_h, h)
                end_h = max(end_h, h)
        return start_w, end_w, start_h, end_h

    def update(self, ct, mr, label, sw, ew, sh, eh):
        new_ct = ct[:, :, sw:ew, sh:eh]
        new_mr = mr[:, :, sw:ew, sh:eh]
        new_label = label[:, :, sw:ew, sh:eh]
        return new_ct, new_mr, new_label

    def get_random_wh(self, label, start_range_w, end_range_w, start_range_h, end_range_h):
        # B, C, W, H = label.shape
        start_H = start_range_h
        end_H = start_H + self.H_size

        start_W = start_range_w
        end_W = start_W + self.W_size

        num_eq = self.eq_cnt   # 允许等于的次数
        if (end_W < end_range_w) and (end_H >= end_range_h):
            for w in range(end_W, end_range_w + 1):
                s_w = w - self.W_size
                sum1 = label[:, :, start_W:end_W, start_H:end_H].sum().item()
                sum2 = label[:, :, s_w:w, start_H:end_H].sum().item()
                if sum2 >= sum1:
                    if (sum2 == sum1) and (num_eq <= 0):
                        continue
                    if sum2 == sum1:
                        num_eq -= 1
                    if sum2 > sum1:
                        num_eq = self.eq_cnt
                    start_W = s_w
                    end_W = w
        elif (end_H < end_range_h) and (end_W >= end_range_w):
            for h in range(end_H, end_range_h + 1):
                s_h = h - self.H_size
                sum1 = label[:, :, start_W:end_W, start_H:end_H].sum().item()
                sum2 = label[:, :, start_W:end_W, s_h:h].sum().item()
                if sum2 >= sum1:
                    if (sum2 == sum1) and (num_eq <= 0):
                        continue
                    if sum2 == sum1:
                        num_eq -= 1
                    if sum2 > sum1:
                        num_eq = self.eq_cnt
                    start_H = s_h
                    end_H = h
        else:
            for w in range(end_W, end_range_w + 1):
                for h in range(end_H, end_range_h + 1):
                    s_w = w - self.W_size
                    s_h = h - self.H_size
                    sum1 = label[:, :, start_W:end_W, start_H:end_H].sum().item()
                    sum2 = label[:, :, s_w:w, s_h:h].sum().item()
                    if sum2 >= sum1:
                        if (sum2 == sum1) and (num_eq <= 0):
                            continue
                        if sum2 == sum1:
                            num_eq -= 1
                        if sum2 > sum1:
                            num_eq = self.eq_cnt
                        start_W = s_w
                        end_W = w
                        start_H = s_h
                        end_H = h



        return start_W, end_W, start_H, end_H

    def up(self, ori_ct, ori_mr, ori_label, y):
 
        B, C, W, H = ori_label.shape
        start_x = 0
        start_y = y
        result_x = start_x
        result_y = start_y
        result_sum = 0
        num_eq = self.eq_cnt  # 允许等于的次数
        while (start_x + self.W_size) < W:
            label_sum = ori_label[:, :, start_x:start_x + self.W_size, start_y:start_y + self.H_size].sum().item()
            if label_sum > result_sum:
                num_eq = self.eq_cnt
                result_sum = label_sum
                result_x = start_x
                result_y = start_y
            start_x += self.step
            if (label_sum == result_sum) and (num_eq <= 0):
                continue
            if label_sum == result_sum:
                result_sum = label_sum
                result_x = start_x
                result_y = start_y
                num_eq -= 1

        new_ct, new_mr, new_label = self.update(ori_ct, ori_mr, ori_label, result_x, result_x + self.W_size, result_y,
                                                result_y + self.H_size)
        return new_ct, new_mr, new_label, result_x, result_x + self.W_size, result_y, result_y + self.H_size

    def left(self, ori_ct, ori_mr, ori_label, x):
 
        B, C, W, H = ori_label.shape
        start_x = x
        start_y = 0
        result_x = start_x
        result_y = start_y
        result_sum = 0
        num_eq = self.eq_cnt  # 允许等于的次数
        while (start_y + self.H_size) < H:
            label_sum = ori_label[:, :, start_x:start_x + self.W_size, start_y:start_y + self.H_size].sum().item()
            if label_sum > result_sum:
                num_eq = self.eq_cnt
                result_sum = label_sum
                result_x = start_x
                result_y = start_y
            start_y += self.step
            if (label_sum == result_sum) and (num_eq <= 0):
                continue
            if label_sum == result_sum:
                result_sum = label_sum
                result_x = start_x
                result_y = start_y
                num_eq -= 1

        new_ct, new_mr, new_label = self.update(ori_ct, ori_mr, ori_label, result_x, result_x + self.W_size, result_y, result_y + self.H_size)
        return new_ct, new_mr, new_label, result_x, result_x + self.W_size, result_y, result_y + self.H_size

    def top_left(self, ori_ct, ori_mr, ori_label, x, y):
  
        B, C, W, H = ori_label.shape
        while (x > 0) and (y > 0):
            x -= 1
            y -= 1
        start_x = x
        start_y = y
        result_x = start_x
        result_y = start_y
        result_sum = 0
        num_eq = self.eq_cnt  # 允许等于的次数
        while ((start_x + self.W_size) < W) and ((start_y + self.H_size) < H):
            label_sum = ori_label[:, :, start_x:start_x + self.W_size, start_y:start_y + self.H_size].sum().item()
            if label_sum > result_sum:
                num_eq = self.eq_cnt
                result_sum = label_sum
                result_x = start_x
                result_y = start_y
            start_x += self.step
            start_y += self.step
            if (label_sum == result_sum) and (num_eq <= 0):
                continue
            if label_sum == result_sum:
                result_sum = label_sum
                result_x = start_x
                result_y = start_y
                num_eq -= 1

        new_ct, new_mr, new_label = self.update(ori_ct, ori_mr, ori_label, result_x, result_x + self.W_size, result_y,
                                                result_y + self.H_size)
        return new_ct, new_mr, new_label, result_x, result_x + self.W_size, result_y, result_y + self.H_size

    def top_right(self, ori_ct, ori_mr, ori_label, x, y):
 
        B, C, W, H = ori_label.shape
        while (x > 0) and (y < (H - 1)):
            x -= 1
            y += 1
        start_x = x
        end_y = y
        result_x = start_x
        result_y = end_y
        result_sum = 0
        num_eq = self.eq_cnt  # 允许等于的次数
        while ((start_x + self.W_size) < W) and ((end_y - self.H_size) >= 0):
            label_sum = ori_label[:, :, start_x:start_x + self.W_size, end_y - self.H_size:end_y].sum().item()
            if label_sum > result_sum:
                num_eq = self.eq_cnt
                result_sum = label_sum
                result_x = start_x
                result_y = end_y
            start_x += self.step
            end_y -= self.step
            if (label_sum == result_sum) and (num_eq <= 0):
                continue
            if label_sum == result_sum:
                result_sum = label_sum
                result_x = start_x
                result_y = end_y
                num_eq -= 1

        new_ct, new_mr, new_label = self.update(ori_ct, ori_mr, ori_label, result_x, result_x + self.W_size, result_y,
                                                result_y + self.H_size)
        return new_ct, new_mr, new_label, result_x, result_x + self.W_size, result_y, result_y + self.H_size

    def get_dir_max_label_num(self, ori_ct, ori_mr, ori_label, new_ct, new_mr, new_label, x, end_W, y, end_H):
        up_ct, up_mr, up_label, up_start_w, up_end_w, up_start_h, up_end_h = self.up(ori_ct, ori_mr, ori_label, y)
        left_ct, left_mr, left_label, left_start_w, left_end_w, left_start_h, left_end_h = self.left(ori_ct, ori_mr, ori_label, x)
        top_left_ct, top_left_mr, top_left_label, top_left_start_w, top_left_end_w, top_left_start_h, top_left_end_h = self.top_left(ori_ct, ori_mr, ori_label, x, y)
        top_right_ct, top_right_mr, top_right_label, top_right_start_w, top_right_end_w, top_right_start_h, top_right_end_h = self.top_right(ori_ct, ori_mr, ori_label, x, y + self.H_size)
        sum_dict = {'up': up_label.sum().item(),
                    'left': left_label.sum().item(),
                    'top_left': top_left_label.sum().item(),
                    'top_right': top_right_label.sum().item(),
                    'new_first': new_label.sum().item()}
        result = sorted(sum_dict.items(), key=lambda x: x[1], reverse=True)
        if result[0][0] == 'up':
            return up_ct, up_mr, up_label, up_start_w, up_end_w, up_start_h, up_end_h
        elif result[0][0] == 'left':
            return left_ct, left_mr, left_label, left_start_w, left_end_w, left_start_h, left_end_h
        elif result[0][0] == 'top_left':
            return top_left_ct, top_left_mr, top_left_label, top_left_start_w, top_left_end_w, top_left_start_h, top_left_end_h
        elif result[0][0] == 'top_right':
            return top_right_ct, top_right_mr, top_right_label, top_right_start_w, top_right_end_w, top_right_start_h, top_right_end_h
        else:
            return new_ct, new_mr, new_label, x, end_W, y, end_H

    def center(self, ori_ct, ori_mr, ori_label, new_ct, new_mr, new_label, start_W, end_W, start_H, end_H):
        B, C, W, H = ori_ct.shape
        new_B, new_C, new_W, new_H = new_label.shape
  
        new_start_w, new_end_w, new_start_h, new_end_h = self.get_reange(new_label)

        col = new_H - new_end_h + new_start_h  
        row = new_W - new_end_w + new_start_w  

  
        right_move_cnt = 0
        left_move_cnt = 0
        up_move_cnt = 0
        down_move_cnt = 0


        right_cnt = int(col / 2)
        left_cnt = math.ceil(col / 2)
        up_cnt = int(row / 2)
        down_cnt = math.ceil(row / 2)

        if new_start_w >= up_cnt:
            up_move_cnt = new_start_w - up_cnt
            start_W = min(start_W + up_move_cnt, W - self.W_size)    # 物体要往上面移动，框框就要往下走
        else:
            down_move_cnt = up_cnt - new_start_w
            start_W = max(start_W - down_move_cnt, 0)    # 物体要往下面移动，框框就要往上走

        if new_start_h >= left_cnt:
            left_move_cnt = new_start_h - left_cnt
            start_H = min(start_H + left_move_cnt, H - self.H_size)   # 物体要往左移动，框框就要往右移动
        else:
            right_move_cnt = left_cnt - new_start_h
            start_H = max(start_H - right_move_cnt, 0)   # 物体要往右移动，框框就要往左移动
        new_ct, new_mr, new_label = self.update(ori_ct, ori_mr, ori_label, start_W, start_W + self.W_size, start_H, start_H + self.H_size)
        return new_ct, new_mr, new_label

    def __call__(self, ct, mr, label):
        start_range_w, end_range_w, start_range_h, end_range_h = self.get_reange(label)
        start_W, end_W, start_H, end_H = self.get_random_wh(label, start_range_w, end_range_w, start_range_h, end_range_h)
        new_ct, new_mr, new_label = self.update(ct, mr, label, start_W, end_W, start_H, end_H)

        new_ct, new_mr, new_label, start_W, end_W, start_H, end_H = self.get_dir_max_label_num(ct, mr, label, new_ct, new_mr, new_label, start_W, end_W, start_H, end_H)
        new_ct, new_mr, new_label = self.center(ct, mr, label, new_ct, new_mr, new_label, start_W, end_W, start_H, end_H)
        return new_ct, new_mr, new_label


class RandomRotate:
 

    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img, cnt, [1, 2])
        return img

    def __call__(self, ct, mr, label):
        B, C, W, H = ct.shape
        cnt = random.randint(0, self.max_cnt)
        for b in range(B):
            ct[b] = self._rotate(ct[b], cnt)
            mr[b] = self._rotate(mr[b], cnt)
            label[b] = self._rotate(label[b], cnt)
        return ct, mr, label

class RandomCropLabel:
    """
        随机裁剪
    """

    def __init__(self, W_size, H_size):
        self.W_size = W_size
        self.H_size = H_size

    def get_reange(self, label):
        B, C, W, H = label.shape
        start_w = W
        end_w = 0
        label_sum = label.sum(dim=1).sum(dim=0)
        for w in range(W):
            if label_sum[w:w + 1, :].sum().item():
                start_w = min(start_w, w)
                end_w = max(end_w, w)
        start_h = H
        end_h = 0
        for h in range(H):
            if label_sum[:, h:h + 1].sum().item():
                start_h = min(start_h, h)
                end_h = max(end_h, h)
        if start_w > end_w:
            start_w, end_w = end_w, start_w
        if start_h > end_h:
            start_h, end_h = end_h, start_h
        return start_w, end_w, start_h, end_h

    def __call__(self, ct, mr, label):
        B, C, W, H = ct.shape
        start_range_w, end_range_w, start_range_h, end_range_h = self.get_reange(label)
        start_H = random.randint(max(0, start_range_h - self.H_size), min(end_range_h, H - self.H_size))
        end_H = start_H + self.H_size
        # start_H = random.randint(0, H - self.H_size)
        # end_H = start_H + self.H_size
        end_W = random.randint(max(start_range_w, self.W_size), min(end_range_w + self.W_size + 1, W))
        start_W = end_W - self.W_size
        new_ct = ct[:, :, start_W:end_W, start_H:end_H]
        new_mr = mr[:, :, start_W:end_W, start_H:end_H]
        new_label = label[:, :, start_W:end_W, start_H:end_H]
        return new_ct, new_mr, new_label


class Compose:
    """
        组合使用数据增强。
        >>>class MockTransform:
        >>>     def __init__(self):
        >>>         pass
        >>>     def __call__(self, img, mask):
        >>>         return img * 2, mask * 3
        >>># 创建Compose的实例，包含两个示例转换函数
        >>>transform_1 = MockTransform()
        >>>transform_2 = MockTransform()
        >>>compose_transform = Compose([transform_1, transform_2])
        >>># 创建示例图像和掩码
        >>>img = torch.ones(3, 256, 256)
        >>>mask = torch.ones(1, 256, 256)
        >>># 应用Compose转换
        >>>transformed_img, transformed_mask = compose_transform(img, mask)
        >>>assert torch.allclose(transformed_img, img * 2 * 2)
        >>>assert torch.allclose(transformed_mask, mask * 3 * 3)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, ct, mr, labels):
        for t in self.transforms:
            ct, mr, labels = t(ct, mr, labels)
        return ct, mr, labels
