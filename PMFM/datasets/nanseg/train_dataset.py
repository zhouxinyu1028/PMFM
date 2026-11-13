import os
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
from pathlib import Path
from utils.transforms2D import Compose, RandomFlip_LR, RandomFlip_UD, Crop, RandomRotate, Resize
import logging


def load_image_pair(data, data_dir):
    """
        图像与标签分文件存放时（图像只存一份在 slices/ 下），按样本里的 image_path 去读；
        兼容旧格式（样本 npz 内自带 images）
    """
    if 'images' in data.files:
        return torch.from_numpy(data['images'][0]), torch.from_numpy(data['images'][1])
    images = np.load(Path(data_dir).parent.joinpath(str(data['image_path'])))['images']
    return torch.from_numpy(images[0]), torch.from_numpy(images[1])


def normalize_minmax(tensor):
    """min-max 归一化；patch 为常数（例如全零的 MR 层）时归一化会 0/0 得到 NaN，这里直接返回全零"""
    tensor = tensor - torch.min(tensor)
    max_value = torch.max(tensor)
    if max_value <= 0:
        return torch.zeros_like(tensor)
    return tensor / max_value


class TrainDataset(dataset):
    """
        只需要传递yaml文件对应的data部分即可
    """

    def __init__(self, args, data_dir=None, datas=None):
        self.args = args
        if data_dir is None:
            self.data_dir = Path(self.args['train']['datas_dir'])  # 图片路径
        else:
            self.data_dir = Path(data_dir)

        if datas is None:
            self.datas = os.listdir(self.data_dir)
        else:
            self.datas = datas

        if not self.datas:
            raise RuntimeError(f'No input file found in {self.data_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.datas)} examples')

        # 数据增强
        self.transforms = Compose([
            # Resize(512, 512),
            Crop(int(self.args['transforms']['Crop_w']), int(self.args['transforms']['Crop_h']), int(self.args['transforms']['step']), int(self.args['transforms']['eq_cnt'])),
            # Resize(float(self.args['transforms']['scale'])),
            # RandomFlip_LR(prob=0.5),
            # RandomFlip_UD(prob=0.5),
            # cropSize(512, 512),
            RandomRotate()
        ])

    def __getitem__(self, index):
        data = np.load(self.data_dir.joinpath(self.datas[index]))
        ct_tensor, mr_tensor = load_image_pair(data, self.data_dir)
        label_tensor = torch.from_numpy(data['labels']).float()
        if self.transforms:
            ct_tensor, mr_tensor, label_tensor = self.transforms(ct_tensor, mr_tensor, label_tensor)
        ct = normalize_minmax(ct_tensor.squeeze(0))
        mr = normalize_minmax(mr_tensor.squeeze(0))
        label = label_tensor.squeeze(0)
        label = (label > 0).float()   # 数据下采样或者上采用中，导致标签出现0.5、0.25的情况
        label_id = data['label_id']
        return ct, mr, label, ct_tensor.squeeze(0), mr_tensor.squeeze(0), label_id

    def __len__(self):
        return len(self.datas)
