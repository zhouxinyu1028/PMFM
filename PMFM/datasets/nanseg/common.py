from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class HanSegSliceDataset(Dataset):
    """Dataset for HaNSeg CT/MR slice npz files used by PMFM training."""

    def __init__(self, data_args, split):
        split_args = data_args[split]
        self.datas_dir = Path(split_args["datas_dir"])
        self.files = sorted(self.datas_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.datas_dir}")

        transforms = data_args.get("transforms", {})
        self.crop_w = int(transforms.get("Crop_w", 128))
        self.crop_h = int(transforms.get("Crop_h", 128))
        self.is_train = split == "train"

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _squeeze_image(arr):
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 3:
            arr = np.squeeze(arr, axis=1)
        return arr

    @staticmethod
    def _normalize(image):
        image = image.astype(np.float32, copy=False)
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value > min_value:
            image = (image - min_value) / (max_value - min_value)
        return image

    def _crop_bounds(self, label):
        _, h, w = label.shape
        crop_h = min(self.crop_h, h)
        crop_w = min(self.crop_w, w)

        foreground = torch.nonzero(label > 0, as_tuple=False)
        if self.is_train and len(foreground) > 0:
            index = torch.randint(0, len(foreground), (1,)).item()
            _, center_h, center_w = foreground[index].tolist()
        else:
            center_h, center_w = h // 2, w // 2

        start_h = max(0, min(center_h - crop_h // 2, h - crop_h))
        start_w = max(0, min(center_w - crop_w // 2, w - crop_w))
        return start_h, start_h + crop_h, start_w, start_w + crop_w

    @staticmethod
    def _pad_to_size(tensor, crop_h, crop_w):
        _, h, w = tensor.shape
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor

    def __getitem__(self, index):
        data = np.load(self.files[index])
        images = self._squeeze_image(data["images"])
        label = self._squeeze_image(data["labels"])
        label_id = int(np.asarray(data["label_id"]).item())

        ct = torch.from_numpy(self._normalize(images[0]))
        mr = torch.from_numpy(self._normalize(images[1]))
        label = torch.from_numpy((label > 0).astype(np.float32))

        start_h, end_h, start_w, end_w = self._crop_bounds(label)
        ct = ct[:, start_h:end_h, start_w:end_w]
        mr = mr[:, start_h:end_h, start_w:end_w]
        label = label[:, start_h:end_h, start_w:end_w]

        ct = self._pad_to_size(ct, self.crop_h, self.crop_w)
        mr = self._pad_to_size(mr, self.crop_h, self.crop_w)
        label = self._pad_to_size(label, self.crop_h, self.crop_w)

        return ct, mr, label, ct.clone(), mr.clone(), torch.tensor(label_id, dtype=torch.long)

