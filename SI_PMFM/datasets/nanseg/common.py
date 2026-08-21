from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HanSegSliceDataset(Dataset):
    """Dataset reader for the SI npz slices exported under Datasets_SI."""

    def __init__(self, data_args, split="train"):
        self.data_args = data_args
        self.split = split
        split_args = data_args[split]
        self.data_dir = Path(split_args["datas_dir"])
        self.crop_w = int(data_args.get("transforms", {}).get("Crop_w", 128))
        self.crop_h = int(data_args.get("transforms", {}).get("Crop_h", 128))
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _squeeze_image(arr):
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr.astype(np.float32)

    @staticmethod
    def _normalize(arr):
        arr = arr.astype(np.float32)
        min_v = float(arr.min())
        max_v = float(arr.max())
        if max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v)
        return arr

    def _crop_bounds(self, label):
        mask = label[0] > 0
        h, w = mask.shape
        if self.split == "train" and mask.any():
            ys, xs = np.where(mask)
            cy = int((ys.min() + ys.max()) / 2)
            cx = int((xs.min() + xs.max()) / 2)
        else:
            cy = h // 2
            cx = w // 2

        y0 = max(0, min(h - self.crop_h, cy - self.crop_h // 2))
        x0 = max(0, min(w - self.crop_w, cx - self.crop_w // 2))
        return y0, x0

    def _crop_or_pad(self, arr, y0, x0):
        arr = arr[:, y0:y0 + self.crop_h, x0:x0 + self.crop_w]
        pad_h = self.crop_h - arr.shape[-2]
        pad_w = self.crop_w - arr.shape[-1]
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant")
        return arr

    def __getitem__(self, index):
        sample = np.load(self.files[index])
        images = self._squeeze_image(sample["images"])
        label = self._squeeze_image(sample["labels"])
        label_id = int(np.asarray(sample["label_id"]).item())

        if images.shape[0] < 2:
            raise ValueError(f"Expected CT/MR two-channel images in {self.files[index]}")

        ct = self._normalize(images[0:1])
        mr = self._normalize(images[1:2])
        label = (label > 0).astype(np.float32)
        y0, x0 = self._crop_bounds(label)

        ct = self._crop_or_pad(ct, y0, x0)
        mr = self._crop_or_pad(mr, y0, x0)
        label = self._crop_or_pad(label, y0, x0)

        ct = torch.from_numpy(ct)
        mr = torch.from_numpy(mr)
        label = torch.from_numpy(label)
        return ct, mr, label, ct.clone(), mr.clone(), torch.tensor(label_id, dtype=torch.long)

