import time

import torch
import torch.nn as nn
import numpy as np
import imageio
import SimpleITK as sitk
from PIL import Image
import cv2
import os
import torch.nn.functional as F


def TensorToPng(tensor, png_path, name):
    """
    :param tensor: (batch_size, c, w, h)
    :param png_path:
    :return:
    """
    tensor1 = tensor.clone()
    fileList = os.listdir(png_path)
    for fileName in fileList:
        os.remove(os.path.join(png_path, fileName))
    batch_size, C, W, H = tensor1.shape
    for b in range(batch_size):
        for c in range(C):
            imageTensor = tensor1[b, c, :, :].unsqueeze(dim=0).unsqueeze(dim=0)
            numpy_var = imageTensor.cpu().numpy()
            image = numpy_var[0][0] * 255.
            new_name1 = name + "_" + str(b) + "_" + str(c)
            imageio.imwrite(os.path.join(png_path, '{}.png'.format(new_name1)), image.astype(np.uint8))

    tensor2 = tensor.clone()
    for b in range(batch_size):
        for c in range(C):
            imageTensor = tensor2[b, c, :, :].unsqueeze(dim=0).unsqueeze(dim=0)
            # imageTensor = F.one_hot(imageTensor.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
            imageTensor = (imageTensor > 0.5).float()
            imageNumpy = imageTensor.cpu().numpy()
            image = imageNumpy[0][0] * 255.
            new_name2 = name + "_one_hot(0.5)" + "_" + str(b) + "_" + str(c)
            imageio.imwrite(os.path.join(png_path, '{}.png'.format(new_name2)), image.astype(np.uint8))

    tensor3 = tensor.clone()
    for b in range(batch_size):
        imageTensor = tensor3[b, :, :, :].unsqueeze(dim=0)
        imageTensor = F.one_hot(imageTensor.argmax(dim=1), C).permute(0, 3, 1, 2).float()
        imageNumpy = imageTensor.cpu().numpy()
        for c in range(C):
            image = imageNumpy[b][c] * 255.
            new_name3 = name + "_one_hot" + "_" + str(b) + "_" + str(c)
            imageio.imwrite(os.path.join(png_path, '{}.png'.format(new_name3)), image.astype(np.uint8))
