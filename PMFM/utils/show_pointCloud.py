import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def show_pointCloud(point):
    """
    :param point: (n, 3)
    :return:
    """
    number_points = point.shape[0]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter3D(point[:, 0], point[:, 1], point[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud")
    plt.show()


def point_distribution(point, target_dis):

    point = torch.from_numpy(point)
    total_point_number = point.shape[0]  # 总共的点数
    point_dis = point.pow(2).sum(dim=-1)  # 求距离0点的公式，不带根号
    target_dis = target_dis * target_dis  # 参考距离公式平方
    cnt = (point_dis < target_dis).float().sum().item()
    print(cnt, total_point_number, cnt / total_point_number)
    if cnt / total_point_number >= 0.3:
        return True
    else:
        return False


def PointCloudShow(point):
    """
    :param point: (batch_size, n, 3)
    :return:
    """
    batch_size, _, _ = point.shape

    for b in range(batch_size):
        show_pointCloud(point[b])


