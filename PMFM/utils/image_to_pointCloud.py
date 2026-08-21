import numpy as np
import torch
import torch.nn as nn


class ImageToPointCloud(nn.Module):
    def __init__(self, class_offset_scale=0.05, reference_depth=1.0):
        super(ImageToPointCloud, self).__init__()
        self.camera_cx = 325.5
        self.camera_cy = 325.5
        self.camera_fx = 519.0
        self.camera_fy = 519.0
        self.class_offset_scale = class_offset_scale
        self.reference_depth = reference_depth

    @staticmethod
    def normalization(point_cloud):
        point_cloud_centered = point_cloud - np.mean(point_cloud, axis=0)
        max_dist = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
        if max_dist == 0:
            return point_cloud_centered
        return point_cloud_centered / max_dist

    def Change(self, image, result_points=None, result_orig_2d_points=None, result_index_C=None):
        index = torch.nonzero(image == 1)
        index = index.cpu().numpy()

        if index.size == 0:
            points = np.empty((0, 3), dtype=np.float32)
            orig_2d_points = np.empty((0, 2), dtype=np.int64)
            index_C = np.empty((0,), dtype=np.int64)
        else:
            index_C = index[:, 0]
            index_H = index[:, 1]
            index_W = index[:, 2]
            orig_2d_points = np.column_stack((index_H, index_W))

            class_id = index_C + 1
            z_point = self.reference_depth + self.class_offset_scale * class_id
            x_point = (index_H - self.camera_cx) * z_point / self.camera_fx
            y_point = (index_W - self.camera_cy) * z_point / self.camera_fy
            points = np.column_stack((
                x_point,
                y_point,
                z_point,
            ))
            points = self.normalization(points)

        if (result_points is None) or (result_orig_2d_points is None) or (result_index_C is None):
            return points, orig_2d_points, index_C

        result_points = np.array([*result_points, *points])
        result_orig_2d_points = np.array([*result_orig_2d_points, *orig_2d_points])
        result_index_C = np.array([*result_index_C, *index_C])
        return result_points, result_orig_2d_points, result_index_C

    def combine(self, ct_pred, mr_pred):
        points = np.array([])
        orig_2d_points = np.array([])
        index_C = np.array([])
        points, orig_2d_points, index_C = self.Change(ct_pred, points, orig_2d_points, index_C)
        points, orig_2d_points, index_C = self.Change(mr_pred, points, orig_2d_points, index_C)
        return points, orig_2d_points, index_C
