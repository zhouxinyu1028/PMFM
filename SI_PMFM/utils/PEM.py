import numpy as np
import torch
import torch.nn as nn


class PEM(nn.Module):
    def __init__(self, enable_neighborhood_expansion=True):
        super(PEM, self).__init__()
        self.camera_cz = 325.5
        self.camera_cx = 325.5
        self.camera_cy = 325.5
        self.camera_fx = 519.0
        self.camera_fy = 519.0
        self.offset = 0
        self.enable_neighborhood_expansion = enable_neighborhood_expansion

    @staticmethod
    def normalization(point_cloud):
        if len(point_cloud) == 0:
            return point_cloud.astype(np.float32)
        centered = point_cloud - np.mean(point_cloud, axis=0)
        max_dist = np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
        if max_dist == 0:
            return centered.astype(np.float32)
        return (centered / max_dist).astype(np.float32)

    def Change(self, image, result_points=None, result_orig_2d_points=None, result_index_C=None):
        index = torch.nonzero(image == 1).detach().cpu().numpy()
        if len(index) == 0:
            points = np.empty((0, 3), dtype=np.float32)
            orig_2d_points = np.empty((0, 2), dtype=np.int64)
            index_C = np.empty((0,), dtype=np.int64)
        else:
            index_C = index[:, 0].astype(np.int64)
            index_H = index[:, 1].astype(np.int64)
            index_W = index[:, 2].astype(np.int64)
            orig_2d_points = np.column_stack((index_H, index_W)).astype(np.int64)
            z = 1.0 - self.camera_cz
            z_point = index_C + self.camera_cz
            x_point = (index_H + self.camera_cx) * z / self.camera_fx
            y_point = (index_W + self.camera_cy) * z / self.camera_fy
            points = np.column_stack((x_point + self.offset, y_point + self.offset, z_point + self.offset))
            points = self.normalization(points)
        if result_points is None or result_orig_2d_points is None or result_index_C is None:
            return points, orig_2d_points, index_C
        return (np.concatenate((result_points, points)), np.concatenate((result_orig_2d_points, orig_2d_points)), np.concatenate((result_index_C, index_C)))

    def combine(self, ct_pred, mr_pred):
        points = np.empty((0, 3), dtype=np.float32)
        orig_2d_points = np.empty((0, 2), dtype=np.int64)
        index_C = np.empty((0,), dtype=np.int64)
        points, orig_2d_points, index_C = self.Change(ct_pred, points, orig_2d_points, index_C)
        points, orig_2d_points, index_C = self.Change(mr_pred, points, orig_2d_points, index_C)
        return points, orig_2d_points, index_C

    @staticmethod
    def _build_point_map(points, orig_2d_points, index_C):
        point_map = {}
        for point, orig_2d, class_id in zip(points, orig_2d_points, index_C):
            key = (int(class_id), int(orig_2d[0]), int(orig_2d[1]))
            point_map.setdefault(key, []).append(point)
        return point_map

    @staticmethod
    def _as_arrays(selected_keys, point_map):
        selected_points = []
        selected_orig_2d_points = []
        selected_index_C = []
        for key in selected_keys:
            class_id, row, col = key
            selected_points.append(np.mean(point_map[key], axis=0))
            selected_orig_2d_points.append([row, col])
            selected_index_C.append(class_id)
        if not selected_points:
            return (np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64))
        return (np.asarray(selected_points, dtype=np.float32), np.asarray(selected_orig_2d_points, dtype=np.int64), np.asarray(selected_index_C, dtype=np.int64))

    def focus_filter(self, ct_pred, mr_pred):
        ct_points, ct_orig_2d_points, ct_index_C = self.Change(ct_pred)
        mr_points, mr_orig_2d_points, mr_index_C = self.Change(mr_pred)
        ct_map = self._build_point_map(ct_points, ct_orig_2d_points, ct_index_C)
        mr_map = self._build_point_map(mr_points, mr_orig_2d_points, mr_index_C)
        merged_map = {}
        for key in set(ct_map) | set(mr_map):
            merged_map[key] = []
            if key in ct_map:
                merged_map[key].extend(ct_map[key])
            if key in mr_map:
                merged_map[key].extend(mr_map[key])
        intersection = set(ct_map) & set(mr_map)
        if not intersection:
            return self._as_arrays(sorted(merged_map), merged_map)
        focused_keys = set(intersection)
        if self.enable_neighborhood_expansion:
            offsets = ((0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0))
            for class_id, row, col in intersection:
                for dc, dr, dcol in offsets:
                    neighbor = (class_id + dc, row + dr, col + dcol)
                    if neighbor in merged_map:
                        focused_keys.add(neighbor)
        return self._as_arrays(sorted(focused_keys), merged_map)
