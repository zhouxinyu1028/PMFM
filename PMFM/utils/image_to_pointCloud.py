import os.path
import time

import numpy as np
import cv2
try:
    import open3d as o3d
except ImportError:
    o3d = None  # Optional viewer; point coordinates use NumPy/Torch only.
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random


class ImageToPointCloud(nn.Module):

    def __init__(self, show_point_cloud=False, is_save_point_cloud=False, save_path="", png_size=(128, 128)):
        super(ImageToPointCloud, self).__init__()
        # Camera intrinsics
        self.camera_factor = 1000
        self.camera_cx = 325.5
        self.camera_cy = 325.5
        self.camera_cz = 325.5
        self.camera_fx = 519.0
        self.camera_fy = 519.0
        # self.camera_cx = 319.5
        # self.camera_cy = 239.5
        # self.camera_fx = 525.0
        # self.camera_fy = 525.0
        self.offset = 0

        self.show_point_cloud = show_point_cloud
        self.is_save_point_cloud = is_save_point_cloud
        self.save_path = save_path
        self.png_size = png_size

    def getRandomColor(self, point):

        
        r = random.randint(0, 255) / 255
        g = random.randint(0, 255) / 255
        b = random.randint(0, 255) / 255

        color = [r, g, b]
        colors = np.array([color for _ in range(len(point))])

        return colors

    def getTime(self):
        """
            获得当前时间字符串
        :return: str
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d(%H_%M_%S)")

    def visualization(self, points):
        """
            点云数据可视化
        :param points: list类型，points[i]都是numpy类型的(point_number, 3)数据
        :return:
        """
        visual_list = []  # 可视化列表
        C = len(points)
        for c in range(C):
            point = points[c]

            if np.all(np.isnan(point)):
                continue

            # 得到一种随机颜色
            color = self.getRandomColor(point)
            # 创建 Open3D 点云对象
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(point)
            point_cloud.colors = o3d.utility.Vector3dVector(color)  # 设置颜色

            visual_list.append(point_cloud)

        return visual_list

    def show(self, visual_list):
        # 可视化点云
        o3d.visualization.draw_geometries(visual_list, window_name="Point Cloud Visualization")

    def save_png(self, visual_list):
        # 使用离屏渲染创建视觉对象
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600, visible=False)  # 设置 visible=False 进行离屏渲染
        for point_cloud in visual_list:
            vis.add_geometry(point_cloud)
        # 渲染并捕获图像
        for point_cloud in visual_list:
            vis.update_geometry(point_cloud)

        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)

        # 将 Open3D 图像转换为 NumPy 数组
        image_np = np.asarray(image)
        image_np = (image_np * 255).astype(np.uint8)  # 转换为 uint8 类型

        # 缩放图像
        target_size = self.png_size  # 目标尺寸
        resized_image = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)

        new_name = self.getTime() + ".png"

        # 保存图像
        # o3d.io.write_image(os.path.join(self.save_path, new_name), image)
        cv2.imwrite(os.path.join(self.save_path, new_name), resized_image)

        # 销毁视觉对象
        vis.destroy_window()

    def normalization(self, point_cloud):
        # 中心化
        point_cloud_centered = point_cloud - np.mean(point_cloud, axis=0)

        # 缩放
        max_dist = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
        point_cloud_normalized = point_cloud_centered / max_dist

        return point_cloud_normalized

    def Change(self, image, result_points=None, result_orig_2d_points=None, result_index_C=None):
        """
            C是类别数
        :param result_index_C:
        :param result_orig_2d_points:
        :param result_points:
        :param image: tensor类型，尺寸(C, H, W)
        :return: (C, point_number, 3), list
        """

        # 由于本论文的image是0和1的二进制图片依次,z是确定的
        z = 1. - self.camera_cz
        # 得到image中含有1的下标
        # image_rot90 = torch.rot90(image, k=1, dims=(1, 2))
        index = torch.nonzero(image == 1)
        index = index.cpu().numpy()
        # 在把index尺寸为(值为1的数量, 3)，3的含义是(C, H, W),把(C, H, W)拆分出来(列)
        index_C = index[:, 0]
        index_H = index[:, 1]
        index_W = index[:, 2]
        # 用于提取重复坐标和非重复坐标
        # coordinates = np.column_stack((np.array([*index_C]), np.array([*index_H]), np.array([*index_W])))
        # 得到原始坐标信息(二维)
        orig_2d_points = np.column_stack((np.array([*index_H]), np.array([*index_W])))
        # 得到三维坐标
        z_point = index_C + self.camera_cz
        x_point = (index_H + self.camera_cx) * z / self.camera_fx
        y_point = (index_W + self.camera_cy) * z / self.camera_fy

        points = np.column_stack((x_point + self.offset, y_point + self.offset, z_point + self.offset))
        # 该模态当前没有任何正体素时 points 为空，normalization 里的 np.max 会抛
        # "zero-size array to reduction operation maximum"，这里直接跳过归一化
        if points.shape[0] == 0:
            points = np.zeros((0, 3), dtype=np.float64)
        else:
            points = self.normalization(points)
        if self.show_point_cloud or self.is_save_point_cloud:
            points_categories = np.column_stack((index_C, points))
            categories = np.unique(points_categories[:, 0])
            points_list = [points_categories[points_categories[:, 0] == cat][:, 1:] for cat in categories]
            visual_list = self.visualization(points_list)
            if self.show_point_cloud:
                self.show(visual_list)
            if self.is_save_point_cloud:
                self.save_png(visual_list)
        if (result_points is None) or (result_orig_2d_points is None) or (result_index_C is None):
            return points, orig_2d_points, index_C
        else:
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


if __name__ == "__main__":

    # -------------测试ImageToPointCloud------
    images_numpy1 = cv2.imread(r"E:\datesets\HaNSeg\image_set\2024131\270-463-66\z\edges\A_Carotid_L\1_1.jpg")
    images_numpy2 = cv2.imread(r"E:\datesets\HaNSeg\image_set\2024131\270-463-66\z\edges\A_Carotid_R\1_1.jpg")
    images_numpy3 = cv2.imread(r"E:\datesets\HaNSeg\image_set\2024131\270-463-66\z\edges\Glottis\1_1.jpg")
    # images_numpy4 = cv2.imread(r"E:\datesets\HaNSeg\image_set\new\Parotid_R.jpg")
    # images_numpy5 = cv2.imread(r"E:\datesets\HaNSeg\image_set\new\Lips.jpg")
    # images_numpy2 = cv2.imread(r"E:\datesets\HaNSeg\image_set\3d\label_3.png")
    # images_numpy3 = cv2.imread(r"E:\datesets\HaNSeg\image_set\3d\label_19.png")
    # images_numpy4 = cv2.imread(r"E:\datesets\HaNSeg\image_set\3d\label_6.png")

    images_numpy1 = np.transpose(images_numpy1, (2, 0, 1))
    images_numpy2 = np.transpose(images_numpy2, (2, 0, 1))
    images_numpy3 = np.transpose(images_numpy3, (2, 0, 1))
    # images_numpy4 = np.transpose(images_numpy4, (2, 0, 1))
    # images_numpy5 = np.transpose(images_numpy5, (2, 0, 1))
    # images_numpy3 = np.transpose(images_numpy3, (2, 0, 1))
    # images_numpy4 = np.transpose(images_numpy4, (2, 0, 1))
    images1 = torch.from_numpy(images_numpy1)
    images2 = torch.from_numpy(images_numpy2)
    images3 = torch.from_numpy(images_numpy3)
    # images4 = torch.from_numpy(images_numpy4)
    # images5 = torch.from_numpy(images_numpy5)
    # images3 = torch.from_numpy(images_numpy3)
    # images4 = torch.from_numpy(images_numpy4)

    print("images1", images_numpy1.shape, images_numpy1.dtype)
    print("images2", images_numpy2.shape, images_numpy2.dtype)
    print("images3", images_numpy3.shape, images_numpy3.dtype)


    # images = torch.cat((images1[0].unsqueeze(dim=0), images2[0].unsqueeze(dim=0), images3[0].unsqueeze(dim=0), images4[0].unsqueeze(dim=0), images5[0].unsqueeze(dim=0)), dim=0)
    images = torch.cat((images1[0].unsqueeze(dim=0), images2[0].unsqueeze(dim=0), images3[0].unsqueeze(dim=0)),  dim=0)
    # images = images1[0].unsqueeze(dim=0)
    images = (images > 35).float()
    obj = ImageToPointCloud(show_point_cloud=True, is_save_point_cloud=False, save_path=r"E:\datesets\HaNSeg\image_set\3d")
    points, _, _ = obj.Change(images)
    print(len(points))


