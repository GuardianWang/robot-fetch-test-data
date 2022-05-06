import open3d as o3d
import numpy as np
import cv2
from scipy.io import loadmat
import os
from os.path import join


dataset_path = r"imvotenet/sunrgbd-toy"
rgb_path = join(dataset_path, "sunrgbd_trainval/image/000001.jpg")
depth_path = join(dataset_path, "sunrgbd_trainval/depth/000001.mat")
calib_path = join(dataset_path, "sunrgbd_trainval/calib/000001.txt")
label_path = join(dataset_path, "sunrgbd_trainval/label/000001.txt")


def viz_point_cloud():
    # point cloud from path
    depth_mat = loadmat(depth_path)["instance"]
    points = depth_mat[:, :3]
    # points[:, [1, 2]] = points[:, [2, 1]]
    colors = depth_mat[:, 3:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


if __name__ == "__main__":
    # viz_point_cloud()
    pass
