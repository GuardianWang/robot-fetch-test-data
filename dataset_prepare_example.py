from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from time import time
import numpngw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import math


def init_controller():
    fov = 45
    width = 600
    height = 300
    controller = Controller(platform=CloudRendering, fieldOfView=fov, width=width, height=height, gpu_device=0)
    controller.reset(renderDepthImage=True)

    return controller


def controller_step(controller):
    return controller.step(action='LookUp', degrees=10)


def save_rgb(controller):
    cv2.imwrite("image.jpg", controller.last_event.cv2img)


def save_depth(controller):
    numpngw.write_png('depth.png', (1000 * controller.last_event.depth_frame).astype(np.uint16))


def get_data(controller):
    pcd = get_point_cloud(controller)

    return {
        'point_cloud': pcd
    }


def viz_point_cloud(pcd):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


def get_point_cloud(controller):
    rgb = o3d.io.read_image("image.jpg")
    depth = o3d.io.read_image("depth.png")
    rgbd = o3d.geometry.RGBDImage()
    rgbd = rgbd.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False, depth_trunc=5)

    height, width, *_ = np.asarray(rgb).shape
    metadata = controller.last_event.metadata
    fov_y = metadata['fov']
    fx = fy = 0.5 * height / math.tan(math.radians(fov_y * 0.5))
    cx, cy = width / 2, height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # first element is math.radians(cameraHorizon)
    # second element is radians(-agent_rotation_y)
    agent = metadata['agent']
    rot_euler = np.array([math.radians(agent['cameraHorizon']), math.radians(-agent['rotation']['y']), 0])
    # looking at the outward axis, positive angle is rotating counterclockwise
    rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_euler)
    position = agent['position']
    position = [position['x'], position['y'], -position['z']]
    trans_vec = np.array(position).astype(np.float32)
    extrinsic = np.eye(4).astype(np.float32)
    extrinsic[:3, :3] = rot_mat

    pcd = o3d.geometry.PointCloud()

    pcd = pcd.create_from_rgbd_image(
        image=rgbd,
        intrinsic=intrinsic,
        extrinsic=extrinsic
    )

    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    pcd.translate(trans_vec)

    return pcd


if __name__ == '__main__':
    ctrl = init_controller()
    controller_step(ctrl)
    save_rgb(ctrl)
    save_depth(ctrl)

    data = get_data(ctrl)
    viz_point_cloud(data['point_cloud'])