import open3d as o3d
import numpy as np
import math

rgb = o3d.io.read_image("image.jpg")
depth = o3d.io.read_image("depth.png")
rgbd = o3d.geometry.RGBDImage()
rgbd = rgbd.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False, depth_trunc=5)

height, width, *_ = np.asarray(rgb).shape
fov_y = 45
fx = fy = 0.5 * height / math.tan(math.radians(fov_y * 0.5))
cx, cy = width / 2, height / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# first element is math.radians(cameraHorizon)
# second element is radians(-agent_rotation_y)
rot_euler = np.array([math.radians(45), math.radians(-90), 0])
# looking at the outward axis, positive angle is rotating counterclockwise
# but in ai2thor, looking at the outward axis, positive angle is rotating clockwise
rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(rot_euler)
# cameraPosition
# camera to world
# use x, y, -z
trans_vec = np.array([0, 1, 0]).astype(np.float32)
extrinsic = np.eye(4).astype(np.float32)
extrinsic[:3, :3] = rot_mat
# extrinsic[:3, 3] = trans_vec

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

# pcd = pcd.create_from_depth_image(
#     depth=depth,
#     intrinsic=intrinsic,
#     extrinsic=extrinsic
# )

# Each point corresponds to depth image in row-order
# The pixel depth[i][j] corresponds to pcd.points[width * i + j]
# np_pcd = np.asarray(pcd.points)[:100_000]
# pcd.points = o3d.utility.Vector3dVector(np_pcd)

# z is outward in open3d
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.io.write_point_cloud("point-cloud.pcd", pcd, write_ascii=True, print_progress=True)
o3d.visualization.draw_geometries([pcd, mesh_frame], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)
