import open3d as o3d
import numpy as np

rgb = o3d.io.read_image("image.jpg")
depth = o3d.io.read_image("depth.png")
rgbd = o3d.geometry.RGBDImage()
rgbd = rgbd.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False)

pcd = o3d.geometry.PointCloud()

pcd = pcd.create_from_rgbd_image(
    image=rgbd,
    intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    extrinsic=np.eye(4)
)

# pcd = pcd.create_from_depth_image(
#              depth=depth,
#              intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
#              extrinsic=np.eye(4)
# )

# Each point corresponds to depth image in row-order
# The pixel depth[i][j] corresponds to pcd.points[width * i + j]
# np_pcd = np.asarray(pcd.points)[:100_000]
# pcd.points = o3d.utility.Vector3dVector(np_pcd)

# o3d.io.write_point_cloud("point-cloud.pcd", pcd, write_ascii=True, print_progress=True)
o3d.visualization.draw_geometries([pcd])
