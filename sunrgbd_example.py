import open3d as o3d
import numpy as np
import cv2
from scipy.io import loadmat
import os
from os.path import join


dataset_path = r"imvotenet/sunrgbd"
sample_id = 1
rgb_path = join(dataset_path, "sunrgbd_trainval/image/{:06d}.jpg".format(sample_id))
# generated by sunrgbd_data.py
depth_path = join(dataset_path, "sunrgbd_pc_bbox_votes_50k_v1_val/{:06d}_pc.npz".format(sample_id))
calib_path = join(dataset_path, "sunrgbd_trainval/calib/{:06d}.txt".format(sample_id))
label_path = join(dataset_path, "sunrgbd_trainval/label/{:06d}.txt".format(sample_id))
bbox_2d_path = join(dataset_path, "sunrgbd_2d_bbox_50k_v1_val/{:06d}.txt".format(sample_id))
bbox_3d_path = join(dataset_path, "sunrgbd_pc_bbox_votes_50k_v1_val/{:06d}_bbox.npy".format(sample_id))

# sunrgbd_utils.py
type2class = {'bed': 0, 'table': 1, 'sofa': 2, 'chair': 3, 'toilet': 4, 'desk': 5, 'dresser': 6, 'night_stand': 7,
              'bookshelf': 8, 'bathtub': 9}
class2type = {type2class[t]: t for t in type2class}


def get_pcd():
    # point cloud from path
    depth_mat = np.load(depth_path)['pc']
    points = depth_mat[:, :3]
    # points[:, [1, 2]] = points[:, [2, 1]]
    colors = depth_mat[:, 3:]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def viz_point_cloud():
    pcd = get_pcd()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


def viz_3d_bbox():
    pcd = get_pcd()
    bboxes_3d = np.load(bbox_3d_path)
    o3d_bboxes = []
    for bbox_3d in bboxes_3d:
        o3d_bbox = o3d.geometry.OrientedBoundingBox(center=bbox_3d[:3], R=np.eye(3), extent=2 * bbox_3d[3:6])
        o3d_bboxes.append(o3d_bbox)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, mesh_frame, *o3d_bboxes], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)


def viz_2d_bbox():
    prob_thr = 0.1
    # sungrbd_utils.py SUNObject3d
    label_bboxes = np.loadtxt(label_path, usecols=tuple(range(1, 13)), dtype='f4')
    label_bboxes[:, [2, 3]] += label_bboxes[:, [0, 1]]
    label_names = np.loadtxt(label_path, usecols=0, dtype='S')
    # authors used other detectors for imvotenet
    # although the detector results are not too accurate
    label_bboxes_2d = np.loadtxt(bbox_2d_path, usecols=tuple(range(4, 9)), dtype='f4')
    probs = label_bboxes_2d[:, -1]
    label_bboxes_2d_names = np.loadtxt(bbox_2d_path, usecols=0, dtype='S')
    label_bboxes_2d_names = label_bboxes_2d_names[probs > prob_thr]
    label_bboxes_2d = label_bboxes_2d[probs > prob_thr][:, :4]
    labels = {
        "classes": label_names.astype(str),
        "2dbboxes": label_bboxes[:, :4],
        "3dbboxes": label_bboxes[:, 4:],
    }
    labels_bboxes_2d = {
        "classes": label_bboxes_2d_names.astype(str),
        "2dbboxes": label_bboxes_2d,
    }

    rgbs = []
    for label_src in (labels, labels_bboxes_2d):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        # cv2, upper left is (0, 0)
        for name, (x1, y1, x2, y2) in zip(label_src["classes"], label_src["2dbboxes"].astype('i4')):
            cv2.putText(rgb, name, (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 0, 0))
        rgbs.append(rgb)

    rgb = np.column_stack(rgbs)
    cv2.imwrite("sunrgbd-2dbbox.jpg", rgb)
    cv2.imshow("image", rgb)
    cv2.waitKey()


if __name__ == "__main__":
    # viz_point_cloud()
    # viz_2d_bbox()
    viz_3d_bbox()
    pass
