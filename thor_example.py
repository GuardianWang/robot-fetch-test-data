from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from time import time
import numpngw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def create_bbox_from_points(points):
    # points: [8, 3]
    points = np.asarray(points)
    # reverse z in o3d coordinates
    points[:, 2] *= -1
    lines = [[0, 1], [1, 3], [2, 3], [0, 2],
             [4, 5], [5, 7], [6, 7], [4, 6],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


# fieldOfView controls the fov on y axis
controller = Controller(platform=CloudRendering, fieldOfView=45, width=800, height=300, gpu_device=0)
# controller = Controller(fieldOfView=45, width=600, height=300, gpu_device=0)
print("controller initialized")

renderDepthImage = True
renderInstanceSegmentation = True
renderSemanticSegmentation = True
renderNormalsImage = True

controller.reset(
    # makes the images a bit higher quality
    # width=800,
    # height=800,

    # Renders several new image modalities
    renderDepthImage=renderDepthImage,
    renderInstanceSegmentation=renderInstanceSegmentation,
    renderSemanticSegmentation=renderSemanticSegmentation,
    renderNormalsImage=renderNormalsImage
)

# adds a cameras from a third-person's point of view
scene_bounds = controller.last_event.metadata['sceneBounds']['center']
controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=0, y=1.55, z=-2.3),
    rotation=dict(x=20, y=0, z=0)
)

# adds an orthographic top-down image
controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=scene_bounds['x'], y=2.5, z=scene_bounds['z']),
    rotation=dict(x=90, y=0, z=0),
    orthographic=True,
    orthographicSize=3.25,
    skyboxColor="white"
)

# x > 0, move right
# y > 0, move upward
# z > 0, move forward
# rotation
# x > 0, look down
# y > 0, rotate right
# z > 0, rotate counterclockwise
event = controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=0, y=1, z=0),
    rotation=dict(x=45, y=90, z=0),
    fieldOfView=45,
)

camera_rgb = event.third_party_camera_frames[-1][..., :3]
camera_depth = event.third_party_depth_frames[-1]
# cv2.imwrite("image.jpg", cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR))
# numpngw.write_png('depth.png', (1000 * camera_depth).astype(np.uint16))
# plt.imshow(camera_rgb)
# plt.show()

controller.step(action='LookUp', degrees=10)
cv2.imwrite("image.jpg", controller.last_event.cv2img)
numpngw.write_png('depth.png', (1000 * controller.last_event.depth_frame).astype(np.uint16))

objects = controller.last_event.metadata['objects']
viz_objs = [x for x in objects if x['visible']]
aabboxes = [x['axisAlignedBoundingBox'] for x in viz_objs]
oobboxes = [x['objectOrientedBoundingBox'] for x in viz_objs]


n_step = 0
start_time = time()
for _ in range(n_step):
    event = controller.step("MoveAhead")
end_time = time()
print(f"{n_step/(end_time - start_time)} fps")

aabboxes_o3d = [create_bbox_from_points(x['cornerPoints']) for x in aabboxes]
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([mesh_frame, *aabboxes_o3d], lookat=[0, 0, -1], up=[0, 1, 0], front=[0, 0, 1], zoom=1)

