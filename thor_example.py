from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from time import time
import numpngw
import numpy as np


controller = Controller(platform=CloudRendering)
# controller = Controller()
print("controller initialized")

renderDepthImage = True
renderInstanceSegmentation = False
renderSemanticSegmentation = False
renderNormalsImage = False

controller.reset(
    # makes the images a bit higher quality
    width=800,
    height=800,

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

numpngw.write_png('depth.png', (1000 * controller.last_event.depth_frame).astype(np.uint16))

n_step = 10
start_time = time()
for _ in range(n_step):
    event = controller.step("MoveAhead")
end_time = time()
print(f"{n_step/(end_time - start_time)} fps")
