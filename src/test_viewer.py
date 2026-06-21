"""Quick test script to verify MuJoCo viewer works."""

import mujoco
import mujoco.viewer
import time
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("Launching MuJoCo viewer...")
print("Close the window to exit.")

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set initial camera
    viewer.cam.lookat[:] = [0.3, 0, 0.2]
    viewer.cam.distance = 1.0
    viewer.cam.elevation = -30
    viewer.cam.azimuth = 45

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

print("Viewer closed.")
