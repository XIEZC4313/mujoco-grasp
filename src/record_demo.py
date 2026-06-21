"""Record demo video - offscreen rendering with ffmpeg pipe."""

import numpy as np
import mujoco
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grasp_controller import GraspController
from color_detector import ColorDetector

WIDTH, HEIGHT = 1280, 720
FPS = 30
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "submission", "demo.mp4")


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    controller = GraspController(model, data)
    detector = ColorDetector(model, data)

    initial_angles = [0.0, -0.5, 0.8, 0.0, 0.0]

    # Camera setup
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0, 0.12]
    cam.distance = 0.9
    cam.elevation = -30
    cam.azimuth = 120

    # Start ffmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        OUTPUT_PATH,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_count = 0

    def render_frame():
        nonlocal frame_count
        renderer.update_scene(data, cam)
        pixels = renderer.render()
        proc.stdin.write(pixels.tobytes())
        frame_count += 1

    def step_and_render(n=1):
        for _ in range(n):
            mujoco.mj_step(model, data)
            render_frame()

    def hold(seconds=1.0):
        for _ in range(int(seconds * FPS)):
            mujoco.mj_step(model, data)
            render_frame()

    print("Recording demo video...")

    # === Scene 1: Opening - show environment ===
    print("  Scene 1: Environment overview")
    mujoco.mj_resetData(model, data)
    controller.ik.set_joint_angles(initial_angles)
    controller.open_gripper()
    for i in range(5):
        data.ctrl[i] = 0
    mujoco.mj_forward(model, data)
    hold(2.0)

    # Slow orbit
    for i in range(90):
        cam.azimuth = 120 + i * 1.5
        mujoco.mj_step(model, data)
        render_frame()
    hold(1.0)

    # Reset camera
    cam.azimuth = 120

    # === Scene 2: Color detection ===
    print("  Scene 2: Color detection")
    mujoco.mj_resetData(model, data)
    controller.ik.set_joint_angles(initial_angles)
    controller.open_gripper()
    for i in range(5):
        data.ctrl[i] = 0
    mujoco.mj_forward(model, data)
    hold(1.0)

    # Top-down view for detection
    cam.elevation = -60
    cam.distance = 0.7
    cam.lookat[:] = [0.35, 0, 0.16]
    hold(2.0)

    # Reset camera to main view
    cam.lookat[:] = [0.3, 0, 0.12]
    cam.distance = 0.9
    cam.elevation = -30

    # === Scene 3: Pick and Place (MPC Control) ===
    print("  Scene 3: Pick and place (MPC)")
    mujoco.mj_resetData(model, data)
    controller.ik.set_joint_angles(initial_angles)
    controller.open_gripper()
    for i in range(5):
        data.ctrl[i] = 0
    mujoco.mj_forward(model, data)
    hold(1.0)

    obj_pos = controller.get_object_pos("green_block")
    target_place = np.array([0.4, -0.12, 0.19])

    # 1. Move above object (MPC)
    print("    1. Move above object (MPC)")
    above = obj_pos.copy()
    above[2] += 0.06
    controller.open_gripper()
    controller.move_to_mpc(above, max_steps=150, viewer=None, delay=0)
    # Render the MPC motion
    for _ in range(int(1.0 * FPS)):
        mujoco.mj_step(model, data)
        render_frame()

    # 2. Lower to object (MPC)
    print("    2. Lower to object (MPC)")
    controller.move_to_mpc(obj_pos, max_steps=150, viewer=None, delay=0)
    for _ in range(int(0.5 * FPS)):
        mujoco.mj_step(model, data)
        render_frame()

    # 3. Close gripper
    print("    3. Close gripper")
    controller.close_gripper()
    hold(2.0)

    # 4. Lift
    print("    4. Lift object")
    controller.incremental_adjust([0, 0.2, -0.2, 0, 0], n_steps=20, steps_per_step=2)
    hold(1.0)

    # 5. Rotate to target
    print("    5. Move to target")
    current_angles = controller.ik.get_joint_angles()
    target_angle = np.arctan2(target_place[1], target_place[0])
    delta_j1 = target_angle - current_angles[0]
    controller.incremental_adjust([delta_j1, 0, 0, 0, 0], n_steps=15, steps_per_step=2)
    hold(1.0)

    # 6. Lower
    print("    6. Lower to place")
    controller.incremental_adjust([0, -0.2, 0.2, 0, 0], n_steps=20, steps_per_step=2)
    hold(0.5)

    # 7. Release
    print("    7. Release")
    controller.open_gripper()
    hold(1.5)

    # 8. Retreat
    print("    8. Retreat")
    controller.incremental_adjust([0, -0.1, 0.1, 0, 0], n_steps=10, steps_per_step=2)
    hold(1.0)

    # === Scene 4: Final view ===
    print("  Scene 4: Final result")
    for i in range(60):
        cam.azimuth = 120 + i * 2
        mujoco.mj_step(model, data)
        render_frame()
    hold(2.0)

    proc.stdin.close()
    proc.wait()

    print(f"\nVideo saved to: {OUTPUT_PATH}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {frame_count / FPS:.1f}s")


if __name__ == "__main__":
    main()
