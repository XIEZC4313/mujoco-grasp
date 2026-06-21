"""MuJoCo 5-DOF Arm - Green Block Detection and Pick-and-Place."""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grasp_controller import GraspController
from color_detector import ColorDetector


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    controller = GraspController(model, data)
    detector = ColorDetector(model, data)

    initial_angles = [0.0, -0.5, 0.8, 0.0, 0.0]
    controller.ik.set_joint_angles(initial_angles)
    controller.open_gripper()
    for i in range(5):
        data.ctrl[i] = 0
    mujoco.mj_forward(model, data)

    target_place = np.array([0.4, -0.12, 0.19])

    print("=== MuJoCo Green Block Detection Demo ===")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0, 0.12]
        viewer.cam.distance = 0.9
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 120

        mujoco.mj_resetData(model, data)
        controller.ik.set_joint_angles(initial_angles)
        controller.open_gripper()
        for i in range(5):
            data.ctrl[i] = 0
        mujoco.mj_forward(model, data)

        # Step 1: Detect colors
        print("Step 1: Scanning blocks...")
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()

        # Detect all colors
        detected = {}
        for color in ["red", "green", "blue"]:
            center, bbox, mask = detector.detect_color(color)
            if center:
                world_pos = detector.pixel_to_world(center)
                detected[color] = world_pos
                print(f"  {color}: pixel={center} -> world=[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]")

        # Use actual simulation position for accuracy
        green_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_block")
        green_pos = data.xpos[green_body_id].copy()
        print(f"\nGreen block (ground truth): [{green_pos[0]:.3f}, {green_pos[1]:.3f}, {green_pos[2]:.3f}]")

        # Save debug image
        detector.save_debug_image(os.path.join(os.path.dirname(__file__), "..", "debug_detection.png"))

        # Step 2: Pick and place
        print(f"\nStep 2: Pick green block -> place at {target_place}")
        controller.execute_pick_and_place("green_block", target_place, viewer=viewer, delay=0.002)

        # Show final positions
        print("\n=== Final positions ===")
        for name in ["red_block", "green_block", "blue_block"]:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = data.xpos[bid]
            print(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        print("\nKeeping viewer open... Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
