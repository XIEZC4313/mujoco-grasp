"""MPC control demo - trajectory tracking with closed-loop control."""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpc_controller import MPCController
from ik_solver import IKSolver


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    mpc = MPCController(model, data, Np=8, Nc=4, dt=0.02)
    ik = IKSolver(model, data)

    initial_angles = [0.0, -0.5, 0.8, 0.0, 0.0]

    # Target positions for trajectory tracking demo
    targets = [
        np.array([0.35, 0.0, 0.25]),   # Above table center
        np.array([0.35, 0.1, 0.20]),   # Right side
        np.array([0.35, -0.1, 0.20]),  # Left side
        np.array([0.30, 0.0, 0.15]),   # Lower center
    ]

    print("=== MPC Control Demo ===")
    print(f"Prediction horizon: {mpc.Np}")
    print(f"Control horizon: {mpc.Nc}")
    print(f"Control dt: {mpc.dt}s")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0, 0.12]
        viewer.cam.distance = 0.9
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 120

        # Initialize
        mujoco.mj_resetData(model, data)
        ik.set_joint_angles(initial_angles)
        for i in range(5):
            data.ctrl[i] = 0
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_left")] = -1.0
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_right")] = -1.0
        mujoco.mj_forward(model, data)

        print("Initial state:")
        p0 = mpc.get_ee_pos()
        print(f"  End-effector: [{p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f}]")
        print()

        # Track each target
        for i, target in enumerate(targets):
            print(f"--- Target {i+1}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] ---")

            # Get IK reference
            q_ref, ik_success = ik.solve_position(target, max_iter=500, tol=0.005)
            print(f"  IK solution: {'success' if ik_success else 'failed'}")

            # MPC tracking
            success, errors = mpc.move_to_target(
                target, target_q=q_ref, max_steps=200,
                tol=0.008, viewer=viewer, delay=0.002
            )

            if errors:
                print(f"  Final error: {errors[-1]:.4f}")
                print(f"  Min error: {min(errors):.4f}")
                print(f"  Steps: {len(errors)}")
            print()

            # Hold position
            for _ in range(100):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.005)

        # Final summary
        print("=== MPC Demo Complete ===")
        p_final = mpc.get_ee_pos()
        print(f"Final end-effector: [{p_final[0]:.3f}, {p_final[1]:.3f}, {p_final[2]:.3f}]")

        # Keep viewer open
        print("\nKeeping viewer open... Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
