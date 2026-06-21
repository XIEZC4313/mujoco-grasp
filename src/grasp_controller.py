"""Grasp controller - pick and place demo."""

import numpy as np
import mujoco
from ik_solver import IKSolver


class GraspController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ik = IKSolver(model, data)

        self.finger_left_ctrl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_left")
        self.finger_right_ctrl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_right")

    def open_gripper(self):
        self.data.ctrl[self.finger_left_ctrl] = -1.0
        self.data.ctrl[self.finger_right_ctrl] = -1.0

    def close_gripper(self):
        self.data.ctrl[self.finger_left_ctrl] = 1.0
        self.data.ctrl[self.finger_right_ctrl] = 1.0

    def get_object_pos(self, object_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        return self.data.xpos[body_id].copy()

    def move_to(self, target_pos, steps=300, viewer=None, delay=0.001):
        import time
        joint_angles, success = self.ik.solve_position(target_pos, max_iter=500, tol=0.005)
        self.ik.set_joint_angles(joint_angles)
        # Reset only arm controls, preserve gripper state
        for i in range(5):  # act_joint1 to act_joint5
            self.data.ctrl[i] = 0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(delay)
        return success

    def incremental_adjust(self, delta_angles, n_steps=10, steps_per_step=50, viewer=None, delay=0.001):
        import time
        for _ in range(n_steps):
            current = self.ik.get_joint_angles()
            new_angles = [c + d / n_steps for c, d in zip(current, delta_angles)]
            self.ik.set_joint_angles(new_angles)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(steps_per_step):
                mujoco.mj_step(self.model, self.data)
                if viewer:
                    viewer.sync()
                    time.sleep(delay)

    def step(self, n=300, viewer=None, delay=0.001):
        import time
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(delay)

    def execute_pick_and_place(self, object_name, target_pos, viewer=None, delay=0.001):
        """Pick object from current position and place at target.

        Args:
            object_name: Name of object body
            target_pos: [x, y, z] position to place the object
            viewer: MuJoCo viewer (optional)
            delay: Step delay for visualization
        """
        obj_pos = self.get_object_pos(object_name)
        print(f"Object at: {obj_pos}")
        print(f"Target: {target_pos}")

        # === PICK ===
        # 1. Move above object
        print("1. Move above object")
        self.open_gripper()
        above = obj_pos.copy()
        above[2] += 0.06
        self.move_to(above, steps=300, viewer=viewer, delay=delay)

        # 2. Lower to object
        print("2. Lower to object")
        self.move_to(obj_pos, steps=300, viewer=viewer, delay=delay)

        # 3. Close gripper
        print("3. Close gripper")
        self.close_gripper()
        self.step(500, viewer, delay)

        # 4. Lift
        print("4. Lift object")
        self.incremental_adjust([0, 0.2, -0.2, 0, 0], n_steps=20, steps_per_step=50, viewer=viewer, delay=delay)
        self.step(300, viewer, delay)

        # === MOVE ===
        # 5. Rotate base toward target
        print("5. Move to target")
        current_angles = self.ik.get_joint_angles()
        target_angle = np.arctan2(target_pos[1], target_pos[0])
        delta_j1 = target_angle - current_angles[0]
        self.incremental_adjust([delta_j1, 0, 0, 0, 0], n_steps=15, steps_per_step=50, viewer=viewer, delay=delay)
        self.step(300, viewer, delay)

        # === PLACE ===
        # 6. Lower
        print("6. Lower to place")
        self.incremental_adjust([0, -0.2, 0.2, 0, 0], n_steps=20, steps_per_step=50, viewer=viewer, delay=delay)
        self.step(300, viewer, delay)

        # 7. Release
        print("7. Release")
        self.open_gripper()
        self.step(400, viewer, delay)

        # 8. Retreat
        print("8. Retreat")
        self.incremental_adjust([0, -0.1, 0.1, 0, 0], n_steps=10, steps_per_step=50, viewer=viewer, delay=delay)

        final_pos = self.get_object_pos(object_name)
        print(f"Object placed at: {final_pos}")
        print("Pick and place complete!")
