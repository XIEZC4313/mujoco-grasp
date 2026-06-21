"""Inverse Kinematics solver using Jacobian iterative method."""

import numpy as np
import mujoco


class IKSolver:
    def __init__(self, model, data, site_name="end_effector"):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        # Joint IDs for the 5-DOF arm
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]

        # Get joint address in qpos
        self.joint_qpos_addrs = [
            model.jnt_qposadr[jid] for jid in self.joint_ids
        ]

        # Get joint address in qvel/ctrl
        self.joint_dof_addrs = [
            model.jnt_dofadr[jid] for jid in self.joint_ids
        ]

        self.dof = len(self.joint_names)

    def get_site_pos(self):
        """Get current end-effector position."""
        return self.data.site_xpos[self.site_id].copy()

    def get_site_mat(self):
        """Get current end-effector rotation matrix."""
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def compute_jacobian(self):
        """Compute Jacobian for end-effector position."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        # Extract only the columns for our 5 joints
        jac_pos = jacp[:, self.joint_dof_addrs]
        jac_rot = jacr[:, self.joint_dof_addrs]
        return jac_pos, jac_rot

    def solve_position(self, target_pos, max_iter=100, tol=1e-3, alpha=0.5, keep_down=False):
        """Solve IK for target end-effector position.

        Args:
            target_pos: Target position [x, y, z]
            max_iter: Maximum iterations
            tol: Position tolerance
            alpha: Step size
            keep_down: If True, constrain gripper to point downward

        Returns:
            joint_angles: Solution joint angles
            success: Whether convergence was achieved
        """
        target_pos = np.array(target_pos)

        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.get_site_pos()
            pos_error = target_pos - current_pos

            if keep_down:
                # Add orientation constraint: gripper z-axis should point down [0,0,-1]
                current_mat = self.get_site_mat()
                gripper_z = current_mat[:, 2]  # z-axis of gripper
                target_z = np.array([0, 0, -1])
                orient_error = np.cross(gripper_z, target_z)

                # Combined error (position + partial orientation)
                error = np.concatenate([pos_error, 0.3 * orient_error])

                jac_pos, jac_rot = self.compute_jacobian()
                jac_full = np.vstack([jac_pos, 0.3 * jac_rot])

                lam = 0.1
                delta_q = jac_full.T @ np.linalg.solve(
                    jac_full @ jac_full.T + lam**2 * np.eye(6), error
                )
            else:
                if np.linalg.norm(pos_error) < tol:
                    return self._get_joint_angles(), True

                jac_pos, _ = self.compute_jacobian()
                lam = 0.1
                delta_q = jac_pos.T @ np.linalg.solve(
                    jac_pos @ jac_pos.T + lam**2 * np.eye(3), pos_error
                )

            for i, addr in enumerate(self.joint_qpos_addrs):
                self.data.qpos[addr] += alpha * delta_q[i]

            # Clamp to joint limits
            for i, jid in enumerate(self.joint_ids):
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    self.data.qpos[self.joint_qpos_addrs[i]] = np.clip(
                        self.data.qpos[self.joint_qpos_addrs[i]], lo, hi
                    )

        # Check convergence
        mujoco.mj_forward(self.model, self.data)
        final_error = np.linalg.norm(target_pos - self.get_site_pos())
        return self._get_joint_angles(), final_error < tol * 5  # Relaxed tolerance

    def solve_pose(self, target_pos, target_rot, max_iter=200, tol_pos=1e-3, tol_rot=1e-2, alpha=0.3):
        """Solve IK for target end-effector pose (position + orientation).

        Args:
            target_pos: Target position [x, y, z]
            target_rot: Target rotation matrix (3x3)
            max_iter: Maximum iterations
            tol_pos: Position tolerance
            tol_rot: Orientation tolerance
            alpha: Step size

        Returns:
            joint_angles: Solution joint angles
            success: Whether convergence was achieved
        """
        target_pos = np.array(target_pos)
        target_rot = np.array(target_rot)

        for _ in range(max_iter):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.get_site_pos()
            current_rot = self.get_site_mat()

            # Position error
            pos_error = target_pos - current_pos

            # Orientation error (using rotation matrix difference)
            rot_error = 0.5 * (
                np.cross(current_rot[:, 0], target_rot[:, 0]) +
                np.cross(current_rot[:, 1], target_rot[:, 1]) +
                np.cross(current_rot[:, 2], target_rot[:, 2])
            )

            # Combined error
            error = np.concatenate([pos_error, rot_error])

            if np.linalg.norm(pos_error) < tol_pos and np.linalg.norm(rot_error) < tol_rot:
                return self._get_joint_angles(), True

            # Compute full Jacobian
            jac_pos, jac_rot = self.compute_jacobian()
            jac_full = np.vstack([jac_pos, jac_rot])

            # Damped least squares
            lam = 0.1
            delta_q = jac_full.T @ np.linalg.solve(
                jac_full @ jac_full.T + lam**2 * np.eye(6), error
            )

            # Apply joint angle update
            for i, addr in enumerate(self.joint_qpos_addrs):
                self.data.qpos[addr] += alpha * delta_q[i]

        return self._get_joint_angles(), False

    def _get_joint_angles(self):
        """Get current joint angles."""
        return [self.data.qpos[addr] for addr in self.joint_qpos_addrs]

    def set_joint_angles(self, angles):
        """Set joint angles in the model."""
        for i, addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[addr] = angles[i]

    def get_joint_angles(self):
        """Get current joint angles."""
        return [self.data.qpos[addr] for addr in self.joint_qpos_addrs]
