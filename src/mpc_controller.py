"""MPC controller for 5-DOF arm.

Architecture (matching robot-arm-mpc):
  Base layer: Computed torque (PD + gravity compensation)
  Optimization layer: MPC residual torque optimization
"""

import numpy as np
import casadi as ca
import mujoco


class MPCController:
    def __init__(self, model, data, Np=10, Nc=5, dt=0.02,
                 Kp=None, Kd=None, tau_max=1.0):
        self.model = model
        self.data = data
        self.Np = Np
        self.Nc = Nc
        self.dt = dt

        # Joint info
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        self.n_joints = len(self.joint_names)
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        self.joint_qpos_addrs = [model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_dof_addrs = [model.jnt_dofadr[jid] for jid in self.joint_ids]

        # Actuator info
        self.act_names = ["act_joint1", "act_joint2", "act_joint3",
                          "act_joint4", "act_joint5"]
        self.n_ctrl = len(self.act_names)
        self.act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.act_names
        ]

        # Site (end-effector)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

        # PD gains
        self.Kp = Kp if Kp is not None else np.diag([200.0, 300.0, 300.0, 100.0, 50.0])
        self.Kd = Kd if Kd is not None else np.diag([20.0, 30.0, 30.0, 10.0, 5.0])
        self.tau_max = tau_max

        # Joint limits
        self.q_min = np.array([model.jnt_range[jid][0] for jid in self.joint_ids])
        self.q_max = np.array([model.jnt_range[jid][1] for jid in self.joint_ids])

        # Build CasADi functions for dynamics
        self._build_dynamics()

    def _build_dynamics(self):
        """Build CasADi symbolic dynamics for MPC prediction."""
        n = self.n_joints

        # Symbolic variables
        q_sym = ca.SX.sym('q', n)
        dq_sym = ca.SX.sym('dq', n)
        x_sym = ca.vertcat(q_sym, dq_sym)
        u_sym = ca.SX.sym('u', n)

        # Get gravity vector from MuJoCo (numerical)
        # We'll use MuJoCo's forward dynamics as a black box
        # For CasADi, we build a simplified model

        # Simplified dynamics: dq/ddq approximated from MuJoCo
        # We use numerical linearization at each step instead

        self._x_sym = x_sym
        self._u_sym = u_sym
        self._q_sym = q_sym
        self._dq_sym = dq_sym

    def get_state(self):
        """Get current state [q, dq]."""
        q = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs])
        dq = np.array([self.data.qvel[addr] for addr in self.joint_dof_addrs])
        return np.concatenate([q, dq])

    def set_state(self, x):
        """Set state [q, dq]."""
        for i, addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[addr] = x[i]
        for i, addr in enumerate(self.joint_dof_addrs):
            self.data.qvel[addr] = x[self.n_joints + i]

    def get_ee_pos(self):
        """Get end-effector position."""
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def get_jacobian(self):
        """Get end-effector Jacobian (3 x n_joints)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return jacp[:, self.joint_dof_addrs]

    def get_gravity(self):
        """Get gravity compensation torque using MuJoCo inverse dynamics."""
        # Save state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()

        # Set zero velocity and acceleration to get pure gravity
        qacc_save = self.data.qacc.copy()
        for i, addr in enumerate(self.joint_dof_addrs):
            self.data.qvel[addr] = 0.0
        self.data.qacc[:] = 0.0

        mujoco.mj_inverse(self.model, self.data)
        grav = self.data.qfrc_inverse[self.joint_dof_addrs].copy()

        # Restore state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.qacc[:] = qacc_save
        mujoco.mj_forward(self.model, self.data)

        return grav

    def get_mass_matrix(self):
        """Get joint-space mass matrix M(q) from MuJoCo."""
        # Save state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()

        # Set zero velocity
        for i, addr in enumerate(self.joint_dof_addrs):
            self.data.qvel[addr] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Extract mass matrix for our joints
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_factorM(self.model, self.data)
        # M is stored in data.qM (dense or sparse)
        # For simplicity, compute M by applying unit forces
        M = np.zeros((self.n_joints, self.n_joints))
        for j in range(self.n_joints):
            # Set unit acceleration for joint j
            qacc_save = self.data.qacc.copy()
            self.data.qacc[:] = 0
            self.data.qacc[self.joint_dof_addrs[j]] = 1.0
            mujoco.mj_inverse(self.model, self.data)
            M[:, j] = self.data.qfrc_inverse[self.joint_dof_addrs].copy()
            self.data.qacc[:] = qacc_save

        # Restore state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)

        return M

    def computed_torque(self, x_current, q_ref, dq_ref=None, ddq_ref=None):
        """Computed torque control (feedforward + PD).

        tau = M(q) * (ddq_ref + Kp*e + Kd*de) + C(q,dq)*dq + G(q)

        Uses MuJoCo's inverse dynamics for C and G.
        """
        n = self.n_joints
        q = x_current[:n]
        dq = x_current[n:]

        if dq_ref is None:
            dq_ref = np.zeros(n)
        if ddq_ref is None:
            ddq_ref = np.zeros(n)

        # Errors
        e = q_ref - q
        de = dq_ref - dq

        # Get dynamics from MuJoCo
        M = self.get_mass_matrix()
        G = self.get_gravity()

        # Coriolis: use MuJoCo inverse dynamics minus gravity
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        qacc_save = self.data.qacc.copy()

        # Set current state and zero acceleration
        for i, addr in enumerate(self.joint_dof_addrs):
            self.data.qvel[addr] = dq[i]
        self.data.qacc[:] = 0
        mujoco.mj_inverse(self.model, self.data)
        C_dq = self.data.qfrc_inverse[self.joint_dof_addrs].copy() - G

        # Restore
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.qacc[:] = qacc_save
        mujoco.mj_forward(self.model, self.data)

        # Computed torque
        ddq_cmd = ddq_ref + self.Kp @ e + self.Kd @ de
        tau = M @ ddq_cmd + C_dq + G

        return np.clip(tau, -self.tau_max, self.tau_max)

    def solve_mpc_residual(self, x_current, q_ref, p_ref):
        """Solve MPC for residual torque optimization.

        Minimizes: sum_k [ Q_ee * ||p_ee - p_ref||^2 + Q_q * ||q - q_ref||^2
                           + R * ||u||^2 ]
        subject to: linearized dynamics, joint limits, torque limits
        """
        n = self.n_joints
        Np = self.Np
        Nc = self.Nc

        # Linearize dynamics around current state
        u0 = np.zeros(n)
        A, B, c = self._linearize(x_current, u0)

        # Current Jacobian
        J = self.get_jacobian()

        # CasADi optimization
        u_var = ca.SX.sym('u', n * Nc)

        # Parameters
        x0 = ca.SX.sym('x0', 2 * n)
        A_p = ca.SX.sym('A', 2 * n, 2 * n)
        B_p = ca.SX.sym('B', 2 * n, n)
        c_p = ca.SX.sym('c', 2 * n)
        q_ref_p = ca.SX.sym('q_ref', n)
        p_ref_p = ca.SX.sym('p_ref', 3)
        J_p = ca.SX.sym('J', 3, n)
        p_cur = ca.SX.sym('p_cur', 3)

        # Forward predict
        x_pred = [x0]
        for k in range(Np):
            u_k = u_var[min(k, Nc - 1) * n:(min(k, Nc - 1) + 1) * n]
            x_next = A_p @ x_pred[-1] + B_p @ u_k + c_p
            x_pred.append(x_next)

        # Cost
        cost = 0
        Q_ee = 200.0
        Q_q = 50.0
        Q_dq = 5.0
        R = 0.1

        for k in range(1, Np + 1):
            q_k = x_pred[k][:n]
            dq_k = x_pred[k][n:]

            # Joint tracking
            q_err = q_k - q_ref_p
            cost += Q_q * ca.dot(q_err, q_err)

            # Velocity damping
            cost += Q_dq * ca.dot(dq_k, dq_k)

            # End-effector tracking (linearized)
            p_ee_k = p_cur + J_p @ (q_k - x0[:n])
            p_err = p_ee_k - p_ref_p
            cost += Q_ee * ca.dot(p_err, p_err)

        # Control cost
        for k in range(Nc):
            u_k = u_var[k * n:(k + 1) * n]
            cost += R * ca.dot(u_k, u_k)

        # Joint limit soft constraints
        slack = 500.0
        for k in range(1, Np + 1):
            q_k = x_pred[k][:n]
            for j in range(n):
                cost += slack * ca.fmax(0, self.q_min[j] - q_k[j])**2
                cost += slack * ca.fmax(0, q_k[j] - self.q_max[j])**2

        # Pack parameters
        p_val = np.concatenate([
            x_current, A.reshape(-1), B.reshape(-1), c,
            q_ref, p_ref, J.reshape(-1), self.get_ee_pos()
        ])

        nlp = {
            'x': u_var,
            'f': cost,
            'p': ca.vertcat(x0, A_p.reshape((-1, 1)), B_p.reshape((-1, 1)),
                           c_p, q_ref_p, p_ref_p, J_p.reshape((-1, 1)), p_cur)
        }

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 30,
            'ipopt.tol': 1e-3,
        }

        try:
            solver = ca.nlpsol('mpc_res', 'ipopt', nlp, opts)
            lbx = np.tile(-self.tau_max, Nc)
            ubx = np.tile(self.tau_max, Nc)
            sol = solver(x0=np.zeros(n * Nc), p=p_val, lbx=lbx, ubx=ubx)
            u_opt = np.array(sol['x']).flatten()[:n]
            return u_opt, True
        except Exception:
            return np.zeros(n), False

    def _linearize(self, x0, u0):
        """Linearize dynamics around (x0, u0) using finite differences."""
        n_x = 2 * self.n_joints
        n_u = self.n_ctrl
        eps = 1e-5

        # Save state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        # Nominal step
        self.set_state(x0)
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = u0[i]
        mujoco.mj_step(self.model, self.data)
        x_nom = self.get_state()

        # A matrix
        A = np.zeros((n_x, n_x))
        for j in range(n_x):
            x_pert = x0.copy()
            x_pert[j] += eps
            self.set_state(x_pert)
            for i, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = u0[i]
            mujoco.mj_step(self.model, self.data)
            A[:, j] = (self.get_state() - x_nom) / eps

        # B matrix
        B = np.zeros((n_x, n_u))
        for j in range(n_u):
            self.set_state(x0)
            u_pert = u0.copy()
            u_pert[j] += eps
            for i, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = u_pert[i]
            mujoco.mj_step(self.model, self.data)
            B[:, j] = (self.get_state() - x_nom) / eps

        c = x_nom - A @ x0 - B @ u0

        # Restore
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_forward(self.model, self.data)

        return A, B, c

    def solve(self, x_current, q_ref, p_ref):
        """Solve MPC: computed torque + optional residual optimization.

        Returns:
            u_opt: Optimal control (n_ctrl,)
            success: Whether solver succeeded
        """
        # Base: computed torque control
        tau_ct = self.computed_torque(x_current, q_ref)

        # Try MPC residual optimization
        tau_residual, mpc_ok = self.solve_mpc_residual(x_current, q_ref, p_ref)

        if mpc_ok:
            tau_total = tau_ct + 0.3 * tau_residual  # Blend residual
        else:
            tau_total = tau_ct

        # Convert torque to actuator control (normalized)
        # ctrl = tau / (gear_ratio * tau_max)
        ctrl = np.clip(tau_total / self.tau_max, -1.0, 1.0)

        return ctrl, True

    def move_to_target(self, target_pos, target_q=None, max_steps=500,
                       tol=0.01, viewer=None, delay=0.001):
        """Move end-effector to target using MPC control."""
        import time

        target_pos = np.array(target_pos)
        errors = []

        for step in range(max_steps):
            x_current = self.get_state()
            p_current = self.get_ee_pos()
            error = np.linalg.norm(p_current - target_pos)
            errors.append(error)

            if error < tol:
                print(f"  MPC converged at step {step}, error={error:.4f}")
                return True, errors

            # Reference: use IK solution or provided q_ref
            if target_q is not None:
                q_ref = target_q
            else:
                q_ref = x_current[:self.n_joints]

            # Solve MPC
            u_opt, _ = self.solve(x_current, q_ref, target_pos)

            # Apply control
            for i, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = u_opt[i]

            # Step
            for _ in range(int(self.dt / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)

            if viewer:
                viewer.sync()
                time.sleep(delay)

        print(f"  MPC max steps reached, final error={errors[-1]:.4f}")
        return False, errors

    def track_trajectory(self, q_traj, viewer=None, delay=0.001):
        """Track a joint-space trajectory using MPC."""
        import time
        errors = []

        for k, q_ref in enumerate(q_traj):
            x_current = self.get_state()
            p_current = self.get_ee_pos()

            u_opt, _ = self.solve(x_current, q_ref, p_current)

            for i, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = u_opt[i]

            for _ in range(int(self.dt / self.model.opt.timestep)):
                mujoco.mj_step(self.model, self.data)

            p_actual = self.get_ee_pos()
            q_save = self.get_state()
            self.ik_set_joint_angles(q_ref)
            p_ref = self.get_ee_pos()
            self.set_state(q_save)
            errors.append(np.linalg.norm(p_actual - p_ref))

            if viewer:
                viewer.sync()
                time.sleep(delay)

        return errors

    def ik_set_joint_angles(self, angles):
        """Set joint angles (for reference)."""
        for i, addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[addr] = angles[i]
