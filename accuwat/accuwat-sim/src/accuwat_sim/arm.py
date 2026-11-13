"""YAM arm control: damped-least-squares IK + Cartesian min-jerk trajectories.

Same controller as the Claret cell, for a YAM whose gripper is replaced by an electromagnet:
6 joints, no finger. Motion primitives are generators that advance one control tick per
``yield``; the task script composes them (see :mod:`accuwat_sim.task`).
"""

from __future__ import annotations

from collections.abc import Generator

import mujoco
import numpy as np

from accuwat_sim import layout as L

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
Tick = Generator[None]
UP = np.array([0.0, 0.0, 1.0])


def tool_frame(approach, side) -> np.ndarray:
    """Rotation of the tool site: z = approach direction, x towards `side`."""
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    x = np.asarray(side, float)
    x = x - x.dot(z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def topdown(xy) -> np.ndarray:
    """Magnet pointing straight down, the wrist turned to its natural yaw over the point xy
    (the magnet is round, so the yaw only matters for the arm)."""
    xy = np.asarray(xy, float)
    inward = -xy[:2] / np.linalg.norm(xy[:2])
    return tool_frame(-UP, [inward[0], inward[1], 0.0])


class _NoSleep:
    """Kinematics on the IK's private MjData must not skip "sleeping" bodies (its sleep state is
    never computed): switch sleeping off for the duration of the call."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model

    def __enter__(self):
        self.flags = self.m.opt.enableflags
        self.m.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)

    def __exit__(self, *exc):
        self.m.opt.enableflags = self.flags


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def slerp_R(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    w = rot_log(R1 @ R0.T) * s
    angle = np.linalg.norm(w)
    if angle < 1e-9:
        return R0.copy()
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, w / angle, angle)
    dR = np.zeros(9)
    mujoco.mju_quat2Mat(dR, q)
    return dR.reshape(3, 3) @ R0


class Arm:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str, control_dt: float,
                 site: str = "magnet_site"):
        self.m, self.d, self.prefix, self.dt = model, data, prefix, control_dt
        jids = [model.joint(prefix + j).id for j in JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0] + 1e-3
        self.hi = model.jnt_range[jids, 1] - 1e-3
        self.act = np.array([model.actuator(prefix + j).id for j in JOINTS])
        self.site = model.site(prefix + site).id
        self._d_ik = mujoco.MjData(model)
        self._nosleep = _NoSleep(model)
        self.q_cmd = data.qpos[self.qadr].copy()
        self._integral = np.zeros(6)
        self._q_prev = self.q_cmd.copy()
        # Velocity feed-forward (the YAM's motors run in MIT mode: tau = kp (q_ref - q) + kd (qd_ref - qd)).
        # The position actuators here apply kp (ctrl - q) - kv qd, so adding kv / kp * qd_ref to the
        # command gives the same law. Without it each joint lags a moving command by kv / kp seconds
        # (62 ms for the big joints, 38 ms for joint 4, 150 ms for the wrist), which bends the hand's path.
        self.ff = np.array([-model.actuator_biasprm[a, 2] / model.actuator_gainprm[a, 0] for a in self.act])
        # Friction feed-forward (the tau_ff term of MIT mode): each joint has 0.1 Nm of dry friction,
        # which the soft servos only overcome after an error of friction / kp (up to 10 mrad on the
        # wrist, 1-3 mm at the tool). Push through it in the direction of the commanded motion.
        kp = np.array([model.actuator_gainprm[a, 0] for a in self.act])
        self.fc = model.dof_frictionloss[self.dadr] / kp
        # Inertia feed-forward (tau_ff = M(q) qdd_ref): without it the shoulder lags by I qdd / kp
        # when the hand accelerates, several millimetres at the tool even for slow vertical moves.
        self.kp = kp
        self._qd_prev = np.zeros(6)
        self._qdd = np.zeros(6)
        self._vec = np.zeros(model.nv)
        self._res = np.zeros(model.nv)
        self.ki = 1.5            # 1/s, removes steady-state sag
        self.i_max = 0.15        # rad

    # --- State -------------------------------------------------------------------
    def set_home(self, q=L.ARM_HOME_Q) -> None:
        self.d.qpos[self.qadr] = q
        self.q_cmd = np.array(q, float)
        self._q_prev = self.q_cmd.copy()
        self.d.ctrl[self.act] = q

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        d = self._d_ik
        d.qpos[:] = self.d.qpos
        d.qpos[self.qadr] = q
        with self._nosleep:
            mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def cmd_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.fk(self.q_cmd)

    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    # --- IK --------------------------------------------------------------------------
    def ik(self, pos, R, seed=None, iters: int = 40, tol: float = 1e-4) -> tuple[np.ndarray, float]:
        with self._nosleep:
            return self._ik(pos, R, seed, iters, tol)

    def _ik(self, pos, R, seed, iters: int, tol: float) -> tuple[np.ndarray, float]:
        m, d = self.m, self._d_ik
        q = (self.q_cmd if seed is None else np.asarray(seed, float)).copy()
        d.qpos[:] = self.d.qpos
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_p = pos - d.site_xpos[self.site]
            e_r = rot_log(R @ d.site_xmat[self.site].reshape(3, 3).T)
            e = np.concatenate([e_p, 0.5 * e_r])
            err = float(np.linalg.norm(e_p) + 0.05 * np.linalg.norm(e_r))
            if err < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], 0.5 * jacr[:, self.dadr]])
            lam = 1e-4 + 0.05 * min(1.0, float(np.linalg.norm(e)))
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return q, err

    # --- Low level control (called every control tick) -------------------------------------
    def apply(self) -> None:
        q_meas = self.d.qpos[self.qadr]
        # Integral action only while the command holds still (anti-windup).
        qd_ref = (self.q_cmd - self._q_prev) / self.dt
        self._qdd = 0.3 * self._qdd + 0.7 * (qd_ref - self._qd_prev) / self.dt   # light low-pass
        self._qd_prev = qd_ref
        self._vec[self.dadr] = self._qdd
        mujoco.mj_mulM(self.m, self.d, self._res, self._vec)
        tau_ff = self._res[self.dadr]
        moving = np.abs(self.q_cmd - self._q_prev) > 2e-4
        self._q_prev = self.q_cmd.copy()
        # Moving: hold the integral (a slow leak only). Draining it fast shifts the hand by about a
        # millimetre each time a move starts.
        self._integral = np.where(moving, 0.998 * self._integral, self._integral + self.ki * (self.q_cmd - q_meas) * self.dt)
        self._integral = np.clip(self._integral, -self.i_max, self.i_max)
        self.d.ctrl[self.act] = (self.q_cmd + self._integral + self.ff * qd_ref + self.fc * np.tanh(qd_ref / 0.01)
                                 + tau_ff / self.kp)

    # --- Motion primitives (generators) ------------------------------------------------
    def move_to(self, pos, R, duration: float) -> Tick:
        """Straight-line Cartesian move of the tool site with min-jerk timing."""
        p0, R0 = self.cmd_pose()
        pos = np.asarray(pos, float)
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            self.q_cmd, _ = self.ik(p0 + s * (pos - p0), slerp_R(R0, R, s), iters=8)
            yield
        self.q_cmd, err = self.ik(pos, R, iters=60)
        if err > 5e-3:
            print(f"[{self.prefix}] warning: IK residual {err * 1000:.1f} mm at {np.round(pos, 3)}")

    def swing_to(self, pos, R, duration: float, lift: float = 0.0) -> Tick:
        """Cartesian move along an arc around the robot base (radius, bearing and height
        interpolated, min-jerk timing), raised by `lift` in the middle. Used for the
        transfers between the boxes and the rack."""
        p0, R0 = self.cmd_pose()
        pos = np.asarray(pos, float)
        r0, a0 = np.hypot(p0[0], p0[1]), np.arctan2(p0[1], p0[0])
        r1, a1 = np.hypot(pos[0], pos[1]), np.arctan2(pos[1], pos[0])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            r, a = r0 + s * (r1 - r0), a0 + s * da
            z = p0[2] + s * (pos[2] - p0[2]) + lift * np.sin(np.pi * s)
            self.q_cmd, _ = self.ik(np.array([r * np.cos(a), r * np.sin(a), z]), slerp_R(R0, R, s), iters=8)
            yield
        self.q_cmd, err = self.ik(pos, R, iters=60)
        if err > 5e-3:
            print(f"[{self.prefix}] warning: IK residual {err * 1000:.1f} mm at {np.round(pos, 3)}")

    def move_joints(self, q, duration: float) -> Tick:
        q0 = self.q_cmd.copy()
        q = np.asarray(q, float)
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            self.q_cmd = q0 + min_jerk(k / n) * (q - q0)
            yield

    def settle(self, tol: float = 0.002, timeout: float = 1.5) -> Tick:
        """Wait until the real tool site has caught up with the commanded pose."""
        for _ in range(max(1, round(timeout / self.dt))):
            if np.linalg.norm(self.site_pose()[0] - self.cmd_pose()[0]) < tol:
                return
            yield

    def wait(self, duration: float) -> Tick:
        for _ in range(max(1, round(duration / self.dt))):
            yield
