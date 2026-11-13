"""YAM arm control: damped-least-squares IK + Cartesian min-jerk trajectories.

Same controller as the Accuwat cell (velocity, inertia and friction feed-forward on the YAM's
MIT-mode motors), with the linear_4310 gripper of the tree-staking cell, and a force
feed-forward: ``push`` is the force the tool should exert on what it touches (tau = J^T f),
used by arm B to press the root ball and the soil down.

Motion primitives are generators that advance one control tick per ``yield``; the task script
composes them (see :mod:`rempotage_sim.task`).
"""

from __future__ import annotations

from collections.abc import Generator

import mujoco
import numpy as np

from rempotage_sim import layout as L

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
Tick = Generator[None]
UP = np.array([0.0, 0.0, 1.0])


def grasp_frame(approach, closing) -> np.ndarray:
    """Rotation of ``grasp_site``: z = approach direction, y = finger closing axis."""
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    y = np.asarray(closing, float)
    y = y - y.dot(z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def horizontal(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


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
                 home=L.ARM_HOME_Q):
        self.m, self.d, self.prefix, self.dt = model, data, prefix, control_dt
        jids = [model.joint(prefix + j).id for j in JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0] + 1e-3
        self.hi = model.jnt_range[jids, 1] - 1e-3
        self.act = np.array([model.actuator(prefix + j).id for j in JOINTS])
        self.grip_act = model.actuator(prefix + "gripper").id
        self.finger_qadr = model.jnt_qposadr[model.joint(prefix + "left_finger").id]
        self.site = model.site(prefix + "grasp_site").id
        self.home = np.array(home, float)
        finger_bodies = {model.body(prefix + n).id for n in ("left_finger", "right_finger", "left_pad", "right_pad")}
        left = {model.body(prefix + "left_finger").id, model.body(prefix + "left_pad").id}
        self.finger_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in finger_bodies}
        self.left_geoms = {g for g in self.finger_geoms if model.geom_bodyid[g] in left}
        self._d_ik = mujoco.MjData(model)
        self.q_cmd = data.qpos[self.qadr].copy()
        self.grip_cmd = L.GRIPPER_OPEN
        self._q_prev = self.q_cmd.copy()
        self._integral = np.zeros(6)
        # Velocity feed-forward (MIT mode: tau = kp (q_ref - q) + kd (qd_ref - qd)); the position
        # actuators apply kp (ctrl - q) - kv qd, so adding kv / kp * qd_ref gives the same law.
        self.ff = np.array([-model.actuator_biasprm[a, 2] / model.actuator_gainprm[a, 0] for a in self.act])
        self.kp = np.array([model.actuator_gainprm[a, 0] for a in self.act])
        # Friction feed-forward (tau_ff), in the direction of the commanded motion.
        self.fc = model.dof_frictionloss[self.dadr] / self.kp
        self._qd_prev = np.zeros(6)
        self._qdd = np.zeros(6)
        self._vec = np.zeros(model.nv)
        self._res = np.zeros(model.nv)
        self._jacp = np.zeros((3, model.nv))
        self.push = np.zeros(3)  # N, force the tool exerts on the environment (world frame)
        self.ki = 1.5            # 1/s, removes steady-state sag
        self.i_max = 0.15        # rad

    # --- State -------------------------------------------------------------------
    def set_home(self) -> None:
        self.d.qpos[self.qadr] = self.home
        self.q_cmd = self.home.copy()
        self._q_prev = self.q_cmd.copy()
        self.d.ctrl[self.act] = self.home
        self.d.ctrl[self.grip_act] = self.grip_cmd

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        d = self._d_ik
        d.qpos[:] = self.d.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def cmd_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.fk(self.q_cmd)

    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def finger_q(self) -> float:
        return float(self.d.qpos[self.finger_qadr])

    def contacts(self, body_ids) -> tuple[int, float]:
        """Fingers (0, 1 or 2) touching any of the given bodies, and the total normal force."""
        touching, force = set(), 0.0
        f6 = np.zeros(6)
        for i, c in enumerate(self.d.contact[: self.d.ncon]):
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in self.finger_geoms and self.m.geom_bodyid[other] in body_ids:
                    touching.add("left" if g in self.left_geoms else "right")
                    mujoco.mj_contactForce(self.m, self.d, i, f6)
                    force += f6[0]
        return len(touching), force

    # --- IK --------------------------------------------------------------------------
    def ik(self, pos, R, seed=None, iters: int = 40, tol: float = 1e-4) -> tuple[np.ndarray, float]:
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
        qd_ref = (self.q_cmd - self._q_prev) / self.dt
        self._qdd = 0.3 * self._qdd + 0.7 * (qd_ref - self._qd_prev) / self.dt   # light low-pass
        self._qd_prev = qd_ref
        self._vec[:] = 0.0
        self._vec[self.dadr] = self._qdd
        mujoco.mj_mulM(self.m, self.d, self._res, self._vec)
        tau_ff = self._res[self.dadr].copy()
        if self.push.any():
            mujoco.mj_jacSite(self.m, self.d, self._jacp, None, self.site)
            tau_ff += self._jacp[:, self.dadr].T @ self.push
        moving = np.abs(self.q_cmd - self._q_prev) > 2e-4
        self._q_prev = self.q_cmd.copy()
        # Integral action only while the command holds still (anti-windup); held while moving.
        self._integral = np.where(moving, 0.998 * self._integral,
                                  self._integral + self.ki * (self.q_cmd - q_meas) * self.dt)
        self._integral = np.clip(self._integral, -self.i_max, self.i_max)
        self.d.ctrl[self.act] = (self.q_cmd + self._integral + self.ff * qd_ref + self.fc * np.tanh(qd_ref / 0.01)
                                 + tau_ff / self.kp)
        self.d.ctrl[self.grip_act] = self.grip_cmd

    def reset_integral(self) -> None:
        self._integral[:] = 0.0

    # --- Motion primitives (generators) ------------------------------------------------
    def _warn(self, err: float, pos) -> None:
        if err > 5e-3:
            print(f"[{self.prefix}] warning: IK residual {err * 1000:.1f} mm at {np.round(pos, 3)}")

    def move_to(self, pos, R, duration: float) -> Tick:
        """Straight-line Cartesian move of grasp_site with min-jerk timing."""
        p0, R0 = self.cmd_pose()
        pos = np.asarray(pos, float)
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            self.q_cmd, _ = self.ik(p0 + s * (pos - p0), slerp_R(R0, R, s), iters=8)
            yield
        self.q_cmd, err = self.ik(pos, R, iters=60)
        self._warn(err, pos)

    def swing_to(self, pos, R, duration: float, lift: float = 0.0, base_xy=None) -> Tick:
        """Cartesian move along an arc around the robot base (radius, bearing and height
        interpolated, min-jerk timing), raised by `lift` in the middle."""
        c = np.zeros(2) if base_xy is None else np.asarray(base_xy, float)
        p0, R0 = self.cmd_pose()
        pos = np.asarray(pos, float)
        v0, v1 = p0[:2] - c, pos[:2] - c
        r0, a0 = np.hypot(*v0), np.arctan2(v0[1], v0[0])
        r1, a1 = np.hypot(*v1), np.arctan2(v1[1], v1[0])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        # The hand turns with the swing about the base; only what is left over is interpolated, so a
        # large swing never turns the hand the short way round, against the arm.
        R_rest = rot_z(-da) @ R
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            r, a = r0 + s * (r1 - r0), a0 + s * da
            z = p0[2] + s * (pos[2] - p0[2]) + lift * np.sin(np.pi * s)
            self.q_cmd, _ = self.ik(np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a), z]),
                                    rot_z(s * da) @ slerp_R(R0, R_rest, s), iters=8)
            yield
        self.q_cmd, err = self.ik(pos, R, iters=60)
        self._warn(err, pos)

    def move_joints(self, q, duration: float) -> Tick:
        q0 = self.q_cmd.copy()
        q = np.asarray(q, float)
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            self.q_cmd = q0 + min_jerk(k / n) * (q - q0)
            yield

    def go_home(self, duration: float = 2.0) -> Tick:
        yield from self.move_joints(self.home, duration)

    def gripper(self, q: float, duration: float = 0.5) -> Tick:
        g0 = self.grip_cmd
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            self.grip_cmd = g0 + min_jerk(k / n) * (q - g0)
            yield

    def settle(self, tol: float = 0.002, timeout: float = 1.5) -> Tick:
        """Wait until the real grasp site has caught up with the commanded pose."""
        for _ in range(max(1, round(timeout / self.dt))):
            if np.linalg.norm(self.site_pose()[0] - self.cmd_pose()[0]) < tol:
                return
            yield

    def wait(self, duration: float) -> Tick:
        for _ in range(max(1, round(duration / self.dt))):
            yield


def parallel(*gens: Tick) -> Tick:
    """Run several primitives side by side until all of them finish."""
    live = list(gens)
    while live:
        nxt = []
        for g in live:
            try:
                next(g)
                nxt.append(g)
            except StopIteration:
                pass
        live = nxt
        if live:
            yield
