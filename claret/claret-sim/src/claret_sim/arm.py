"""YAM arm control: damped-least-squares IK + Cartesian min-jerk trajectories.

Each :class:`Arm` owns the position targets of one YAM (6 joints + gripper). Motion
primitives are generators that advance one control tick per ``yield``; the task script
composes them (see :mod:`claret_sim.task`).
"""

from __future__ import annotations

from collections.abc import Generator

import mujoco
import numpy as np

from claret_sim import layout as L

JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
Tick = Generator[None]


def grasp_frame(approach, closing) -> np.ndarray:
    """Rotation of ``grasp_site``: z = approach direction, y = finger closing axis."""
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    y = np.asarray(closing, float)
    y = y - y.dot(z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


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
                 site: str = "grasp_site"):
        self.m, self.d, self.prefix, self.dt = model, data, prefix, control_dt
        jids = [model.joint(prefix + j).id for j in JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0] + 1e-3
        self.hi = model.jnt_range[jids, 1] - 1e-3
        self.act = np.array([model.actuator(prefix + j).id for j in JOINTS])
        self.grip_act = model.actuator(prefix + "gripper").id
        self.finger_qadr = model.jnt_qposadr[model.joint(prefix + "left_finger").id]
        self.site = model.site(prefix + site).id
        self.link6 = model.body(prefix + "link_6").id
        finger_bodies = {model.body(prefix + n).id for n in ("left_finger", "right_finger", "left_pad", "right_pad")}
        self.left_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in
                           {model.body(prefix + "left_finger").id, model.body(prefix + "left_pad").id}}
        self.finger_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in finger_bodies}
        self._d_ik = mujoco.MjData(model)
        self.q_cmd = data.qpos[self.qadr].copy()
        self.grip_cmd = L.GRIPPER_OPEN
        self._integral = np.zeros(6)
        self._q_prev = self.q_cmd.copy()
        self.ki = 1.5            # 1/s, removes steady-state sag under payloads
        self.i_max = 0.15        # rad

    # --- State -------------------------------------------------------------------
    def set_home(self, q=L.ARM_HOME_Q) -> None:
        self.d.qpos[self.qadr] = q
        self.q_cmd = np.array(q, float)
        self._q_prev = self.q_cmd.copy()
        self.d.ctrl[self.act] = q
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

    def pad_contacts(self, body_id: int) -> int:
        """Number of fingers (0, 1 or 2) touching the given body."""
        touching = set()
        for c in self.d.contact[: self.d.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in self.finger_geoms and self.m.geom_bodyid[other] == body_id:
                    touching.add("left" if g in self.left_geoms else "right")
        return len(touching)

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
        # Integral action only while the command holds still (anti-windup): during a fast move
        # the tracking error is lag, not sag, and integrating it makes the arm overshoot.
        moving = np.abs(self.q_cmd - self._q_prev) > 2e-4
        self._q_prev = self.q_cmd.copy()
        self._integral = np.where(moving, 0.98 * self._integral, self._integral + self.ki * (self.q_cmd - q_meas) * self.dt)
        self._integral = np.clip(self._integral, -self.i_max, self.i_max)
        self.d.ctrl[self.act] = self.q_cmd + self._integral
        self.d.ctrl[self.grip_act] = self.grip_cmd

    # --- Motion primitives (generators) ------------------------------------------------
    def move_to(self, pos, R, duration: float) -> Tick:
        """Straight-line Cartesian move of grasp_site with min-jerk timing."""
        p0, R0 = self.cmd_pose()
        pos = np.asarray(pos, float)
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            q, err = self.ik(p0 + s * (pos - p0), slerp_R(R0, R, s), iters=8)
            self.q_cmd = q
            yield
        self.q_cmd, err = self.ik(pos, R, iters=60)
        if err > 5e-3:
            print(f"[{self.prefix}] warning: IK residual {err * 1000:.1f} mm at {np.round(pos, 3)}")

    def swing_to(self, pos, R, duration: float, q_goal=None) -> Tick:
        """Cartesian move along an arc around the robot base (radius, bearing and height
        interpolated, min-jerk timing). Used for the long transfers: the hand never cuts
        close to the base, and the height changes smoothly from start to goal.

        With q_goal (a joint solution at the goal, e.g. from the planner), each step's IK starts
        from the joint-space blend towards q_goal, so the arm arrives in that configuration."""
        p0, R0 = self.cmd_pose()
        q0 = self.q_cmd.copy()
        pos = np.asarray(pos, float)
        r0, a0 = np.hypot(p0[0], p0[1]), np.arctan2(p0[1], p0[0])
        r1, a1 = np.hypot(pos[0], pos[1]), np.arctan2(pos[1], pos[0])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        n = max(1, round(duration / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            r, a = r0 + s * (r1 - r0), a0 + s * da
            p = np.array([r * np.cos(a), r * np.sin(a), p0[2] + s * (pos[2] - p0[2])])
            if q_goal is None:
                self.q_cmd, _ = self.ik(p, slerp_R(R0, R, s), iters=8)
            else:
                self.q_cmd, _ = self.ik(p, slerp_R(R0, R, s), seed=q0 + s * (q_goal - q0), iters=15)
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
