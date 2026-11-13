"""Switching the task's equality constraints on and off at runtime without jumps."""

from __future__ import annotations

import mujoco
import numpy as np


def _pose(d: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    return d.xpos[body].copy(), d.xquat[body].copy()


def retarget(m: mujoco.MjModel, eq: int, body2: int | None) -> None:
    """Point an equality's second body at another body (one constraint serves any stake / pot)."""
    if body2 is not None:
        m.eq_obj2id[eq] = body2


def activate_weld(m: mujoco.MjModel, d: mujoco.MjData, name: str, body2: int | None = None) -> None:
    """Weld body2 to body1 at their *current* relative pose."""
    eq = m.equality(name).id
    retarget(m, eq, body2)
    b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
    p1, q1 = _pose(d, b1)
    p2, q2 = _pose(d, b2)
    R1 = d.xmat[b1].reshape(3, 3)
    q1_inv = np.zeros(4)
    mujoco.mju_negQuat(q1_inv, q1)
    rel_q = np.zeros(4)
    mujoco.mju_mulQuat(rel_q, q1_inv, q2)
    # data: anchor in body2 (3) | relpose pos in body1 (3) | relpose quat (4) | torquescale
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = R1.T @ (p2 - p1)
    m.eq_data[eq, 6:10] = rel_q
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def activate_connect(m: mujoco.MjModel, d: mujoco.MjData, name: str, world_point, body2: int | None = None) -> None:
    """Pin body1 and body2 together at a world point (ball joint, no jump)."""
    eq = m.equality(name).id
    retarget(m, eq, body2)
    b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
    w = np.asarray(world_point, float)
    m.eq_data[eq, 0:3] = d.xmat[b1].reshape(3, 3).T @ (w - d.xpos[b1])
    m.eq_data[eq, 3:6] = d.xmat[b2].reshape(3, 3).T @ (w - d.xpos[b2])
    d.eq_active[eq] = 1


def deactivate(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> None:
    d.eq_active[m.equality(name).id] = 0


def is_active(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> bool:
    return bool(d.eq_active[m.equality(name).id])
