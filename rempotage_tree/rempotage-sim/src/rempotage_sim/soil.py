"""Potting soil: filling, the planting hole, pushing the root ball in, and firming.

The soil is not simulated as particles. Each pot carries:

- **visual layers** that appear as the filler pours (bottom up), and are swapped for holed versions
  once the dibber has punched the planting hole;
- **the planting hole.** The root ball does not collide with the soil. While its bottom is below the
  soil surface and it is over the hole, forces stand in for the soil around it:
  - along the hole axis, dry friction that grows with depth (the hole is 2 mm narrower than the root
    ball), as an elastic-plastic element: a stiff spring to an anchor point, which slips once the spring
    force reaches the friction limit. A root ball left in the hole stays there; pushing harder than the
    limit drives it deeper;
  - across the axis, the hole wall: a spring towards the axis and a torque keeping the root ball
    upright, growing with depth.
  The opposite forces go to the pot. The floor of the hole is a real collision geom (the root ball
  sits on it at full depth);
- **the top soil ring** in sectors that the robot's fingertips press on. A sector settles (moves down)
  while it is pressed harder than the soil's yield force, up to TAMP_MAX. That is the firming.

Once firmed, a weld (``planted_<k>``) holds the root ball in the pot.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rempotage_sim import constraints, meshgen
from rempotage_sim import layout as L
from rempotage_sim.scene import HIDDEN_GROUP, SHOWN_GROUP, set_collision

N_LAYERS = len(meshgen.SOIL_LAYERS) - 1


class Soil:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, robot_prefixes=("a/", "b/")):
        self.m, self.d = model, data
        n = L.N_POTS
        self.pot = [model.body(f"pot_{k}").id for k in range(n)]
        self.plant = [model.body(f"plant_{k}").id for k in range(L.N_PLANTS)]
        g = model.geom
        self.layers = [[g(f"pot_{k}/soil_{j}").id for j in range(N_LAYERS)] for k in range(n)]
        self.holed = [[g(f"pot_{k}/soil_{j}_holed").id for j in range(1, N_LAYERS)] for k in range(n)]
        self.plug = [g(f"pot_{k}/plug").id for k in range(n)]
        self.rings = [[g(f"pot_{k}/ring_{j}_col").id for j in range(L.TAMP_SECTORS)] for k in range(n)]
        self.ring_vis = [[g(f"pot_{k}/ring_{j}").id for j in range(L.TAMP_SECTORS)] for k in range(n)]
        self.ring_z0 = np.array([[model.geom_pos[r][2] for r in rr] for rr in self.rings])
        self.ring_vis_z0 = np.array([[model.geom_pos[r][2] for r in rr] for rr in self.ring_vis])
        self.floor = [g(f"pot_{k}/hole_floor").id for k in range(n)]
        self.ring_of = {r: (k, j) for k, rr in enumerate(self.rings) for j, r in enumerate(rr)}
        self.robot_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith(robot_prefixes)}
        self.fill = np.zeros(n)
        self.dibbled = [False] * n
        self.comp = np.zeros((n, L.TAMP_SECTORS))
        self.anchor: dict[int, float] = {}       # pot -> anchor of the axial soil spring (pot frame z)
        self.in_hole: dict[int, int] = {}        # pot -> plant whose root ball is in its hole
        self.planted: dict[int, int] = {}        # pot -> plant held by the firmed soil (weld)
        self.max_force: dict[int, float] = {}    # pot -> largest soil resistance met while pushing in
        self._mass0 = float(model.body_mass[self.pot[0]])
        self._inertia0 = model.body_inertia[self.pot[0]].copy()
        self._ipos0 = model.body_ipos[self.pot[0]].copy()
        self._f6 = np.zeros(6)
        self._scratch = mujoco.MjData(model)
        self.press = np.zeros(n)                 # robot fingertips pressing on each pot's top soil (N)

    # --- Filling and dibbling (driven by the line controller) ---------------------------------
    def _show(self, geom: int, on: bool) -> None:
        self.m.geom_group[geom] = SHOWN_GROUP if on else HIDDEN_GROUP

    def set_fill(self, k: int, f: float) -> None:
        """Pot k filled to fraction f (0..1) of its soil."""
        self.fill[k] = f = float(np.clip(f, 0.0, 1.0))
        z = meshgen.SOIL_LAYERS[0] + f * (L.SOIL_Z - meshgen.SOIL_LAYERS[0])
        for j, geom in enumerate(self.layers[k]):
            self._show(geom, z >= meshgen.SOIL_LAYERS[j + 1] - 1e-6)
        full = f >= 1.0
        self._show(self.plug[k], full)
        for r in self.ring_vis[k]:
            self._show(r, full)
        for r in self.rings[k]:
            set_collision(self.m, r, 2 if full else 0, 0)
        if full:
            # The soil's mass, added once the pot is full (MuJoCo's derived constants follow).
            b = self.pot[k]
            ms = L.soil_mass()
            self.m.body_ipos[b] = (self._mass0 * self._ipos0 + ms * np.array([0, 0, 0.5 * L.SOIL_Z])) / (self._mass0 + ms)
            self.m.body_mass[b] = self._mass0 + ms
            self.m.body_inertia[b] = self._inertia0 + ms * np.array([0.0045, 0.0045, 0.0035])
            eq_data = self.m.eq_data.copy()            # mj_setConst recomputes the welds' relative poses
            mujoco.mj_setConst(self.m, self._scratch)   # its own MjData: it runs kinematics at qpos0
            self.m.eq_data[:] = eq_data

    def dibble(self, k: int) -> None:
        """The dibber has punched the planting hole."""
        self.dibbled[k] = True
        for j, geom in enumerate(self.layers[k]):
            if j > 0:
                self._show(geom, False)
        for geom in self.holed[k]:
            self._show(geom, True)
        self._show(self.plug[k], False)
        set_collision(self.m, self.floor[k], 4, 0)

    # --- Root ball in the hole ------------------------------------------------------------------
    def rb_in_pot(self, k: int, p: int) -> tuple[np.ndarray, np.ndarray]:
        """Root ball p's bottom centre and axis, in pot k's frame."""
        d = self.d
        Rp = d.xmat[self.pot[k]].reshape(3, 3)
        pos = Rp.T @ (d.xpos[self.plant[p]] - d.xpos[self.pot[k]])
        axis = Rp.T @ d.xmat[self.plant[p]].reshape(3, 3)[:, 2]
        return pos, axis

    def depth(self, k: int, p: int) -> float:
        """How far root ball p's bottom is below pot k's soil surface (negative: above it)."""
        return float(L.SOIL_Z - self.rb_in_pot(k, p)[0][2])

    def _candidate(self, k: int) -> int | None:
        if k in self.in_hole:
            return self.in_hole[k]
        for p in range(len(self.plant)):
            if p in self.planted.values():
                continue
            pos, _ = self.rb_in_pot(k, p)
            if np.hypot(pos[0], pos[1]) < 0.015 and -0.002 < L.SOIL_Z - pos[2] < L.HOLE_DEPTH + 0.01:
                return p
        return None

    def apply(self) -> None:
        """Soil forces on the root balls being pushed in (call before every physics step)."""
        m, d = self.m, self.d
        for k in range(len(self.pot)):
            if not self.dibbled[k] or k in self.planted:
                continue
            p = self._candidate(k)
            if p is None:
                continue
            bp, bk = self.plant[p], self.pot[k]
            pos, axis = self.rb_in_pot(k, p)
            depth = L.SOIL_Z - pos[2]
            if depth <= 0.0:
                self.in_hole.pop(k, None)
                self.anchor.pop(k, None)
                d.xfrc_applied[bp] = 0.0
                continue
            self.in_hole[k] = p
            Rp = d.xmat[bk].reshape(3, 3)
            vp, vk = np.zeros(6), np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bp, vp, 0)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bk, vk, 0)
            r = d.xpos[bp] - d.xpos[bk]
            v_rel = Rp.T @ (vp[3:] - vk[3:] - np.cross(vk[:3], r))
            w_rel = vp[:3] - vk[:3]
            frac = min(depth / L.HOLE_DEPTH, 1.0)
            # Axial: elastic-plastic friction.
            a = self.anchor.setdefault(k, pos[2])
            f_lim = L.SOIL_F0 + L.SOIL_F_PER_M * depth
            s = pos[2] - a
            if L.SOIL_K * abs(s) > f_lim:
                a = pos[2] - np.sign(s) * f_lim / L.SOIL_K
                self.anchor[k] = a
                s = pos[2] - a
            fz = -L.SOIL_K * s - L.SOIL_C * v_rel[2]
            self.max_force[k] = max(self.max_force.get(k, 0.0), abs(L.SOIL_K * s))
            # Lateral: the hole wall.
            fxy = -L.SOIL_K_LAT * frac * pos[:2] - 30.0 * frac * v_rel[:2]
            n = np.linalg.norm(fxy)
            if n > 20.0:
                fxy *= 20.0 / n
            f_world = Rp @ np.array([fxy[0], fxy[1], fz])
            # Tilt only: damping the spin about the axis too would be unstable (small spin inertia).
            w_tilt = w_rel - (w_rel @ Rp[:, 2]) * Rp[:, 2]
            tau = L.SOIL_K_ROT * frac * np.cross(Rp @ axis, Rp[:, 2]) - 0.15 * frac * w_tilt
            d.xfrc_applied[bp, :3] = f_world
            d.xfrc_applied[bp, 3:] = tau
            arm = d.xipos[bp] - d.xipos[bk]
            d.xfrc_applied[bk, :3] = -f_world
            d.xfrc_applied[bk, 3:] = -tau - np.cross(arm, f_world)

    def plant_in(self, k: int, p: int) -> None:
        """Firming done: the soil holds root ball p in pot k (weld at the current pose)."""
        constraints.activate_weld(self.m, self.d, f"planted_{k}", body2=self.plant[p])
        self.planted[k] = p
        self.in_hole.pop(k, None)
        self.anchor.pop(k, None)
        self.d.xfrc_applied[self.plant[p]] = 0.0
        self.d.xfrc_applied[self.pot[k]] = 0.0

    # --- Firming ---------------------------------------------------------------------------------
    def tamp(self, dt: float) -> np.ndarray:
        """Settle the top soil sectors pressed by the robot (call every control tick). Returns the
        pressing force per pot."""
        m, d = self.m, self.d
        force = np.zeros((len(self.pot), L.TAMP_SECTORS))
        for i, c in enumerate(d.contact[: d.ncon]):
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in self.ring_of and m.geom_bodyid[other] in self.robot_bodies:
                    mujoco.mj_contactForce(m, d, i, self._f6)
                    k, j = self.ring_of[g]
                    force[k, j] += self._f6[0]
        pressed = force > L.TAMP_YIELD
        if pressed.any():
            self.comp = np.minimum(self.comp + pressed * L.TAMP_RATE * (force - L.TAMP_YIELD) * dt, L.TAMP_MAX)
            for k, j in zip(*np.nonzero(pressed), strict=True):
                m.geom_pos[self.rings[k][j]][2] = self.ring_z0[k, j] - self.comp[k, j]
                m.geom_pos[self.ring_vis[k][j]][2] = self.ring_vis_z0[k, j] - self.comp[k, j]
        self.press = force.sum(axis=1)
        return self.press
