"""Collision-checked hand poses: IK + MuJoCo collision detection on a private copy of the model.

The parts are packed tight against the carton walls, so near the walls the hand cannot simply
point down: the YAM's gripper housing (10 x 6.5 cm) and its forearm would hit the cardboard.
For every slot the planner tries hand orientations (which side the vacuum cup faces) and
leans (the wrist tilted away from the walls), and keeps the first one for which the IK solves
and the arm stays clear (MARGIN) of the carton, the tray, and the parts already packed.

The check model only reports arm-vs-cell contacts: every arm geom (links, gripper, jaws, cup)
gets contype 8 and every other collidable geom conaffinity 8.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from claret_sim import layout as L
from claret_sim.arm import Arm, grasp_frame
from claret_sim.scene import ARM

UP = np.array([0.0, 0.0, 1.0])
MARGIN = 0.006
IK_TOL = 2e-3
N_ALTS = 4             # clear hand poses kept per slot (the tray pick picks one that suits the part)
PUSH_REACH = 0.018     # at the end of the push the grip site may be this far past the part axis (jaw play in the bore)


def rot(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def turn(u, angle: float) -> np.ndarray:
    """Rotate a horizontal 2D vector."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def topdown(u) -> np.ndarray:
    """Vertical grasp frame with the vacuum cup on the side u (horizontal 2D vector).
    The cup sits on the grasp-frame +x side; the fingers close along u turned by -90 deg."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    return grasp_frame(-UP, [u[1], -u[0], 0.0])


def leaned(R: np.ndarray, w, angle: float) -> np.ndarray:
    """Tilt a hand frame so that its wrist moves towards the horizontal direction w."""
    if abs(angle) < 1e-9:
        return R
    axis = np.cross(UP, [w[0], w[1], 0.0])
    return rot(axis, angle) @ R


@dataclass
class SlotPlan:
    R: np.ndarray                    # hand rotation at the slot (part upright)
    q: np.ndarray                    # joint solution at the packed position (a good IK seed)
    lean: float                      # deg, for the log
    ok: bool = True
    hits: list = field(default_factory=list)
    alts: list = field(default_factory=list)   # all the clear poses found (this one first)


class Planner:
    def __init__(self, model: mujoco.MjModel, arm: Arm):
        self.arm = arm
        self.m = copy.deepcopy(model)
        self.d = mujoco.MjData(self.m)
        m = self.m
        arm_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(ARM) or m.body(b).name == "cup"}
        self.arm_geoms = set()
        for g in range(m.ngeom):
            collidable = m.geom_contype[g] or m.geom_conaffinity[g]
            if m.geom_bodyid[g] in arm_bodies and collidable:
                m.geom_contype[g], m.geom_conaffinity[g] = 8, 0
                m.geom_margin[g] = MARGIN
                self.arm_geoms.add(g)
            elif collidable and m.geom_contype[g] != 4 and m.geom(g).name != "floor":
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 8
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
        self.finger_q = [m.jnt_qposadr[m.joint(ARM + j).id] for j in ("left_finger", "right_finger")]
        self.part_q = {}
        for b in range(m.nbody):
            if m.body(b).name.startswith("part_"):
                self.part_q[b] = m.jnt_qposadr[m.body_jntadr[b]]

    # --- Checks ----------------------------------------------------------------------------
    def collisions(self, q, world_qpos, finger: float, ignore=()) -> list[tuple[str, str, float]]:
        m, d = self.m, self.d
        d.qpos[:] = world_qpos
        d.qpos[self.arm.qadr] = q
        d.qpos[self.finger_q] = finger
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        hits = []
        for c in d.contact[: d.ncon]:
            g_arm, g_env = (c.geom1, c.geom2) if c.geom1 in self.arm_geoms else (c.geom2, c.geom1)
            if m.geom_bodyid[g_env] in ignore:
                continue
            hits.append((m.geom(g_arm).name or f"geom{g_arm}", m.geom(g_env).name or m.body(m.geom_bodyid[g_env]).name,
                         float(c.dist)))
        return hits

    def check(self, poses, R, seed, world_qpos, ignore=()) -> tuple[bool, list, list]:
        """IK + collision check of the hand at each (grip-site position, finger opening), rotation R."""
        qs, hits = [], []
        for pos, finger in poses:
            q, err = self.arm.ik(pos, R, seed=seed, iters=80)
            if err > IK_TOL:
                return False, qs, [("ik", f"{err * 1000:.1f} mm", 0.0)]
            h = self.collisions(q, world_qpos, finger, ignore)
            if h:
                return False, qs, h
            qs.append(q)
            seed = q
        return True, qs, hits

    # --- Carton slots ------------------------------------------------------------------------
    def candidates(self, xy):
        """Hand rotations to try at a slot, most natural first: vacuum cup facing the robot,
        hand vertical; then turned, then leaning (the wrist away from the nearest walls)."""
        inward = -np.asarray(xy, float) / np.linalg.norm(xy)
        out = []
        for lean_deg in [a for a in (0, 10, 18, 26, 32) if a <= L.MAX_LEAN]:
            for gamma in ((0,) if lean_deg == 0 else (0, 30, -30, 60, -60, 90, -90, 135, -135, 180)):
                for alpha in (0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100):
                    cost = lean_deg / 8 + abs(alpha) / 60 + abs(gamma) / 90
                    R = leaned(topdown(turn(inward, np.deg2rad(alpha))), turn(inward, np.deg2rad(gamma)),
                               np.deg2rad(lean_deg))
                    out.append((cost, lean_deg, R))
        out.sort(key=lambda c: c[0])
        return out

    def plan_slot(self, slot_xy, down_xy, surface, world_qpos, seed, transit_z: float) -> SlotPlan:
        """Hand pose for packing a part at slot_xy (set down at down_xy, then pushed)."""
        grip = np.r_[slot_xy, surface + L.GZ]
        down = np.r_[down_xy, surface + L.GZ + 0.004]
        push_end = grip + np.r_[PUSH_REACH * L.PUSH_DIR, 0.0]   # the loose jaws end up past the bore centre
        best, found = None, []
        for _, lean_deg, R in self.candidates(slot_xy):
            out = push_end - (L.JAW_INSERT + 0.01) * R[:, 2]  # jaws pulled out along their length
            poses = [(push_end, L.JAW_LOOSE), (down, L.JAW_HOLD), (down + 0.10 * UP, L.JAW_HOLD), (out, L.JAW_LOOSE),
                     (np.r_[down[:2], transit_z], L.JAW_HOLD), (np.r_[out[:2], transit_z], L.JAW_LOOSE)]
            ok, qs, hits = self.check(poses, R, seed, world_qpos)
            if ok:
                found.append(SlotPlan(R, qs[0], lean_deg))
                if len(found) == N_ALTS:
                    break
                continue
            n = len(hits)
            if hits and hits[0][0] != "ik" and (best is None or n < best[0]):
                best = (n, R, lean_deg, hits, qs)
        if found:
            first = found[0]
            first.alts = found
            return first
        if best is None:
            return SlotPlan(topdown(-np.asarray(slot_xy) / np.linalg.norm(slot_xy)), seed, 0.0, ok=False)
        _, R, lean_deg, hits, qs = best
        return SlotPlan(R, qs[0] if qs else seed, lean_deg, ok=False, hits=hits)

    # --- Placing a held part ---------------------------------------------------------------
    def plan_place(self, p_rel, R_rel, slot_xy, down_xy, surface, world_qpos, seed, transit_z, R_pref, held_body):
        """Hand rotation for the part as it really sits in the hand (p_rel, R_rel: its pose in the
        grip-site frame): the part stands upright, the hand turns about the vertical to whichever
        yaw is clear of the carton and the packed parts (checked where they really are), the one
        closest to R_pref first. Returns None if no yaw is clear."""
        cands = []
        for psi in np.deg2rad(np.arange(0, 360, 15)):
            c, s_ = np.cos(psi), np.sin(psi)
            R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ R_rel.T
            cands.append((-np.trace(R.T @ R_pref), R))
        cands.sort(key=lambda c: c[0])
        M = R_pref @ R_rel                             # first the yaw that matches R_pref best
        psi = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
        c, s_ = np.cos(psi), np.sin(psi)
        cands.insert(0, (0.0, np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ R_rel.T))
        u = np.r_[L.PUSH_DIR, 0.0]
        for _, R in cands[:12]:
            down = np.r_[down_xy, surface + 0.004] - R @ p_rel
            push_end = np.r_[slot_xy, surface] - R @ p_rel + PUSH_REACH * u
            out = push_end - (L.JAW_INSERT + 0.01) * R[:, 2]
            poses = [(down, L.JAW_HOLD), (push_end, L.JAW_LOOSE), (out, L.JAW_LOOSE), (down + 0.10 * UP, L.JAW_HOLD),
                     (np.r_[down[:2], transit_z], L.JAW_HOLD)]
            ok, _, _ = self.check(poses, R, seed, world_qpos, ignore={held_body})
            if ok:
                return R
        return None

    # --- Tray pick --------------------------------------------------------------------------
    def plan_pick(self, p_part, R_part, R_slot, world_qpos, held_body, transit_z, seed):
        """Hand rotation for taking the part off the tray, such that it ends up upright at the slot
        with the hand at R_slot. The part is round: any turn phi about its own axis will do, so
        take the first one that is reachable and clear of the tray walls and the other parts."""
        axis = R_part[:, 2]
        grip = p_part + L.GZ * axis
        inward = -p_part[:2] / np.linalg.norm(p_part[:2])
        cands = []
        for phi in np.deg2rad(np.arange(0, 360, 20)):
            c, s = np.cos(phi), np.sin(phi)
            Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            R = R_part @ Rz @ R_slot
            cup_side = R[:2, 0] / max(1e-9, np.linalg.norm(R[:2, 0]))
            cands.append((-float(cup_side @ inward), R))
        cands.sort(key=lambda c: c[0])
        fallback = None
        for _, R in cands:
            app = R[:, 2]
            poses = [(grip - 0.07 * app, L.JAW_CLOSED), (grip, L.JAW_HOLD), (grip + 0.04 * axis, L.JAW_HOLD),
                     (np.r_[grip[:2] + 0.04 * axis[:2], transit_z], L.JAW_HOLD)]
            ok, _, hits = self.check(poses, R, seed, world_qpos, ignore={held_body})
            if ok:
                return R, True
            if fallback is None and hits and hits[0][0] != "ik":
                fallback = R
        return (fallback if fallback is not None else cands[0][1]), False


def nominal_surface(layer: int, sheets: bool) -> float:
    """Height the parts of a layer stand on (the sheets are 3 mm thick)."""
    return L.BOX_FLOOR_Z + layer * L.PART_H + ((layer + 1) * L.SHEET_T if sheets else 0.0)


def plan_carton(planner: Planner, world_qpos, part_bodies, layers: int, sheets: bool, transit_z: float) -> dict:
    """Hand pose for every slot of every layer, with the parts packed before it in place."""
    nr, nc = L.GRID
    slots = [(r, c) for r in range(nr) for c in range(nc)]
    per = len(slots)
    seed, _ = planner.arm.ik(np.r_[L.BOX_CENTER, nominal_surface(0, sheets) + L.GZ + 0.1], topdown([-1, 0]), iters=200)
    plans = {}
    w = np.array(world_qpos, float)
    for layer in range(layers):
        surface = nominal_surface(layer, sheets)
        for i, (r, c) in enumerate(slots):
            plan = planner.plan_slot(L.slot_xy(r, c), L.set_down_xy(r, c), surface, w, seed, transit_z)
            plans[layer, (r, c)] = plan
            if plan.ok:
                seed = plan.q
            a = planner.part_q[part_bodies[layer * per + i]]    # now this part is packed too
            w[a:a + 3] = [*L.slot_xy(r, c), surface]
            w[a + 3:a + 7] = [1, 0, 0, 0]
    return plans
