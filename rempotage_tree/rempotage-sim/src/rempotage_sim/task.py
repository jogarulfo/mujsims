"""Bimanual repotting routine, one plant after the other.

Line     : the wheel brings a pot, filled and dibbled, to the planting station and waits for the
           robots' "planting done" (line.py).
Arm A    : pulls a plant out of its square pot, front row of the first crate on the trolley (hand
           tilted 45 deg down, fingers closed on the trunk just above the root ball: the pots touch,
           there is no room beside the root ball), and waits with it next to the planting station
           while the wheel brings the pot. It sets the root ball almost to the bottom of the
           planting hole, lets go, and leaves at once to fetch the next plant: it is on its way
           while arm B pushes.
Arm B    : its fingertips carry tamper feet (a flat sole each, level in its tilted hand). It waits
           next to the pot while arm A sets the plant in, comes in with its hand tilted 45 deg (the
           trunk passes in front of its gripper), fingers closed around the trunk, and pushes the
           root ball the last centimetres down to the floor of the hole with both soles, on the plant
           only (one sole each side of the trunk), then firms it there. (With layout.TAMP_SIDES it also presses the soil
           beside the root ball, in two presses.)
The soil then holds the plant and the line indexes at once: arm A finishes its fetch meanwhile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rempotage_sim import layout as L
from rempotage_sim.arm import UP, Tick, grasp_frame, horizontal, parallel

if TYPE_CHECKING:
    from rempotage_sim.sim import CellSim

PICK_ORDER = tuple(range(L.N_PLANTS))      # cart places, nearest the wheel first
TAMP_YAW_MAX = np.deg2rad(60.0)           # arm B reaches the pot with its approach turned up to this
TAMP_GRIP = 0.045                         # soles from 45 to 67 mm either side of the trunk: on the soil ring
TAMP_FORCE = 30.0                         # N pressed down by arm B on the soil (two soles, four sectors)
TAMP_DEPTH = 0.012                        # m the press command goes below where the soles touch
PUSH_GRIP = 0.012                         # arm B's finger travel while pushing (24 mm gap, trunk 13 mm)
PUSH_SPEED = 0.04                         # m/s
FIRM_FORCE = 22.0                         # N, arm B firms the root ball on the floor of the hole
FIRM_TIME = 0.4                           # s
SOLE_DOWN = L.FOOT_TOP + L.FOOT_T         # arm B's soles: bottom face below its grasp site
A_STAGE_BACK = 0.13                       # arm A waits with the plant this far from the hole, towards its base
A_STAGE_UP = 0.12                         # root ball bottom this far above the soil while waiting


def rot_z(v, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def wait_until(sim: CellSim, cond, what: str, hint_after: float = 10.0) -> Tick:
    t0, hinted = sim.data.time, False
    while not cond():
        if not hinted and sim.data.time - t0 > hint_after:
            sim.say(f"waiting for {what}")
            hinted = True
        yield


def push_frame(h, tilt: float = L.PUSH_TILT) -> np.ndarray:
    """Arm B's pushing hand: approach along h tilted down by `tilt`, fingers closing across h."""
    h = horizontal(h)
    return grasp_frame(np.cos(tilt) * h - np.sin(tilt) * UP, np.cross(UP, h))


def background(main: Tick, job: Tick | None) -> Tick:
    """Run `main` with `job` alongside, one tick each, until `main` ends. Returns `job` if it is
    not finished yet (the caller keeps running it), else None."""
    live = job is not None
    for _ in main:
        if live:
            try:
                next(job)
            except StopIteration:
                live = False
        yield
    return job if live else None


def routine(sim: CellSim) -> Tick:
    A, B, line = sim.arm_a, sim.arm_b, sim.line
    yield from parallel(A.wait(0.3), B.wait(0.3))
    order = list(PICK_ORDER)
    p = order.pop(0)
    sim.say(f"Plant {p + 1}: arm A takes it from the cart")
    a_job: Tick | None = _fetch(sim, p, after_pot=None)
    done: set[int] = set()
    while p is not None:
        # Arm A finishes fetching the plant while the line brings the pot.
        def pot_here() -> Tick:
            while line.pot_ready is None or line.pot_ready in done:
                sim.phase = "pick" if sim.a_doing == "pick" else "wait"
                yield
        yield from parallel(pot_here(), *(j for j in (a_job,) if j is not None))
        k = line.pot_ready
        t0 = sim.data.time
        nxt = order.pop(0) if order else None
        a_job = yield from _plant(sim, p, k, nxt)
        sim.cycle_times.append(sim.data.time - t0)
        line.planting_done(k)
        done.add(k)
        p = nxt
    sim.phase = "home"
    yield from parallel(B.go_home(2.5), *(j for j in (a_job,) if j is not None))
    sim.phase = "line"
    sim.say("All plants potted: the line mulches the last pots and puts them on the pallet")
    yield from wait_until(sim, lambda: line.pallet_full, "the pallet to fill", hint_after=60.0)
    yield from A.wait(1.5)
    sim.say(f"Done: {len(sim.potted)} plants repotted, pallet full ({len(line.on_pallet)} pots)")


def grip_frame(h) -> np.ndarray:
    """Arm A's hand: approach along h tilted down by GRIP_TILT, fingers closing across h."""
    return push_frame(h, L.GRIP_TILT)


def _pick(sim: CellSim, p: int) -> Tick:
    """Arm A pulls plant p out of its pot in the crate: hand tilted down, fingers on the trunk just above
    the root ball."""
    A, m, d = sim.arm_a, sim.model, sim.data
    body = m.body(f"plant_{p}").id
    c = d.xpos[body].copy()
    # Radially from the base; for the plants beyond the base (along +y) turned halfway to straight
    # across the row, else the back of the hand brushes the next plant (6 cm from the root ball).
    a = horizontal(c[:2] - L.ARM_A_BASE)
    if a[1] > 0:
        a = horizontal(a + np.array([-1.0, 0.0, 0.0]))
    R = grip_frame(a)
    z = R[:, 2]                                               # approach: forwards and down
    g = c + (L.RB_H + L.TRUNK_GRIP_ABOVE) * UP
    yield from A.gripper(L.TRUNK_GRIP_OPEN, 0.3)
    yield from A.swing_to(g - 0.12 * z + 0.03 * UP, R, 1.9, lift=0.05, base_xy=L.ARM_A_BASE)
    yield from A.move_to(g - 0.12 * z, R, 0.4)
    yield from A.move_to(g, R, 0.8)                         # the trunk comes in between the fingers
    yield from A.settle(0.0015)
    yield from A.gripper(L.TREE_R_BASE - 0.003, 0.3)
    yield from sim.grasp_assist(f"plant_{p}", touch=f"plant_{p}/seg_0")
    sim.carried = p
    yield from A.move_to(g + 0.11 * UP, R, 0.8)            # straight up, out of its pot and the crate
    yield from A.move_to(g + 0.18 * UP - 0.06 * a, R, 0.5)


def _hole_nominal() -> np.ndarray:
    """Planting hole of a pot in the S3 nest (centre at the soil surface)."""
    return np.r_[L.station_xy(L.PLANT_STATION), L.WHEEL_Z + L.SOIL_Z]


def _site_for_bottom(sim: CellSim, plant: int, R: np.ndarray):
    """Where arm A's grasp site goes to put the root ball's bottom centre at a point, holding it in R
    (from where the root ball really hangs in the hand)."""
    ps, Rs = sim.arm_a.site_pose()
    bottom_in_site = Rs.T @ (sim.data.xpos[plant] - ps)
    return lambda xyz: np.asarray(xyz) - R @ bottom_in_site


def _fetch(sim: CellSim, p: int, after_pot: int | None) -> Tick:
    """Arm A fetches plant p and waits with it beside the planting station, once arm B is done with
    the previous pot (`after_pot`) and the wheel is indexing."""
    A = sim.arm_a
    sim.a_doing = "pick"
    yield from _pick(sim, p)
    sim.a_doing = "stage"
    while after_pot is not None and sim.line.pot_ready == after_pot:
        yield
    hole = _hole_nominal()
    R = grip_frame(hole[:2] - L.ARM_A_BASE)
    site_for = _site_for_bottom(sim, sim.model.body(f"plant_{p}").id, R)
    back = horizontal(np.r_[L.ARM_A_BASE, 0.0] - hole)
    yield from A.swing_to(site_for(hole + A_STAGE_UP * UP + A_STAGE_BACK * back), R, 2.2, lift=0.03,
                          base_xy=L.ARM_A_BASE)
    sim.a_doing = None


def _plant(sim: CellSim, p: int, k: int, next_plant: int | None) -> Tick:
    """Plant p into pot k (at S3). Returns arm A's fetch of the next plant, still under way."""
    A, B, soil, m, d = sim.arm_a, sim.arm_b, sim.soil, sim.model, sim.data
    tag = f"Plant {p + 1} -> pot {k + 1}"
    plant = m.body(f"plant_{p}").id
    pot = m.body(f"pot_{k}").id
    hole = d.xpos[pot].copy()
    soil_z = (d.xpos[pot] + d.xmat[pot].reshape(3, 3) @ [0, 0, L.SOIL_Z])[2]
    h_b = rot_z(horizontal(hole[:2] - L.ARM_B_BASE), L.PUSH_YAW)
    RB = push_frame(h_b)
    rb_top_final = soil_z * UP + np.r_[hole[:2], 0.0]
    b_wait = rb_top_final + (L.RB_H - L.PRE_INSERT + 0.15) * UP - 0.16 * h_b

    # ---- Arm A: from beside the pot, root ball into the hole; arm B moves up close -------------------
    sim.phase = "insert"
    sim.say(f"{tag}: arm A sets the root ball deep in the hole, arm B gets ready")
    R = grip_frame(hole[:2] - L.ARM_A_BASE)
    site_for_bottom = _site_for_bottom(sim, plant, R)

    def a_inserts() -> Tick:
        target = site_for_bottom(np.r_[hole[:2], soil_z + 0.01])
        if np.linalg.norm(A.cmd_pose()[0] - target) > 0.25:      # not waiting beside the pot
            yield from A.swing_to(site_for_bottom(np.r_[hole[:2], soil_z + 0.12]), R, 2.4, lift=0.03,
                                  base_xy=L.ARM_A_BASE)
        yield from A.move_to(target, R, 1.1)
        yield from A.settle(0.0015)
        # Correct the last millimetres from where the root ball really is (a camera would see the hole).
        err = hole[:2] - d.xpos[plant][:2]
        A.push = np.array([0.0, 0.0, -(L.SOIL_F0 + L.SOIL_F_PER_M * L.PRE_INSERT + 2.0)])
        yield from A.move_to(A.cmd_pose()[0] + np.r_[err, 0.0], R, 0.3)
        yield from A.move_to(site_for_bottom(np.r_[hole[:2] + err, soil_z - L.PRE_INSERT]), R, 1.5)
        yield from A.wait(0.2)
        A.push[:] = 0.0

    yield from parallel(a_inserts(), B.gripper(PUSH_GRIP, 0.4), B.move_to(b_wait, RB, 2.0))
    sim.say(f"{tag}: root ball {soil.depth(k, p) * 100:.1f} cm deep, arm A lets go and fetches the next plant")
    sim.release_assist()
    sim.carried = None
    yield from A.gripper(L.GRIPPER_OPEN, 0.25)

    # ---- Arm A leaves for the next plant at once; arm B comes in over the root ball and pushes it --------
    def a_next() -> Tick:
        ps, _ = A.cmd_pose()
        yield from A.move_to(ps - 0.08 * R[:, 2] + 0.03 * UP, R, 0.5)    # step back from the root ball
        if next_plant is None:
            yield from A.go_home(2.0)
        else:
            yield from _fetch(sim, next_plant, after_pot=k)

    rb_top = d.xpos[plant] + L.RB_H * UP
    b_above = rb_top + (SOLE_DOWN + 0.015) * UP + 0.006 * h_b    # soles 15 mm over the root ball

    def b_comes_in() -> Tick:
        yield from B.move_to(b_above + 0.05 * UP - 0.05 * h_b, RB, 1.0)
        yield from B.move_to(b_above, RB, 0.5)
        yield from B.settle(0.0015)

    floor_z = (d.xpos[pot] + d.xmat[pot].reshape(3, 3) @ [0, 0, L.HOLE_FLOOR_Z])[2]

    def b_pushes() -> Tick:
        sim.say(f"{tag}: arm B pushes the root ball the last {(L.HOLE_DEPTH - soil.depth(k, p)) * 100:.1f} cm down"
                + ("" if next_plant is None else f", arm A fetches plant {next_plant + 1}"))
        for _ in range(round(1.5 / sim.dt)):                    # down until the soles touch the root ball
            n, f = B.contacts({plant})
            if n >= 1 and f > 0.5:
                break
            ps, _ = B.cmd_pose()
            B.q_cmd, _ = B.ik(ps - 0.06 * sim.dt * UP, RB, iters=8)
            yield
        t_push = 0.0
        while d.xpos[plant][2] > floor_z + 0.001 and t_push < 12.0:
            # Feed-forward: the soil's resistance at this depth plus a little; the position command sets the pace.
            f_soil = L.SOIL_F0 + L.SOIL_F_PER_M * max(soil.depth(k, p), 0.0) + 2.0
            B.push = np.array([0.0, 0.0, -min(f_soil, 4.0 + 20.0 * t_push)])
            ps, _ = B.cmd_pose()
            lead = ps[2] - (d.xpos[plant][2] + L.RB_H + SOLE_DOWN)  # command at most ~4 mm under the soles
            step = PUSH_SPEED * sim.dt if lead > -0.004 else 0.0
            B.q_cmd, _ = B.ik(ps - step * UP, RB, iters=8)
            t_push += sim.dt
            yield
        sim.push_report.append((p, k, soil.max_force.get(k, 0.0), float(d.xpos[plant][2] - floor_z), t_push))
        # Firm the root ball down on the floor of the hole, then ease off.
        sim.say(f"{tag}: root ball at the bottom ({soil.max_force.get(k, 0.0):.1f} N at the end), arm B firms it")
        B.push = np.array([0.0, 0.0, -FIRM_FORCE])
        yield from B.wait(FIRM_TIME)
        n = round(0.3 / sim.dt)
        for i in range(n):
            B.push = np.array([0.0, 0.0, -FIRM_FORCE * (1 - (i + 1) / n)])
            yield
        B.push[:] = 0.0

    sim.phase = "push"
    a_job = yield from background(b_comes_in(), a_next())
    a_job = yield from background(b_pushes(), a_job)
    if L.TAMP_SIDES:                                          # optional: also firm the soil beside the root ball
        sim.phase = "tamp"
        a_job = yield from background(_tamp(sim, k, h_b, tag), a_job)
        sim.say(f"{tag}: soil firmed by {soil.comp[k].mean() * 1000:.1f} mm on average "
                f"(least {soil.comp[k].min() * 1000:.1f} mm)")

    # ---- The soil holds the plant: arm B steps back, the line takes the pot ----------------------------
    soil.plant_in(k, p)
    sim.potted.append((p, k))
    # Arm B goes up off the root ball, then back to its waiting pose (its own side, clear of the wheel).
    ps, Rb = B.cmd_pose()
    a_job = yield from background(B.move_to(ps + 0.015 * UP, Rb, 0.2), a_job)
    a_job = yield from background(B.move_to(b_wait, RB, 0.9), a_job)
    return a_job


def tamp_yaws(sim: CellSim, k: int, h_b: np.ndarray) -> list[float]:
    """Two presses, 90 deg apart, each with the soles across a boundary between soil sectors (so a
    press firms the four sectors either side of two boundaries), within arm B's reach."""
    Rp = sim.data.xmat[sim.model.body(f"pot_{k}").id].reshape(3, 3)
    pot_yaw = np.arctan2(Rp[1, 0], Rp[0, 0])
    da = 2 * np.pi / L.TAMP_SECTORS
    # Soles' line (the finger closing axis) at approach yaw y: atan2(h_b) + y + 90 deg.
    y0 = pot_yaw - np.arctan2(h_b[1], h_b[0]) - np.pi / 2
    y0 = (y0 + da / 2) % da - da / 2                          # in [-22.5, 22.5) deg
    # Positive yaw first: arm B's forearm on the side away from arm A while A swings out to the cart.
    return [float(np.clip(y0 + s * np.pi / 4, -TAMP_YAW_MAX, TAMP_YAW_MAX)) for s in (1, -1)]


def _tamp(sim: CellSim, k: int, h_b: np.ndarray, tag: str) -> Tick:
    """Arm B presses the soil ring around the root ball down with its two soles, twice."""
    B, soil, m, d = sim.arm_b, sim.soil, sim.model, sim.data
    pot = m.body(f"pot_{k}").id
    ps, _ = B.cmd_pose()
    yield from parallel(B.move_to(ps + 0.05 * UP - 0.02 * h_b, push_frame(h_b), 0.45),   # up, clear of the rim,
                        B.gripper(TAMP_GRIP, 0.45))                                     # opening the soles
    centre = d.xpos[pot] + L.SOIL_Z * d.xmat[pot].reshape(3, 3)[:, 2]
    pot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pot and m.geom_contype[g] == 1}

    def on_rim() -> bool:
        for c in d.contact[: d.ncon]:
            if (c.geom1 in pot_geoms and c.geom2 in B.finger_geoms) or (c.geom2 in pot_geoms and c.geom1 in B.finger_geoms):
                return True
        return False

    for n_tamp, yaw in enumerate(tamp_yaws(sim, k, h_b)):
        h = rot_z(h_b, yaw)
        Rt = push_frame(h)
        top = centre + (SOLE_DOWN + 0.012) * UP + 0.006 * horizontal(h)    # soles 12 mm over the soil
        err = min(B.ik(top, Rt, seed=s, iters=80)[1] for s in (B.q_cmd, B.home))
        if err > 2e-3:
            sim.say(f"{tag}: press {n_tamp + 1} skipped (out of reach)")
            continue
        yield from B.move_to(top + 0.02 * UP, Rt, 0.7 if n_tamp else 0.5)
        yield from B.move_to(top, Rt, 0.2)
        for _ in range(round(1.0 / sim.dt)):              # down until the soles are on the soil
            if soil.press[k] > 1.0 or on_rim():
                break
            ps, _ = B.cmd_pose()
            B.q_cmd, _ = B.ik(ps - 0.06 * sim.dt * UP, Rt, iters=8)
            yield
        if on_rim():
            sim.say(f"{tag}: press {n_tamp + 1}: a sole touches the pot rim, skipped")
        else:
            B.push = np.array([0.0, 0.0, -TAMP_FORCE])
            ps, _ = B.cmd_pose()
            yield from B.move_to(ps - TAMP_DEPTH * UP, Rt, 0.4)
            yield from B.wait(0.3)
            B.push[:] = 0.0
        ps, _ = B.cmd_pose()
        yield from B.move_to(ps + 0.03 * UP, Rt, 0.25)                # straight up off the soil
