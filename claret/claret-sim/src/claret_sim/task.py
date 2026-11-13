"""Packing routine: take the parts off the machine's outlet tray and pack them tight in the carton.

Machine  : finishes parts at random intervals; each one slides down the wide outlet tray and
           stops wherever it stops (machine.py). The tray camera reports the parts lying still.
Robot    : takes the part nearest the end stop. Its narrow jaws go into the part's bore
           closed, then open and grip it from the inside, so nothing sticks out around the
           part. In the carton it sets the part down 12 mm short of its pocket, lets go,
           and pushes it with the jaws into the pocket until it stops against its
           neighbours (or the walls): square packing, parts touching, filled to the walls.
Poses    : near the walls the hand leans and turns to keep the gripper, the forearm and the
           vacuum cup clear of the cardboard. These poses are planned with collision checks
           when the cell starts (planner.py).
Sheets   : (optional, on by default) with the retractable vacuum cup, the robot lays a
           cardboard sheet from the magazine on the carton floor, between the layers, and
           on top of the last layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from claret_sim import layout as L
from claret_sim.arm import Tick, parallel, rot_log
from claret_sim.planner import PUSH_REACH, topdown

if TYPE_CHECKING:
    from claret_sim.sim import CellSim

UP = np.array([0.0, 0.0, 1.0])
GZ = L.GZ                                             # grip point height above the part bottom
WALL_TOP = L.BOX_FLOOR_Z + L.BOX_H
TRANSIT_Z = WALL_TOP + 0.045 + GZ                     # grip-site height when carrying a part around
CUP_LIP = L.CUP_EXTENDED - L.GRIP_SITE                # extended cup lip below the grip site
SHEET_TRANSIT_Z = WALL_TOP + 0.045 + CUP_LIP          # ... carrying a sheet (6 cm above the carton wall)
# Sheets: the cup holds the sheet at its centre, beside the hand. The hand turns by 90 deg
# between the magazine and the carton (and the sheet with it).
U_MAG = np.array([-1.0, 0.0])                         # cup side at the magazine
U_BOX = np.array([0.0, -1.0])                         # cup side when laying the sheet
PUSH_SPEED = 0.03                                     # m/s
DESCENT_SPEED = 0.15                                  # m/s average, into the carton
PUSH_OVERTRAVEL = 0.022                               # jaw play in the bore + a few mm to spare


def transit_z(y: float) -> float:
    """Grip-site height for carrying a part: above the carton walls and the tray walls at y."""
    return max(TRANSIT_Z, L.chute_z(y) + L.CHUTE_WALL_H + 0.035 + GZ)


def hand_for(R_obj_goal, R_rel, R_pref) -> np.ndarray:
    """Hand rotation that puts a held object (rotation R_rel in the hand) at R_obj_goal,
    turned about the vertical by whatever yaw keeps the hand closest to R_pref."""
    M = R_pref @ R_rel @ R_obj_goal.T                 # ~ yaw that would be ideal
    yaw = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ R_obj_goal @ R_rel.T


def wait_until(sim: CellSim, cond, what: str, hint_after: float = 3.0) -> Tick:
    t0, hinted = sim.data.time, False
    while not cond():
        if not hinted and sim.data.time - t0 > hint_after:
            sim.say(f"waiting for {what}")
            hinted = True
        sim.waiting = True
        yield
    sim.waiting = False


def routine(sim: CellSim) -> Tick:
    R, M, cup = sim.arm, sim.machine, sim.cup
    m, d = sim.model, sim.data
    nr, nc = L.GRID
    slots = [(r, c) for r in range(nr) for c in range(nc)]
    below: dict[tuple[int, int], int] = {}            # slot -> body id of the part standing there
    sheet_top: float | None = None                    # top of the sheet the current layer stands on
    used: set[str] = set()                            # sheets taken from the magazine
    layer_of = [0]                                    # current layer (for the log)

    def surface(slot) -> float:
        if sheet_top is not None:
            return sheet_top
        if slot in below:
            return float(d.xpos[below[slot]][2]) + L.PART_H
        return L.BOX_FLOOR_Z

    def held_pose(body):
        p_s, R_s = R.site_pose()
        return R_s.T @ (d.xpos[body] - p_s), R_s.T @ d.xmat[body].reshape(3, 3)

    def pick_part(n: int, plan) -> Tick:
        sim.phase = "pick"
        told = False
        while True:
            yield from wait_until(sim, lambda: bool(M.pickable()), "a part on the tray")
            choice = None
            for k in M.pickable():                     # nearest the end stop first
                body = M.parts[k]
                p, Rp = d.xpos[body].copy(), d.xmat[body].reshape(3, 3).copy()
                tz = transit_z(p[1])
                for alt in plan.alts or [plan]:        # the slot's hand poses, most natural first
                    Rh, ok = sim.planner.plan_pick(p, Rp, alt.R, d.qpos.copy(), body, tz, R.q_cmd)
                    if ok:
                        choice = k, body, p, Rp, tz, Rh, alt
                        break
                if choice:
                    break
            if choice is None:                         # e.g. a part against the tray wall
                nudged = yield from nudge(n)
                if not nudged:
                    if not told:
                        sim.say(f"Part {n}: no clear way to take the parts on the tray, waiting for another one")
                        told = True
                    sim.waiting = True
                    yield from R.wait(0.5)
                continue
            k, body, p, Rp, tz, Rh, plan = choice
            axis, app = Rp[:, 2], Rh[:, 2]
            g = p + GZ * axis                          # grip point: on the part axis, in the bore
            entry = g - 0.07 * app                     # jaw tips 4.5 cm above the part, in line with the bore
            sim.say(f"Part {n}: take it off the tray")
            yield from parallel(R.gripper(L.JAW_CLOSED, 0.3), R.swing_to(np.r_[entry[:2], tz], Rh, 1.1))
            yield from R.move_to(entry, Rh, 0.35)
            yield from R.settle(tol=0.001, timeout=1.0)
            if np.linalg.norm(d.xpos[body] - p) > 0.003:   # bumped by a newcomer: look again
                sim.say(f"Part {n}: the part moved, looking again")
                yield from R.move_to(np.r_[entry[:2], tz], Rh, 0.3)
                continue
            yield from R.move_to(g, Rh, 0.6)           # jaws into the bore, along their length
            yield from R.settle(tol=0.0015, timeout=0.6)
            depth = (d.xmat[body].reshape(3, 3).T @ (R.site_pose()[0] - d.xpos[body]))[2] - GZ
            if depth > 0.006:                          # the jaws sit on the rim, not in the bore
                sim.say(f"Part {n}: jaws on the rim ({depth * 1000:.0f} mm high), trying again")
                yield from R.move_to(entry, Rh, 0.4)
                continue
            yield from R.gripper(L.JAW_GRIP, 0.25)     # open: the jaws press on the bore
            gripped = yield from sim.grasp_assist(body)
            if not gripped:                            # jaws missed the bore: back out and look again
                sim.say(f"Part {n}: missed the grip, trying again")
                yield from R.gripper(L.JAW_CLOSED, 0.2)
                yield from R.move_to(entry, Rh, 0.4)
                continue
            M.taken(k)
            sim.waiting = False
            told = False
            held = held_pose(body)
            yield from R.move_to(g + 0.04 * axis, Rh, 0.3)
            yield from R.move_to(np.r_[g[:2] + 0.04 * axis[:2], tz], Rh, 0.4)  # clear of the tray walls
            yield from R.settle(tol=0.004, timeout=0.5)
            return body, held, plan

    def nudge(n: int) -> Tick:
        """No part on the tray can be taken with the hand poses of this slot (typically a part
        lying against a tray wall: the leaning hand would hit the wall). Put the closed jaws
        into the bore of the nearest part with the hand vertical, and push it towards the
        middle of the tray. Returns False if that is not possible either."""
        for k in M.pickable():
            body = M.parts[k]
            p, Rp = d.xpos[body].copy(), d.xmat[body].reshape(3, 3).copy()
            if abs(p[0] - L.CHUTE_X) < 0.06:
                continue                               # already in the middle: nothing to gain
            tz = transit_z(p[1])
            Rh, ok = sim.planner.plan_pick(p, Rp, topdown(-p[:2]), d.qpos.copy(), body, tz, R.q_cmd)
            if not ok:
                continue
            sim.say(f"Part {n}: the part lies against the tray wall, pushing it to the middle first")
            axis, app = Rp[:, 2], Rh[:, 2]
            g = p + GZ * axis
            entry = g - 0.07 * app
            yield from parallel(R.gripper(L.JAW_CLOSED, 0.3), R.swing_to(np.r_[entry[:2], tz], Rh, 1.1))
            yield from R.move_to(entry, Rh, 0.35)
            yield from R.settle(tol=0.001, timeout=1.0)
            yield from R.move_to(g, Rh, 0.6)
            yield from R.settle(tol=0.0015, timeout=0.6)
            goal = np.array([L.CHUTE_X + np.sign(p[0] - L.CHUTE_X) * 0.05, p[1]])
            yield from push(body, goal, Rh, log=False)
            out = R.cmd_pose()[0] - 0.06 * app
            yield from R.move_to(out, Rh, 0.4)
            yield from R.move_to(np.r_[out[:2], tz], Rh, 0.3)
            yield from R.wait(0.3)                     # let it settle (the tray camera waits for it too)
            return True
        return False

    def push(body, goal_xy, R_hand, log: bool = True) -> Tick:
        """Push the released part with the jaws towards its pocket until it stops."""
        p0, _ = R.cmd_pose()
        start = d.xpos[body][:2].copy()
        u = goal_xy - start
        dist = float(np.linalg.norm(u))
        u = u / dist if dist > 1e-4 else L.PUSH_DIR
        # never further than the pose the planner checked: grip site PUSH_REACH past the pocket
        reach = float((goal_xy + PUSH_REACH * L.PUSH_DIR - p0[:2]) @ u) if log else dist + PUSH_OVERTRAVEL
        dof = M.dadr[M.parts.index(body)]
        travel, still, moved = 0.0, 0.0, False
        while travel < min(dist + PUSH_OVERTRAVEL, reach):
            travel += PUSH_SPEED * R.dt
            R.q_cmd, _ = R.ik(p0 + travel * np.r_[u, 0.0], R_hand, iters=8)
            yield
            moved = moved or np.linalg.norm(d.xpos[body][:2] - start) > 0.0015
            v = float(np.linalg.norm(d.qvel[dof:dof + 2]))
            still = still + R.dt if (moved and v < 0.004) else 0.0
            if still > 0.12:                           # it sits in its pocket: stop pushing
                break
        if log:
            sim.push_log.append((float(np.linalg.norm(d.xpos[body][:2] - start)), dist))
        yield from R.move_to(R.cmd_pose()[0] - 0.003 * np.r_[u, 0.0], R_hand, 0.15)  # unload the jaws

    def place_part(n: int, body: int, held, slot, plan) -> Tick:
        sim.phase = "place"
        final, down = L.slot_xy(*slot), L.set_down_xy(*slot)
        sim.say(f"Part {n}: into the carton, row {slot[0] + 1} column {slot[1] + 1}"
                + (f" (hand leaning {plan.lean:.0f} deg)" if plan.lean else ""))
        # Hand pose that stands the part upright on the set-down spot, whatever its pose in the hand,
        # turned to a yaw that is clear of the carton and the parts already packed.
        level = surface(slot) + 0.004

        def hand_pose(p_rel, R_rel):
            R_ = sim.planner.plan_place(p_rel, R_rel, final, down, level - 0.004, d.qpos.copy(), plan.q, TRANSIT_Z,
                                        plan.R, body)
            if R_ is None:
                sim.say(f"Part {n}: no clear hand pose for this grip, going with the nearest one")
                R_ = hand_for(np.eye(3), R_rel, plan.R)
            return R_

        p_rel, R_rel = held
        R_hand = hand_pose(p_rel, R_rel)
        target = np.r_[down, level] - R_hand @ p_rel
        above = np.r_[target[:2], TRANSIT_Z]
        drop = above[2] - target[2]
        q_above, err = R.ik(above, R_hand, seed=plan.q, iters=120)   # in the planned arm configuration
        yield from R.swing_to(above, R_hand, 1.2, q_goal=q_above if err < 2e-3 else None)
        yield from R.settle(tol=0.002, timeout=1.2)
        # Re-measure the part in the hand (it may have shifted on the way) and correct.
        p_rel2, R_rel2 = held_pose(body)
        if np.linalg.norm(rot_log(R_rel2 @ R_rel.T)) > np.deg2rad(2):   # the part has turned in the jaws
            R_hand = hand_pose(p_rel2, R_rel2)
        p_rel = p_rel2
        target = np.r_[down, level] - R_hand @ p_rel
        # Down to just above the surface; let the arm catch up while the part is still free,
        # then correct the last millimetres from where the part really is, and set it down.
        # (Slowly: the parts go in with 8 mm to spare, and a fast descent swings the hand out.)
        yield from R.move_to(target + 0.015 * UP, R_hand, 0.4 + drop / DESCENT_SPEED)
        yield from R.settle(tol=0.0015, timeout=1.0)
        err = down - d.xpos[body][:2]
        if np.linalg.norm(err) > 0.0005:
            target[:2] += err
            yield from R.move_to(target + 0.015 * UP, R_hand, 0.2)
            yield from R.settle(tol=0.001, timeout=0.6)
        yield from R.move_to(target, R_hand, 0.3)
        sim.release_assist()
        yield from R.gripper(L.JAW_LOOSE, 0.2)        # let go: the jaws close back to the bore centre
        sim.phase = "push"
        if np.linalg.norm(d.xpos[body][:2] - down) < 0.01:
            yield from push(body, final, R_hand)
        else:
            sim.say(f"Part {n}: set down {np.linalg.norm(d.xpos[body][:2] - down) * 1000:.0f} mm off, not pushing")
        below[slot] = body
        sim.placed.append((layer_of[0], slot, body))
        sim.packed += 1
        out = R.cmd_pose()[0] - (L.JAW_INSERT + 0.01) * R_hand[:, 2]
        yield from R.move_to(out, R_hand, 0.35)        # jaws out of the bore, along their length
        yield from R.move_to(np.r_[out[:2], TRANSIT_Z], R_hand, 0.3 + 1.0 * drop)

    def lay_sheet(where: str, level: float) -> Tick:
        """Take the top sheet from the magazine and lay it flat in the carton at `level`."""
        nonlocal sheet_top
        sim.phase = "sheet_pick"
        left = [k for k in range(L.N_SHEETS) if f"sheet_{k}" not in used]
        if not left:
            sim.say("sheet magazine empty: refill it (packing goes on without sheets)")
            return
        k = max(left, key=lambda i: sim.body_z(f"sheet_{i}"))
        used.add(f"sheet_{k}")
        sim.say(f"Cardboard sheet {where}: take it from the magazine")
        c = d.xpos[m.body(f"sheet_{k}").id]
        top = c[2] + L.SHEET_T / 2
        Rm = topdown(U_MAG)
        hand = np.r_[c[:2] - L.CUP_OFFSET * U_MAG, 0.0]
        yield from R.gripper(L.JAW_CLOSED, 0.2)
        cup.extend()
        yield from R.swing_to(np.r_[hand[:2], max(SHEET_TRANSIT_Z, top + CUP_LIP + 0.06)], Rm, 1.5)
        yield from wait_until(sim, cup.extended, "the cup cylinder")
        yield from R.move_to(np.r_[hand[:2], top + CUP_LIP + 0.02], Rm, 0.5)
        yield from R.settle(tol=0.002, timeout=0.8)
        yield from R.move_to(np.r_[hand[:2], top + CUP_LIP - 0.001], Rm, 0.5)
        yield from R.settle(tol=0.002, timeout=0.6)
        cup.vacuum = True  # seal while the hand is still: the sheet keeps its pose in the hand
        yield from wait_until(sim, lambda: cup.held is not None, "vacuum on the sheet", hint_after=1.0)
        yield from R.wait(0.15)
        p_s, R_s = R.site_pose()
        body = m.body(f"sheet_{cup.held}").id
        p_rel = R_s.T @ (d.xpos[body] - p_s)            # sheet pose in the hand, as sealed
        R_rel = R_s.T @ d.xmat[body].reshape(3, 3)
        yield from R.move_to(np.r_[p_s[:2], SHEET_TRANSIT_Z], R_s, 1.0)  # straight up, clear of the guides
        sim.phase = "sheet_place"
        sim.say(f"Cardboard sheet {where}: lay it in the carton")
        # Hand pose that puts the sheet centred and square in the carton (long side along y). The
        # sheet is symmetric, so take the one of the two yaws closest to the natural wrist pose.
        Rb = topdown(U_BOX)
        cands = []
        for yaw in (np.pi / 2, -np.pi / 2):            # sheet long side (its x) along the carton's y
            c_, s_ = np.cos(yaw), np.sin(yaw)
            R_c = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]) @ R_rel.T
            cands.append((np.trace(R_c.T @ Rb), R_c))
        R_hand = max(cands, key=lambda x: x[0])[1]
        drop = level + 0.006 + L.SHEET_T / 2
        place = np.r_[L.BOX_CENTER, drop] - R_hand @ p_rel
        lift = SHEET_TRANSIT_Z - place[2]
        yield from R.swing_to(place + lift * UP, R_hand, 2.0)
        yield from R.settle(tol=0.002, timeout=1.0)
        yield from R.move_to(place, R_hand, 0.6 + 3.0 * lift)
        yield from R.settle()
        cup.vacuum = False
        yield from R.wait(0.5)
        sheet_top = float(d.xpos[body][2]) + L.SHEET_T / 2
        sim.sheets_laid += 1
        yield from R.move_to(place + lift * UP, R_hand, 0.8)
        cup.retract()

    yield from R.wait(0.3)
    n = 0
    if sim.sheets:
        yield from lay_sheet("on the carton floor", L.BOX_FLOOR_Z)
    for layer in range(sim.layers):
        layer_of[0] = layer
        sim.say(f"Layer {layer + 1}/{sim.layers}")
        for slot in slots:
            n += 1
            body, held, plan = yield from pick_part(n, sim.plans[layer, slot])
            yield from place_part(n, body, held, slot, plan)
        sheet_top = None
        if sim.sheets:
            level = max(float(d.xpos[b][2]) + L.PART_H for b in below.values())
            where = "between the layers" if layer < sim.layers - 1 else "on top"
            yield from lay_sheet(where, level)
    sim.phase = "done"
    sim.say(f"Carton full: {sim.packed} parts in {sim.layers} layers"
            + (f", {sim.sheets_laid} cardboard sheets" if sim.sheets else "") + ". Swap the carton.")
    yield from R.gripper(L.JAW_CLOSED, 0.3)
    yield from R.move_joints(L.ARM_HOME_Q, 2.5)
