"""Rack-filling routine: take the cells from the two boxes and fill the rack + / - / +.

Boxes : the "+ up" box (left) and the "- up" box (right) hold 100 cells each, standing in
        rows. The robot always takes the cell nearest the rack in the box it needs.
Rack  : 15 x 8 staggered holes. Columns 1-5 get + up cells, columns 6-10 - up cells,
        columns 11-15 + up cells again (80 + cells, 40 - cells). A cell never turns over: it
        goes in the rack the way it stood in its box.
Robot : the electromagnet comes down on the top end of the cell (the + cap or the - can
        bottom, both nickel-plated steel), switches on, and lifts the cell straight out of the
        box. Above its hole, the robot corrects the last millimetres from where the cell really
        hangs, lowers it through the chamfered hole onto the base plate and switches off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from accuwat_sim import layout as L
from accuwat_sim.arm import UP, Tick, min_jerk, topdown
from accuwat_sim.scene import box_cells

if TYPE_CHECKING:
    from accuwat_sim.sim import CellSim

EMPTY_Z = L.TOPS + 0.030              # magnet face height when travelling empty (3 cm over the cell tops)
TOUCH_PRESS = 0.0008                  # the magnet comes down this far past the cell top (it stops on it)
PRE_INSERT = 0.004                    # cell bottom above the holder plate before going in
INSERT_SPEED = 0.04                   # m/s average, through the holder
TOUCH_SPEED = 0.05                    # m/s average, magnet down onto the cell
LIFT_SPEED = 0.08                     # m/s average, cell out of the box
LIFT_GAIN = 0.35                      # correction gain while lifting (per 10 ms tick)
SWING_LIFT = 0.02
READY_SEED = np.array([0.0, 2.0, 2.0, -1.0, 0.0, 0.0])   # elbow up, wrist bent down


def rack_order() -> list[tuple[int, int]]:
    """Rack holes in filling order: block by block (+, -, +), each from the far row (operator
    side) to the near row, columns from +y to -y."""
    out = []
    for c0, c1, _ in L.BLOCKS:
        for r in range(L.RACK_ROWS):
            for c in range(c0, c1):
                out.append((r, c))
    return out


def swing_time(a, b) -> float:
    return 0.7 + 1.3 * float(np.linalg.norm(np.asarray(b)[:2] - np.asarray(a)[:2]))


def wait_for(sim: CellSim, cond, timeout: float) -> Tick:
    t0 = sim.data.time
    while not cond() and sim.data.time - t0 < timeout:
        yield


def routine(sim: CellSim) -> Tick:
    R, mag = sim.arm, sim.magnet
    m, d = sim.model, sim.data
    # Cells left in each box, nearest the rack first.
    queues: dict[int, list[int]] = {+1: [], -1: []}
    for k, (*_, pol) in enumerate(box_cells()):
        queues[pol].append(m.body(f"cell_{k}").id)
    for queue in queues.values():
        queue.sort(key=lambda b: float(np.linalg.norm(d.xpos[b][:2] - L.RACK_CENTER)))

    def pick(n: int, body: int, pol: int) -> Tick:
        sim.phase = "pick"
        sign = "+" if pol > 0 else "-"
        sim.say(f"Cell {n}: take a cell from the '{sign} up' box")
        for attempt in range(3):
            top = sim.cell_top(body)
            Rh = topdown(top[:2])
            approach = top + L.APPROACH * UP
            p0 = R.cmd_pose()[0]
            yield from R.swing_to(np.r_[top[:2], EMPTY_Z], Rh, swing_time(p0, top), lift=SWING_LIFT)
            yield from R.move_to(approach, Rh, 0.3)
            # The 20 mm face must land square on the 21.1 mm cell, 0.85 mm from its neighbours:
            # vertical() corrects the tracking error on the way down.
            mag.energise(True)
            touch = top - TOUCH_PRESS * UP
            yield from vertical(approach, touch, Rh, TOUCH_SPEED, until=lambda: mag.held is not None,
                                hover=TOUCH_PRESS + 0.004)
            yield from wait_for(sim, lambda: mag.held is not None, 0.3)
            if mag.held is not None:
                break
            sim.say(f"Cell {n}: the magnet did not catch it, trying again")
            mag.energise(False)
            yield from R.move_to(approach, Rh, 0.3)
        else:
            raise RuntimeError(f"cell {n}: could not pick the cell")
        if mag.held != body:                            # a neighbour: take it instead
            sim.say(f"Cell {n}: warning, the magnet caught the neighbouring cell")
            mag.grabbed_wrong.append(mag.held)
            queue = queues[pol]
            if mag.held in queue:
                queue.remove(mag.held)
            queue.insert(0, body)
            body = mag.held
        yield from R.wait(0.03)
        # Straight up out of the box, slowly while the cell is still among its neighbours (a fast
        # lift makes the arm sway), then on to the carrying height.
        top = sim.cell_top(body)
        out = np.r_[top[:2], L.BOX_FLOOR_Z + 2 * L.CELL_H + 0.004]
        sim.lifting = body
        yield from vertical(R.site_pose()[0], out, Rh, LIFT_SPEED, gain=LIFT_GAIN)
        sim.lifting = None
        yield from R.move_to(np.r_[R.cmd_pose()[0][:2], L.TRANSIT_Z], Rh, 0.3)
        return body

    def vertical(start, goal, Rh, speed: float, until=None, hover: float = 0.0, gain: float = 0.2) -> Tick:
        """Straight down (or up) from start to goal. Even with the feed-forward terms (arm.py), the
        YAM's soft joint servos leave a tracking error of up to a millimetre or two, which pulls the
        hand off the vertical line; so the commanded x, y are corrected
        each tick from the measured tool position (joint encoders + forward kinematics).
        With `hover`, it stops that far above the goal until the hand is within 0.15 mm of the
        line and has stopped bobbing, and does the last part at 1 cm/s."""
        trim = R.cmd_pose()[0][:2] - start[:2]             # carry on from the correction already made

        def leg(a, b, v, hold: float):
            nonlocal trim
            n = max(1, round(abs(a[2] - b[2]) / v / R.dt))
            steady = 0
            for k in range(1, n + int(hold / R.dt) + 1):
                if until is not None and until():
                    return
                s = min_jerk(min(1.0, k / n))
                want = a + s * (b - a)
                site = R.site_pose()[0]
                err = want[:2] - site[:2]
                steady = steady + 1 if np.linalg.norm(err) < 0.00015 and abs(want[2] - site[2]) < 0.0005 else 0
                if k > n and steady >= 5:
                    return
                trim = np.clip(trim + gain * err, -0.004, 0.004)
                R.q_cmd, _ = R.ik(np.r_[want[:2] + trim, want[2]], Rh, iters=8)
                yield

        if hover > 0:
            mid = goal + hover * UP
            yield from leg(start, mid, speed, 0.8)
            yield from leg(mid, goal, 0.01, 0.3)
        else:
            yield from leg(start, goal, speed, 0.3)

    def place(n: int, body: int, slot: tuple[int, int]) -> Tick:
        sim.phase = "place"
        r, c = slot
        hole = L.rack_hole_xy(r, c)
        pol = L.rack_polarity(c)
        sim.say(f"Cell {n}: into the rack, row {r + 1} column {c + 1} ({'+' if pol > 0 else '-'} up)")
        centre = np.r_[hole, L.RACK_BASE_Z + L.CELL_H / 2 + L.INSERT_GAP]   # cell centre when released
        Rh = topdown(hole)

        def goal_for_hand():
            p_s, R_s = R.site_pose()
            p_rel = R_s.T @ (d.xpos[body] - p_s)            # the cell as it hangs under the magnet
            return centre - Rh @ p_rel

        goal = goal_for_hand()
        p0 = R.cmd_pose()[0]
        yield from R.swing_to(np.r_[goal[:2], L.TRANSIT_Z], Rh, swing_time(p0, goal), lift=SWING_LIFT)
        yield from R.settle(tol=0.0015, timeout=1.0)
        goal = goal_for_hand()                               # re-measure: it may have shifted on the way
        rise = L.HOLDER_TOP + PRE_INSERT - (L.RACK_BASE_Z + L.INSERT_GAP)
        pre = goal + rise * UP
        drop = L.TRANSIT_Z - pre[2]
        yield from R.move_to(pre, Rh, 0.25 + drop / 0.25)
        yield from R.settle(tol=0.0006, timeout=1.0)
        # Correct the last millimetres from where the cell really is (0.45 mm play in the hole).
        for _ in range(2):
            err = centre[:2] - d.xpos[body][:2]
            if np.linalg.norm(err) < 0.00015:
                break
            pre[:2] += err
            goal[:2] += err
            yield from R.move_to(pre, Rh, 0.15)
            yield from R.settle(tol=0.0004, timeout=0.6)
        sim.phase = "insert"
        yield from vertical(pre, goal, Rh, INSERT_SPEED)
        mag.energise(False)
        yield from wait_for(sim, lambda: mag.held is None, 0.5)
        yield from R.wait(0.04)
        sim.placed.append((slot, body))
        out = R.cmd_pose()[0]
        yield from R.move_to(np.r_[out[:2], EMPTY_Z], Rh, 0.3)

    yield from R.wait(0.3)
    # Ready pose: magnet pointing down over the middle of the rack (joint move, IK from a good seed).
    ready = np.r_[L.RACK_CENTER, EMPTY_Z + 0.05]
    q_ready, _ = R.ik(ready, topdown(ready), seed=READY_SEED, iters=300)
    yield from R.move_joints(q_ready, 2.0)
    block = -1
    for n, slot in enumerate(sim.slots[: sim.n_cells], start=1):
        b = next(i for i, (c0, c1, _) in enumerate(L.BLOCKS) if c0 <= slot[1] < c1)
        if b != block:
            block = sim.block = b
            c0, c1, pol = L.BLOCKS[b]
            sim.say(f"Block {b + 1}/3: columns {c0 + 1}-{c1}, cells {'+' if pol > 0 else '-'} up")
        pol = L.rack_polarity(slot[1])
        if not queues[pol]:
            sim.say(f"the '{'+' if pol > 0 else '-'} up' box is empty: refill it")
            break
        t0 = d.time
        body = yield from pick(n, queues[pol].pop(0), pol)
        sim.box_left = {k: len(q) for k, q in queues.items()}
        yield from place(n, body, slot)
        sim.cycle_times.append(d.time - t0)
    sim.phase = "done"
    rep = sim.rack_report()
    ok = sum(v[0] for v in rep["per_block"].values())
    sim.say(f"Rack done: {len(sim.placed)} cells, {ok} the right way up (+ / - / +). Swap the rack.")
    out = R.cmd_pose()[0]
    yield from R.move_to(np.r_[out[:2], EMPTY_Z + 0.05], R.cmd_pose()[1], 0.6)
    yield from R.move_joints(L.ARM_HOME_Q, 2.5)
