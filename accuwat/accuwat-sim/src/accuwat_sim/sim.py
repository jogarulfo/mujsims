"""Cell simulation: physics stepping, electromagnet, rack check."""

from __future__ import annotations

import mujoco
import numpy as np

from accuwat_sim import constraints, task
from accuwat_sim import layout as L
from accuwat_sim.arm import Arm
from accuwat_sim.scene import ARM, box_cells, build_model

CONTROL_DT = 0.01
LED_ON, LED_OFF = [1.0, 0.12, 0.08, 1], [0.25, 0.08, 0.08, 1]


class Magnet:
    """Holding electromagnet on the wrist. Energised, it holds the cell its face touches (a weld
    at the pose of the moment of contact stands in for the magnetic force). The coil takes
    MAGNET_RISE to build up and to let go."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m, self.d = model, data
        self.face = model.geom(ARM + "magnet").id
        self.led = model.geom("magnet_led").id
        self.site = model.site(ARM + "magnet_site").id
        self.cells = {model.body(f"cell_{k}").id: k for k in range(len(box_cells()))}
        self.cell_geom = {b: model.geom(f"cell_{k}").id for b, k in self.cells.items()}
        self.on = False                  # coil command
        self._t_switch = -1.0
        self.held: int | None = None     # body held
        self.releasing: int | None = None
        self.grabbed_wrong: list[int] = []
        self.pick_offsets: list[float] = []   # magnet axis to cell axis when it catches a cell

    def energise(self, on: bool) -> None:
        if on != self.on:
            self.on, self._t_switch = on, self.d.time

    def field(self) -> bool:
        """Is the holding force there (the coil current has built up)?"""
        settled = self.d.time - self._t_switch >= L.MAGNET_RISE
        return self.on if settled else not self.on

    def touching(self) -> int | None:
        for c in self.d.contact[: self.d.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g == self.face and self.m.geom_bodyid[other] in self.cells:
                    return int(self.m.geom_bodyid[other])
        return None

    def step(self) -> None:
        if self.field() and self.held is None:
            b = self.touching()
            if b is not None:
                constraints.activate_weld(self.m, self.d, "magnet", body2=b)
                self.pick_offsets.append(float(np.linalg.norm(self.d.site_xpos[self.site][:2] - self.d.xpos[b][:2])))
                # While held, the cell does not collide with the arm (the weld holds it against the face).
                self.m.geom_conaffinity[self.cell_geom[b]] = 0
                self.held = b
        elif not self.field() and self.held is not None:
            constraints.deactivate(self.m, self.d, "magnet")
            self.releasing, self.held = self.held, None
        if self.releasing is not None:  # touch the arm again once the face has come off the cell
            b = self.releasing
            gap = self.d.site_xpos[self.site][2] - (self.d.xpos[b][2] + L.CELL_H / 2)
            if gap > 0.003:
                self.m.geom_conaffinity[self.cell_geom[b]] = 1
                self.releasing = None
        self.m.geom_rgba[self.led] = LED_ON if self.on else LED_OFF


class CellSim:
    def __init__(self, n_cells: int | None = None, verbose: bool = True, sleep: bool = True):
        self.model, self.spec = build_model()
        self.data = mujoco.MjData(self.model)
        self.verbose = verbose
        self.n_sub = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.arm = Arm(self.model, self.data, ARM, CONTROL_DT)
        self.arm.set_home()
        self.magnet = Magnet(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if sleep:
            # Resting cells sleep (no contacts to compute) and wake when the magnet touches them.
            # Switched on after the first forward pass: MuJoCo 3.16 fails on a model that starts
            # with sleeping enabled and contacts not computed yet.
            self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
        self.slots = task.rack_order()
        self.n_cells = len(self.slots) if n_cells is None else min(n_cells, len(self.slots))
        self.placed: list[tuple[tuple[int, int], int]] = []   # (rack slot, cell body) in placing order
        self.messages: list[str] = []
        self.status = ""
        self.phase = "start"              # what the robot is doing (drives the video camera)
        self.block = 0
        self.box_left = {+1: L.BOX_GRID[0] * L.BOX_GRID[1], -1: L.BOX_GRID[0] * L.BOX_GRID[1]}
        self.cycle_times: list[float] = []
        self.arm_hits: dict[str, float] = {}   # arm geom - cell world contacts (deepest)
        self.lifting: int | None = None        # cell being lifted out of its box
        self._lift_start = None
        self.lift_dev = 0.0                    # largest sideways drift of a cell while still in the box
        self._script = task.routine(self)
        self.done = False
        for _ in range(int(0.3 / self.model.opt.timestep)):  # let the scene settle
            mujoco.mj_step(self.model, self.data)

    # --- Helpers used by the task ------------------------------------------------------------
    def say(self, msg: str) -> None:
        self.status = msg
        line = f"t={self.data.time:6.2f}s {msg}"
        self.messages.append(line)
        if self.verbose:
            print(line, flush=True)

    def cell_top(self, body: int) -> np.ndarray:
        """Centre of the upper end face of a standing cell."""
        return self.data.xpos[body] + np.array([0, 0, L.CELL_H / 2])

    # --- Stepping ---------------------------------------------------------------------------
    def _watch_arm(self) -> None:
        """Log contacts between the arm and the cell world (table, boxes, rack) or a cell it does
        not hold: there should be none, except the magnet face on the cell it picks."""
        m, d = self.model, self.data
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            n1, n2 = m.body(b1).name, m.body(b2).name
            arm_g = c.geom1 if n1.startswith(ARM) else c.geom2 if n2.startswith(ARM) else None
            if arm_g is None:
                continue
            other = c.geom2 if arm_g == c.geom1 else c.geom1
            if arm_g == self.magnet.face and m.geom_bodyid[other] in self.magnet.cells:
                continue                  # the magnet on a cell: that is the job
            key = f"{m.geom(arm_g).name or m.body(m.geom_bodyid[arm_g]).name} / " \
                  f"{m.geom(other).name or m.body(m.geom_bodyid[other]).name}"
            self.arm_hits[key] = min(self.arm_hits.get(key, 0.0), float(c.dist))

    def _watch_lift(self) -> None:
        """How far the cell being lifted strays sideways while its bottom is still below the
        neighbours' tops (in a real box it would rub on them)."""
        b = self.lifting
        if b is None:
            self._lift_start = None
            return
        if self._lift_start is None:
            self._lift_start = self.data.xpos[b][:2].copy()
        if self.data.xpos[b][2] - L.CELL_H / 2 < L.BOX_FLOOR_Z + L.CELL_H:
            dev = float(np.linalg.norm(self.data.xpos[b][:2] - self._lift_start))
            self.lift_dev = max(self.lift_dev, dev)

    def step(self) -> None:
        """Advance one control tick (CONTROL_DT)."""
        if not self.done:
            try:
                next(self._script)
            except StopIteration:
                self.done = True
        self.arm.apply()
        self.magnet.step()
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self._watch_arm()
        self._watch_lift()

    def rack_count(self) -> str:
        return f"{len(self.placed)}/{self.n_cells}"

    def rack_report(self) -> dict:
        """Where each placed cell stands and which way up, against the plan."""
        d = self.data
        offsets, tilts, heights, wrong = [], [], [], []
        per_block = {k: [0, 0] for k in range(len(L.BLOCKS))}       # block -> [right way up, cells]
        for (r, c), b in self.placed:
            p, z = d.xpos[b], d.xmat[b].reshape(3, 3)[:, 2]
            offsets.append(float(np.linalg.norm(p[:2] - L.rack_hole_xy(r, c))))
            tilts.append(float(np.degrees(np.arccos(min(1.0, abs(z[2]))))))
            heights.append(float(p[2] - L.CELL_H / 2 - L.RACK_BASE_Z))
            up = 1 if z[2] > 0 else -1
            blk = next(i for i, (c0, c1, _) in enumerate(L.BLOCKS) if c0 <= c < c1)
            per_block[blk][1] += 1
            if up == L.rack_polarity(c):
                per_block[blk][0] += 1
            else:
                wrong.append((r, c))
        return {"offsets": np.array(offsets), "tilts": np.array(tilts), "heights": np.array(heights),
                "wrong": wrong, "per_block": per_block}

    def draw(self, scn: mujoco.MjvScene, marks: bool = True) -> None:
        """Polarity check overlay: a small red (+) or blue (-) disc over each placed cell, the way
        it really stands (not the plan): a wrong cell would show the wrong colour in its block."""
        if not marks:
            return
        d = self.data
        for _, b in self.placed:
            if scn.ngeom >= scn.maxgeom:
                return
            up = d.xmat[b][8] > 0
            p = d.xpos[b] + np.array([0, 0, L.CELL_H / 2 + 0.0015])
            g = scn.geoms[scn.ngeom]
            rgba = np.array([0.95, 0.08, 0.05, 0.95] if up else [0.05, 0.15, 0.85, 0.95], dtype=np.float32)
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.005, 0.0005, 0]), p,
                                np.eye(3).reshape(-1), rgba)
            scn.ngeom += 1
