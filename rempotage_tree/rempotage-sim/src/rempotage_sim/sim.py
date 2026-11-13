"""Cell simulation: physics stepping, line controller, soil, the two arms and their routine."""

from __future__ import annotations

import mujoco
import numpy as np

from rempotage_sim import constraints, task
from rempotage_sim import layout as L
from rempotage_sim.arm import Arm
from rempotage_sim.line import Line
from rempotage_sim.scene import build_model
from rempotage_sim.soil import Soil

CONTROL_DT = 0.01


class CellSim:
    def __init__(self, verbose: bool = True, assist: bool = True):
        self.model, self.spec = build_model()
        self.data = mujoco.MjData(self.model)
        self.verbose, self.assist = verbose, assist
        self.dt = CONTROL_DT
        self.n_sub = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.arm_a = Arm(self.model, self.data, "a/", CONTROL_DT)
        self.arm_b = Arm(self.model, self.data, "b/", CONTROL_DT)
        self.arm_a.set_home()
        self.arm_b.set_home()
        self.soil = Soil(self.model, self.data)
        self.line = Line(self.model, self.data, self.soil, CONTROL_DT)
        mujoco.mj_forward(self.model, self.data)
        self.line.reset()
        self.messages: list[str] = []
        self.status = ""
        self.phase = "start"
        self.a_doing: str | None = None        # arm A: "pick", "stage" or None
        self.carried: int | None = None        # plant in arm A's hand
        self.potted: list[tuple[int, int]] = []   # (plant, pot) in planting order
        self.push_report: list[tuple[int, int, float, float, float]] = []
        self.cycle_times: list[float] = []
        self.arm_hits: dict[str, float] = {}
        self._assist_on = False
        self._robot_bodies = {b: b_name for b in range(self.model.nbody)
                              if (b_name := self.model.body(b).name).startswith(("a/", "b/"))}
        self._script = task.routine(self)
        self.done = False
        self._seen_log = 0
        for _ in range(int(0.3 / self.model.opt.timestep)):  # let the scene settle
            mujoco.mj_step(self.model, self.data)

    # --- Helpers used by the task ------------------------------------------------------------
    def say(self, msg: str) -> None:
        self.status = msg
        line = f"t={self.data.time:7.2f}s {msg}"
        self.messages.append(line)
        if self.verbose:
            print(line, flush=True)

    def grasp_assist(self, body: str, touch: str | None = None, timeout: float = 0.8):
        """Weld `body` to arm A's hand once both fingers touch it, or the part `touch` of the plant they
        close on (repeatable demo grasps)."""
        b = self.model.body(body).id
        t = {self.model.body(touch or body).id}
        for _ in range(int(timeout / CONTROL_DT)):
            if self.arm_a.contacts(t)[0] == 2:
                break
            yield
        n = self.arm_a.contacts(t)[0]
        if n < 2:
            self.say(f"warning: only {n} finger(s) on {body}")
        if self.assist and n:
            constraints.activate_weld(self.model, self.data, "grasp_a", body2=b)
            self._assist_on = True

    def release_assist(self) -> None:
        if self._assist_on:
            constraints.deactivate(self.model, self.data, "grasp_a")
            self._assist_on = False

    # --- Stepping ---------------------------------------------------------------------------
    def _watch_arms(self) -> None:
        """Log contacts of the arms with anything but their work (root ball, trunk, top soil, each other's
        fingers excepted): there should be none."""
        m, d = self.model, self.data
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            arm_g, other = (c.geom1, c.geom2) if b1 in self._robot_bodies else (c.geom2, c.geom1) \
                if b2 in self._robot_bodies else (None, None)
            if arm_g is None:
                continue
            on = m.body(m.geom_bodyid[other]).name
            gn = m.geom(other).name
            if on.startswith("plant_") or "/ring_" in gn:
                continue                       # root ball, trunk, top soil: the job
            key = f"{m.geom(arm_g).name or m.body(m.geom_bodyid[arm_g]).name} / {gn or on}"
            self.arm_hits[key] = min(self.arm_hits.get(key, 0.0), float(c.dist))

    def step(self) -> None:
        """Advance one control tick (CONTROL_DT)."""
        if not self.done:
            try:
                next(self._script)
            except StopIteration:
                self.done = True
        self.arm_a.apply()
        self.arm_b.apply()
        self.line.step()
        for _ in range(self.n_sub):
            self.soil.apply()
            mujoco.mj_step(self.model, self.data)
        self.soil.tamp(CONTROL_DT)
        self._watch_arms()
        log = self.line.log
        while self._seen_log < len(log):
            if self.verbose:
                print(log[self._seen_log], flush=True)
            self.messages.append(log[self._seen_log])
            self._seen_log += 1

    # --- Report --------------------------------------------------------------------------------
    def report(self) -> dict:
        m, d = self.model, self.data
        out = {"plants": []}
        for p, k in self.potted:
            pot, plant = m.body(f"pot_{k}").id, m.body(f"plant_{p}").id
            Rp = d.xmat[pot].reshape(3, 3)
            rel = Rp.T @ (d.xpos[plant] - d.xpos[pot])
            tilt = np.degrees(np.arccos(np.clip((Rp.T @ d.xmat[plant].reshape(3, 3)[:, 2])[2], -1, 1)))
            pot_tilt = np.degrees(np.arccos(np.clip(Rp[2, 2], -1, 1)))
            out["plants"].append({
                "plant": p, "pot": k,
                "off_axis_mm": float(np.hypot(rel[0], rel[1]) * 1000),
                "above_floor_mm": float((rel[2] - L.HOLE_FLOOR_Z) * 1000),
                "tilt_deg": float(tilt),
                "firmed_mm": float(self.soil.comp[k].mean() * 1000),
                "firmed_min_mm": float(self.soil.comp[k].min() * 1000),
                "pot_xy": d.xpos[pot][:2].round(3).tolist(),
                "pot_tilt_deg": float(pot_tilt),
                "mulched": float(self.line.mulch[k]),
                "state": self.line.state[k],
            })
        out["push"] = self.push_report
        out["cycle_times"] = self.cycle_times
        out["arm_hits"] = self.arm_hits
        return out
