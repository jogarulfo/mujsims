"""Cell simulation: physics stepping, machine, robot, vacuum cup and stack light."""

from __future__ import annotations

import mujoco
import numpy as np

from claret_sim import constraints, task
from claret_sim import layout as L
from claret_sim.arm import Arm
from claret_sim.machine import Machine
from claret_sim.planner import Planner, plan_carton
from claret_sim.scene import ARM, LAMPS, build_model, n_parts

CONTROL_DT = 0.01


class VacuumCup:
    """Retractable suction cup on the gripper. With vacuum on, it holds the sheet it touches."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, n_sheets: int):
        self.m, self.d = model, data
        self.act = model.actuator("cup/cylinder").id
        self.q = model.jnt_qposadr[model.joint("cup_slide").id]
        self.lip = model.geom("cup_lip").id
        self.sheets = {model.body(f"sheet_{k}").id: k for k in range(n_sheets)}
        self.stroke = L.CUP_EXTENDED - L.CUP_RETRACTED
        self.vacuum = False
        self.held: int | None = None     # sheet index held by suction

    def extend(self) -> None:
        self.d.ctrl[self.act] = self.stroke

    def retract(self) -> None:
        self.d.ctrl[self.act] = 0.0

    def extended(self) -> bool:
        return self.d.qpos[self.q] > self.stroke - 0.002

    def touching(self) -> int | None:
        """The (topmost) sheet the cup lip is in contact with."""
        hit = []
        for c in self.d.contact[: self.d.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g == self.lip:
                    b = self.m.geom_bodyid[other]
                    if b in self.sheets:
                        hit.append((self.d.xpos[b][2], self.sheets[b]))
        return max(hit)[1] if hit else None

    def step(self) -> None:
        if self.vacuum and self.held is None:
            k = self.touching()
            if k is not None:  # the cup seals on the sheet
                constraints.activate_weld(self.m, self.d, "vacuum", body2=self.m.body(f"sheet_{k}").id)
                self.held = k
        elif not self.vacuum and self.held is not None:
            constraints.deactivate(self.m, self.d, "vacuum")
            self.held = None


class CellSim:
    def __init__(self, sheets: bool = True, layers: int = L.N_LAYERS, cycle: float = L.MACHINE_CYCLE,
                 assist: bool = True, verbose: bool = True, seed: int | None = None):
        self.model, self.spec = build_model(layers)
        self.data = mujoco.MjData(self.model)
        self.sheets, self.layers, self.assist, self.verbose = sheets, layers, assist, verbose
        self.n_sub = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.arm = Arm(self.model, self.data, ARM, CONTROL_DT, site="grip_site")
        self.arm.set_home()
        self.machine = Machine(self.model, self.data, n_parts(layers), cycle=cycle, seed=seed)
        self.cup = VacuumCup(self.model, self.data, L.N_SHEETS)
        mujoco.mj_forward(self.model, self.data)
        # Hand pose for every carton slot, collision-checked (about 2 s).
        self.planner = Planner(self.model, self.arm)
        self.plans = plan_carton(self.planner, self.data.qpos.copy(), self.machine.parts, layers, sheets,
                                 task.TRANSIT_Z)
        bad = [k for k, p in self.plans.items() if not p.ok]
        if bad and verbose:
            print(f"planner: no collision-free hand pose for {len(bad)} slots, e.g. {bad[0]}: {self.plans[bad[0]].hits[:2]}")
        self.push_log: list[tuple[float, float]] = []   # (pushed distance, distance to the pocket)
        self.placed: list[tuple[int, tuple[int, int], int]] = []   # (layer, slot, part body) in packing order
        self.messages: list[str] = []
        self.status = ""
        self.phase = "start"             # what the robot is doing (drives the video camera)
        self.packed = 0                  # parts in the carton
        self.sheets_laid = 0
        self.waiting = False             # robot waiting for a part (stack light amber)
        self._assist = False
        self._lamps = {k: self.model.geom(f"lamp_{k}").id for k in LAMPS}
        self._script = task.routine(self)
        self.done = False
        self._log_seen = 0
        for _ in range(int(0.3 / self.model.opt.timestep)):  # let the scene settle
            mujoco.mj_step(self.model, self.data)

    # --- Helpers used by the task ------------------------------------------------------------
    def say(self, msg: str) -> None:
        self.status = msg
        line = f"t={self.data.time:6.2f}s {msg}"
        self.messages.append(line)
        if self.verbose:
            print(line, flush=True)

    def grasp_assist(self, body: int, timeout: float = 0.6):
        """Weld the part to the hand once both jaws touch it (robust, repeatable demo grasps).
        Returns False if the jaws do not touch the part at all."""
        for _ in range(int(timeout / CONTROL_DT)):
            if self.arm.pad_contacts(body) == 2:
                break
            yield
        n = self.arm.pad_contacts(body)
        if n == 0:
            return False                 # the jaws do not touch the part
        if self.assist:
            constraints.activate_weld(self.model, self.data, "grasp", body2=body)
            self._assist = True
        return True

    def release_assist(self) -> None:
        if self._assist:
            constraints.deactivate(self.model, self.data, "grasp")
            self._assist = False

    def body_z(self, name: str) -> float:
        return float(self.data.xpos[self.model.body(name).id][2])

    # --- Stepping ---------------------------------------------------------------------------
    def _stack_light(self) -> None:
        if self.done:
            on = {"blue": int(self.data.time * 2) % 2 == 0}   # carton full: blinking blue
        elif self.waiting:
            on = {"amber": True}
        else:
            on = {"green": True}
        for k, g in self._lamps.items():
            self.model.geom_rgba[g] = LAMPS[k] if on.get(k) else [0.25, 0.25, 0.25, 1]

    def step(self) -> None:
        """Advance one control tick (CONTROL_DT)."""
        if not self.done:
            try:
                next(self._script)
            except StopIteration:
                self.done = True
        self.arm.apply()
        self.machine.step(CONTROL_DT)
        self.cup.step()
        self._stack_light()
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        log = self.machine.log
        while self._log_seen < len(log):
            if self.verbose:
                print(log[self._log_seen], flush=True)
            self.messages.append(log[self._log_seen])
            self._log_seen += 1

    def carton_count(self) -> str:
        per_layer = L.GRID[0] * L.GRID[1]
        return f"{self.packed}/{per_layer * self.layers}"

    def draw(self, scn: mujoco.MjvScene) -> None:
        """Extra visuals: a red beam on the "tray full" light barrier while the machine holds."""
        if not self.machine.chute_full or scn.ngeom >= scn.maxgeom:
            return
        y = L.CHUTE_FULL_Y
        z = L.chute_z(y) + 0.03
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1.0, 0.1, 0.1, 0.8], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015,
                             np.array([-L.CHUTE_W / 2, y, z]), np.array([L.CHUTE_W / 2, y, z]))
        scn.ngeom += 1

    def packing_report(self) -> dict:
        """How tight the carton is: gaps between neighbours (row and column) and to the walls."""
        d = self.data
        pos = {(lay, slot): d.xpos[b] for lay, slot, b in self.placed}
        gaps, offsets, walls = [], [], []
        x_far = L.BOX_CENTER[0] + L.BOX_IN[0] / 2
        y_left = L.BOX_CENTER[1] + L.BOX_IN[1] / 2
        for (lay, (r, c)), p in pos.items():
            offsets.append(float(np.linalg.norm(p[:2] - L.slot_xy(r, c))))
            for dr, dc in ((1, 0), (0, 1)):
                q = pos.get((lay, (r + dr, c + dc)))
                if q is not None:
                    gaps.append(float(np.linalg.norm(p[:2] - q[:2])) - L.PITCH)
            if r == 0:
                walls.append(x_far - p[0] - L.PITCH / 2)
            if c == 0:
                walls.append(y_left - p[1] - L.PITCH / 2)
        tilt = [float(np.degrees(np.arccos(np.clip(d.xmat[b][8], -1, 1)))) for _, _, b in self.placed]
        return {"gaps": np.array(gaps), "walls": np.array(walls), "offsets": np.array(offsets), "tilt": np.array(tilt)}
