"""Production machine and outlet tray: parts come out at random times and land anywhere.

The machine is modelled from the outside: its parts wait, hidden and weightless, inside
the cabinet (scene.park_pos). When a part is finished (random intervals, MACHINE_CYCLE on
average), the machine drops it at a random spot on its flat outlet plate behind the rubber
curtain, and the ejector pushes it over the lip. The part slides down the wide tray and
stops wherever it stops: against the end stop, or against parts already lying there.

Sensors:
  tray camera  reports the parts lying still and upright on the tray (where the robot can
               take them). A real cell would use a camera over the tray; here it reads the
               simulator's state.
  tray full    light barrier near the machine, plus a part count: the machine holds the
               next part while the tray is full (the robot has fallen behind).
"""

from __future__ import annotations

import mujoco
import numpy as np

from claret_sim import layout as L

EJECT_DELAY = 0.4      # s between the part landing on the outlet plate and the ejector stroke
EJECT_HOLD = 0.5       # s the ejector stays out
REST_TIME = 0.3        # s a part must lie still before the camera reports it


class Machine:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, n_parts: int, cycle: float = L.MACHINE_CYCLE,
                 seed: int | None = None):
        self.m, self.d = model, data
        self.cycle = cycle
        self.parts = [model.body(f"part_{k}").id for k in range(n_parts)]
        self.qadr = [model.jnt_qposadr[model.body_jntadr[b]] for b in self.parts]
        self.dadr = [model.jnt_dofadr[model.body_jntadr[b]] for b in self.parts]
        self.ej_act = model.actuator("ejector/cylinder").id
        self.ej_q = model.jnt_qposadr[model.joint("ejector").id]
        self.rng = np.random.default_rng(seed)
        self.made = 0                    # parts produced so far
        self.running = True
        self.t_cycle = 0.0
        self.next_in = 1.5               # first part comes out soon after start
        self.phase = "idle"              # idle -> drop -> eject -> retract
        self.t_phase = 0.0
        self.log: list[str] = []
        self.chute_full = False
        self.waiting_full = False
        self.packed: set[int] = set()    # parts taken by the robot (no longer on the tray)
        self._rest = np.zeros(n_parts)   # how long each part has been lying still
        self.lying: set[int] = set()     # parts reported fallen over (an operator stands them up)

    # --- Sensors ------------------------------------------------------------------------------
    def _speed(self, k: int) -> float:
        return float(np.linalg.norm(self.d.qvel[self.dadr[k]:self.dadr[k] + 6] * [1, 1, 1, 0.03, 0.03, 0.03]))

    def on_tray(self) -> list[int]:
        """Indices of the parts made and not yet taken by the robot."""
        return [k for k in range(self.made) if self.parts[k] not in self.packed]

    def _upright(self, k: int) -> bool:
        a = self.d.xmat[self.parts[k]].reshape(3, 3)[:, 2]
        n = np.array([0, -np.sin(L.CHUTE_SLOPE), np.cos(L.CHUTE_SLOPE)])
        return float(a @ n) > 0.97

    def pickable(self) -> list[int]:
        """Parts lying still and upright on the tray, nearest the end stop first."""
        out = []
        for k in self.on_tray():
            p = self.d.xpos[self.parts[k]]
            if p[1] < L.MACHINE_FRONT_Y - 0.03 and self._rest[k] >= REST_TIME and self._upright(k):
                out.append(k)
        return sorted(out, key=lambda k: self.d.xpos[self.parts[k]][1] + 0.3 * abs(self.d.xpos[self.parts[k]][0]))

    def _update_sensors(self, dt: float) -> None:
        full = False
        for k in self.on_tray():
            still = self._speed(k) < 0.01
            self._rest[k] = self._rest[k] + dt if still else 0.0
            p = self.d.xpos[self.parts[k]]
            if abs(p[1] - L.CHUTE_FULL_Y) < L.PART_R and still:
                full = True
            if still and self._rest[k] > 1.0 and not self._upright(k) and k not in self.lying:
                self.lying.add(k)
                self.log.append(f"t={self.d.time:6.2f}s tray camera: part {k + 1} lies on its side")
        self.chute_full = full or self.queue_length() >= L.TRAY_MAX

    def queue_length(self) -> int:
        """Parts on the tray (out of the machine, not yet picked)."""
        return sum(1 for k in self.on_tray() if self.d.xpos[self.parts[k]][1] < L.MACHINE_FRONT_Y + 0.05)

    # --- Production -----------------------------------------------------------------------------
    def _drop_part(self) -> None:
        k = self.made
        b = self.parts[k]
        yaw = self.rng.uniform(0, 2 * np.pi)
        a = self.qadr[k]
        x = L.CHUTE_X + self.rng.uniform(-L.SPAWN_X, L.SPAWN_X)
        y = L.SPAWN_Y + self.rng.uniform(-0.03, 0.03)
        self.d.qpos[a:a + 3] = [x, y, L.OUTLET_Z + 0.001]
        self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qvel[self.dadr[k]:self.dadr[k] + 6] = 0
        self.m.body_gravcomp[b] = 0.0
        self.made += 1
        self.log.append(f"t={self.d.time:6.2f}s machine: part {self.made}/{len(self.parts)} finished")

    def step(self, dt: float) -> None:
        self._update_sensors(dt)
        self.t_cycle += dt
        self.t_phase += dt
        if self.phase == "idle":
            if self.made >= len(self.parts) or not self.running:
                return
            if self.t_cycle < self.next_in:
                return
            if self.chute_full:
                if not self.waiting_full:
                    self.log.append(f"t={self.d.time:6.2f}s machine: tray full, holding the next part")
                    self.waiting_full = True
                return
            self.waiting_full = False
            self._drop_part()
            self.t_cycle = 0.0
            self.next_in = self.cycle * self.rng.uniform(1 - L.CYCLE_SPREAD, 1 + L.CYCLE_SPREAD)
            self.phase, self.t_phase = "drop", 0.0
        elif self.phase == "drop" and self.t_phase >= EJECT_DELAY:
            self.d.ctrl[self.ej_act] = L.EJECT_STROKE
            self.phase, self.t_phase = "eject", 0.0
        elif self.phase == "eject" and self.t_phase >= EJECT_HOLD:
            self.d.ctrl[self.ej_act] = 0.0
            self.phase, self.t_phase = "retract", 0.0
        elif self.phase == "retract" and self.d.qpos[self.ej_q] < 0.003:
            self.phase, self.t_phase = "idle", 0.0

    def taken(self, k: int) -> None:
        """The robot has picked part k off the tray."""
        self.packed.add(self.parts[k])

    def done(self) -> bool:
        return self.made >= len(self.parts)
