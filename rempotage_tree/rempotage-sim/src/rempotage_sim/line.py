"""Line controller: potting wheel and its stations, outfeed conveyor, row former, sweep, pallet.

Wheel cycle (the PLC's sequence). At every stop, the stations work in parallel:

  S0 denester  if pots are left in the stack and the nest is empty: the fingers open, the bottom
               pot drops 3 cm into the nest, the fingers close and the stack comes down one step
  S1 filler    the plough pushes soil off the soil table into an empty pot (the soil level rises,
               the pot gets heavier)
  S2 dibber    a filled pot gets its planting hole (the dibber goes down to the hole floor and back)
  S3 planting  a pot with its hole is offered to the robots ("pot ready"); the line waits for the
               robots' "planting done"
  S6 ejector   a planted pot is pushed off the wheel onto the outfeed conveyor

Then the wheel indexes one station (30 deg, counterclockwise: 12 nests). The cycle ends when the denester is
empty and no pot is left on the wheel.

Outfeed: the conveyor (powered rollers with slip clutches) runs the pots under the mulcher, whose
side spout drops wood chips on the soil while a pot rolls by (its sensor starts and stops the
feeder), then to the row stop, where they queue. When the third pot has stood in front of the row sensor for a second, the conveyor stops,
the sweep pushes the row across onto the pallet (pushing the previous row ahead of it) and comes
back, and the conveyor starts again. Two rows of three fill the pallet.
"""

from __future__ import annotations

import mujoco
import numpy as np

from rempotage_sim import constraints
from rempotage_sim import layout as L
from rempotage_sim.arm import min_jerk
from rempotage_sim.scene import HIDDEN_GROUP, SHOWN_GROUP, dibber_stroke, pot_collision_bits, set_collision
from rempotage_sim.soil import Soil


class Line:
    ROW_SETTLE = 1.0

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, soil: Soil, dt: float):
        self.m, self.d, self.soil, self.dt = model, data, soil, dt
        self.wheel_act = model.actuator("wheel").id
        self.wheel_q = model.jnt_qposadr[model.joint("wheel").id]
        self.dibber_act = model.actuator("dibber").id
        self.dibber_q = model.jnt_qposadr[model.joint("dibber").id]
        self.eject_act = model.actuator("ejector").id
        self.eject_q = model.jnt_qposadr[model.joint("ejector").id]
        self.sweep_act = model.actuator("sweep").id
        self.sweep_q = model.jnt_qposadr[model.joint("sweep").id]
        self.finger_act = [model.actuator(f"denester/finger_{j}").id for j in range(2)]
        self.conv_act = np.array([a for a in range(model.nu) if model.actuator(a).name.startswith("conveyor/")])
        self.stream = model.geom("soil_stream").id
        self.row_eye = model.site("row_eye").id
        self.chip_stream = model.geom("chip_stream").id
        self.stack_mocap = model.body_mocapid[model.body("pot_stack").id]   # the pots above the run's pots
        self.chips = [[model.geom(f"pot_{k}/chip_{i}").id for i in range(L.N_CHIPS)] for k in range(L.N_POTS)]
        self.mulch = np.zeros(L.N_POTS)                       # share of each pot's chips poured
        self.pot_body = [model.body(f"pot_{k}").id for k in range(L.N_POTS)]
        self.pot_geoms = {k: [(g, bits) for g in range(model.ngeom)
                              if model.geom_bodyid[g] == self.pot_body[k]
                              and (bits := pot_collision_bits(model.geom(g).name)) is not None]
                          for k in range(L.N_POTS)}
        self.nests: list[int | None] = [None] * L.N_STATIONS   # pot in the nest at each station
        self.stack = list(range(L.N_POTS))                    # pots left in the denester, bottom first
        self.state = ["stacked"] * L.N_POTS
        self.wheel_cmd = 0.0
        self.dibber_cmd = self.eject_cmd = self.sweep_cmd = 0.0
        self.finger_cmd = 0.0                                 # 0 = closed (holding the stack)
        self.conveyor_on = True
        self.pot_ready: int | None = None                     # pot offered to the robots at S3
        self._planted_ack: set[int] = set()
        self.rows = 0
        self.on_pallet: list[int] = []
        self.indexes = 0
        self.log: list[str] = []
        self.status = "starting"
        self._seq = self._wheel_cycle()
        self._rows_seq = self._row_former()
        self.done = False
        self.pallet_full = False

    # --- Setup --------------------------------------------------------------------------------
    def reset(self) -> None:
        """Hang the pots in the denester stack (after the first forward pass)."""
        for k in self.stack:
            constraints.activate_weld(self.m, self.d, f"denest_{k}", body2=self.pot_body[k])

    # --- Interface for the robots' task -------------------------------------------------------
    def planting_done(self, pot: int) -> None:
        self._planted_ack.add(pot)

    # --- Helpers ---------------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.status = msg
        self.log.append(f"t={self.d.time:6.2f}s line: {msg}")

    def _wait(self, t: float):
        for _ in range(max(1, round(t / self.dt))):
            yield

    def _ramp(self, attr: str, target: float, t: float):
        v0 = getattr(self, attr)
        n = max(1, round(t / self.dt))
        for i in range(1, n + 1):
            setattr(self, attr, v0 + min_jerk(i / n) * (target - v0))
            yield

    @staticmethod
    def _parallel(*gens):
        live = [g for g in gens if g is not None]
        while live:
            nxt = []
            for g in live:
                try:
                    next(g)
                    nxt.append(g)
                except StopIteration:
                    pass
            live = nxt
            if live:
                yield

    # --- Stations ----------------------------------------------------------------------------------
    def _denest(self):
        k = self.stack.pop(0)
        self._say(f"denester releases pot {k + 1} ({len(self.stack)} left in the stack)")
        yield from self._ramp("finger_cmd", 0.018, 0.25)
        constraints.deactivate(self.m, self.d, f"denest_{k}")
        for g, (ct, ca) in self.pot_geoms[k]:
            set_collision(self.m, g, ct, ca)
        yield from self._wait(0.5)
        self.nests[L.S_DENEST] = k
        self.state[k] = "empty"
        yield from self._ramp("finger_cmd", 0.0, 0.25)
        # The rest of the stack comes down one step onto the fingers.
        eqs = [self.m.equality(f"denest_{j}").id for j in self.stack]
        z0 = [float(self.m.eq_data[e, 5]) for e in eqs]
        zs = float(self.d.mocap_pos[self.stack_mocap, 2])
        n = round(0.6 / self.dt)
        for i in range(1, n + 1):
            for e, z in zip(eqs, z0, strict=True):
                self.m.eq_data[e, 5] = z - L.NEST_PITCH * min_jerk(i / n)
            self.d.mocap_pos[self.stack_mocap, 2] = zs - L.NEST_PITCH * min_jerk(i / n)
            yield

    def _fill(self, k: int):
        self._say(f"filler: soil into pot {k + 1}")
        self.m.geom_group[self.stream] = SHOWN_GROUP
        n = round(L.FILL_TIME / self.dt)
        for i in range(1, n + 1):
            self.soil.set_fill(k, i / n)
            yield
        self.m.geom_group[self.stream] = HIDDEN_GROUP
        self.state[k] = "filled"

    def _dibble(self, k: int):
        self._say(f"dibber: planting hole in pot {k + 1}")
        yield from self._ramp("dibber_cmd", dibber_stroke(), 0.9)
        yield from self._wait(0.25)
        self.soil.dibble(k)
        yield from self._wait(0.25)
        yield from self._ramp("dibber_cmd", 0.0, 0.8)
        self.state[k] = "dibbled"

    def _plant(self, k: int):
        self.pot_ready = k
        self._say(f"pot {k + 1} at the planting station: robots")
        while k not in self._planted_ack:
            yield
        self.pot_ready = None
        self.state[k] = "planted"

    def _eject(self, k: int):
        self._say(f"ejector: pot {k + 1} out onto the conveyor")
        yield from self._ramp("eject_cmd", L.EJECT_STROKE, 1.2)
        yield from self._wait(0.2)
        self.nests[L.S_EJECT] = None
        self.state[k] = "conveyor"
        yield from self._ramp("eject_cmd", 0.0, 0.9)

    def _index(self):
        self.indexes += 1
        self._say("wheel indexes one station")
        yield from self._ramp("wheel_cmd", self.wheel_cmd + L.STATION_PITCH, L.INDEX_TIME)
        self.nests = [self.nests[-1]] + self.nests[:-1]
        if self.nests[L.S_DENEST] is not None:   # cannot happen: the nests after the ejector come back empty
            self._say(f"warning: pot {self.nests[L.S_DENEST] + 1} came back to the denester")

    def _wheel_cycle(self):
        yield from self._wait(0.5)
        while True:
            jobs = []
            if self.stack and self.nests[L.S_DENEST] is None:
                jobs.append(self._denest())
            if (k := self.nests[L.S_FILL]) is not None and self.state[k] == "empty":
                jobs.append(self._fill(k))
            if (k := self.nests[L.S_DIBBLE]) is not None and self.state[k] == "filled":
                jobs.append(self._dibble(k))
            if (k := self.nests[L.PLANT_STATION]) is not None and self.state[k] == "dibbled":
                jobs.append(self._plant(k))
            if (k := self.nests[L.S_EJECT]) is not None:
                jobs.append(self._eject(k))
            yield from self._parallel(*jobs)
            if not self.stack and all(n is None for n in self.nests):
                break
            yield from self._index()
        self._say("wheel empty: last pot out")
        self.done = True

    # --- Outfeed -------------------------------------------------------------------------------------
    def row_pot(self) -> int | None:
        """Pot standing still in front of the row sensor (third place of the row), if any."""
        geomid = np.zeros(1, np.int32)
        groups = np.array([0, 0, 0, 1, 0, 0], np.uint8)   # pot collision geoms only
        dist = mujoco.mj_ray(self.m, self.d, self.d.site_xpos[self.row_eye], np.array([0.0, 1.0, 0.0]), groups, 1,
                             -1, geomid)
        if not 0 <= dist < L.LANE_W:
            return None
        b = int(self.m.geom_bodyid[geomid[0]])
        if b not in self.pot_body:
            return None
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        return b if np.linalg.norm(v[3:]) < 0.005 else None

    def _row_former(self):
        while self.rows < L.ROWS:
            t = 0.0
            while t < self.ROW_SETTLE:
                t = t + self.dt if self.row_pot() is not None else 0.0
                yield
            row = [k for k in range(L.N_POTS) if self.state[k] == "conveyor"]
            self._say(f"row {self.rows + 1} complete: sweep onto the pallet")
            self.conveyor_on = False
            yield from self._ramp("sweep_cmd", L.SWEEP_STROKE, 2.0)
            yield from self._wait(0.2)
            yield from self._ramp("sweep_cmd", 0.0, 1.5)
            for k in row:
                self.state[k] = "pallet"
            self.on_pallet += row
            self.rows += 1
            self.conveyor_on = True
        self._say("pallet full")
        self.pallet_full = True

    # --- Stepping --------------------------------------------------------------------------------------
    def step(self) -> None:
        for seq_name in ("_seq", "_rows_seq"):
            seq = getattr(self, seq_name)
            if seq is not None:
                try:
                    next(seq)
                except StopIteration:
                    setattr(self, seq_name, None)
        d = self.d
        d.ctrl[self.wheel_act] = self.wheel_cmd
        d.ctrl[self.dibber_act] = self.dibber_cmd
        d.ctrl[self.eject_act] = self.eject_cmd
        d.ctrl[self.sweep_act] = self.sweep_cmd
        d.ctrl[self.finger_act] = self.finger_cmd
        d.ctrl[self.conv_act] = (L.CONV_SPEED / L.ROLLER_R) if self.conveyor_on else 0.0
        self._mulcher()

    def _mulcher(self) -> None:
        """Pour chips on a pot while it passes under the spout (the share follows its position)."""
        m, d = self.m, self.d
        pouring = False
        for k in range(L.N_POTS):
            if self.state[k] != "conveyor" or self.mulch[k] >= 1.0:
                continue
            x, y = d.xpos[self.pot_body[k]][:2]
            f = min((x - (L.CHIP_X - L.CHIP_ZONE)) / (2 * L.CHIP_ZONE), 1.0)
            if f <= self.mulch[k] or abs(y - L.CONV_Y) > L.LANE_W / 2:
                pouring = pouring or 0.0 < self.mulch[k] < 1.0
                continue
            if self.mulch[k] == 0.0:
                self._say(f"mulcher: wood chips on pot {k + 1}")
            n0, n1 = int(self.mulch[k] * L.N_CHIPS), int(f * L.N_CHIPS)
            for i in range(n0, n1):
                g = self.chips[k][i]
                if i % 4 != 3:   # on the soil ring: rest on the sector as firmed
                    gx, gy = m.geom_pos[g][:2]
                    j = int(np.arctan2(gy, gx) % (2 * np.pi) // (2 * np.pi / L.TAMP_SECTORS)) % L.TAMP_SECTORS
                    m.geom_pos[g][2] = L.SOIL_Z + 0.001 - self.soil.comp[k, j]
                m.geom_group[g] = SHOWN_GROUP
            self.mulch[k] = f
            pouring = pouring or f < 1.0
        m.geom_group[self.chip_stream] = SHOWN_GROUP if pouring else HIDDEN_GROUP

    def pot_at(self, station: int) -> int | None:
        return self.nests[station]
