"""Builds the MuJoCo model of the repotting cell with MjSpec.

Cell contents: a potting wheel (12 nests) indexing around a fixed hub and soil table (pot denester,
soil plough, dibber, planting station, ejector), a trolley with a 4 x 6 tray of young trees in root
balls and a camera over it, two I2RT YAM arms on pedestals
(prefixes ``a/`` and ``b/``), an outfeed roller conveyor with a wood-chip mulcher, ending in a
row former, a sweep and a pallet.

Collision bits (contype / conaffinity):
  1 / 1   everything solid (pots, wheel, conveyor, pallet, trunks, ...)
  4 / 4   root balls: they touch the robots' fingers, the cart tray, the floor of the planting
          hole and each other, not the pot walls or the soil (soil.py pushes back on them)
  2 / 0   top soil ring of a pot (collision boxes): only the robots' fingers touch it (tamping)
  4 / 0   floor of the planting hole: only the root balls touch it
  Robots: contype 1, conaffinity 1 | 2 | 4.

Equality constraints switched at runtime (inactive in the model):
  grasp_a       weld   a/link_6 <-> plant      grasp assist while arm A carries a plant
Mocap body: pot_stack (the empty pots above the run's pots in the denester, stepping down with them).
  denest_<k>    weld   world    <-> pot_<k>    the denester holding pot k in its stack
  planted_<k>   weld   pot_<k>  <-> plant      the firmed soil holding the root ball
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from rempotage_sim import layout as L
from rempotage_sim import meshgen

ASSETS = Path(__file__).parent / "assets"
YAM_XML = ASSETS / "yam" / "yam_linear_4310.xml"

GEOM = mujoco.mjtGeom
JNT = mujoco.mjtJoint
VIS = {"contype": 0, "conaffinity": 0, "group": 2, "density": 0}
HIDDEN_GROUP = 5          # soil pieces not there yet are moved to this (not rendered) group
SHOWN_GROUP = 2

COLORS = {
    "frame": [0.93, 0.42, 0.10, 1],        # orange machine frame
    "machine": [0.82, 0.83, 0.84, 1],      # light grey wheel and hub
    "machine_dark": [0.30, 0.31, 0.33, 1],
    "yellow": [0.98, 0.78, 0.10, 1],
    "steel": [0.70, 0.72, 0.75, 1],
    "pneumatic": [0.80, 0.82, 0.85, 1],
    "pot": [0.10, 0.10, 0.11, 1],
    "soil": [0.27, 0.18, 0.11, 1],
    "rootball": [0.22, 0.15, 0.09, 1],
    "root": [0.80, 0.72, 0.55, 1],
    "bark": [0.42, 0.30, 0.20, 1],
    "leaf": [0.24, 0.55, 0.20, 1],
    "leaf2": [0.34, 0.66, 0.24, 1],
    "cart": [0.25, 0.45, 0.30, 1],
    "cart_tray": [0.15, 0.15, 0.16, 1],
    "rubber": [0.08, 0.08, 0.08, 1],
    "roller": [0.72, 0.74, 0.76, 1],
    "pallet": [0.72, 0.58, 0.38, 1],
    "pedestal": [0.35, 0.37, 0.40, 1],
    "motor": [0.15, 0.30, 0.55, 1],
    "hopper": [0.55, 0.58, 0.62, 1],
    "chips": [0.80, 0.58, 0.32, 1],        # wood-chip mulch
    "chips_dark": [0.62, 0.40, 0.20, 1],
    "chips_light": [0.90, 0.72, 0.47, 1],
    "bigbag": [0.85, 0.86, 0.80, 1],
    "crate": [0.93, 0.93, 0.92, 1],
}
CHIP_MATERIALS = ("chips", "chips_dark", "chips_light")


def yaw_quat(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def axis_angle_quat(axis, angle: float) -> list[float]:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]


def quat_mul(a, b) -> list[float]:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return list(out)


def look_at(pos, target) -> list[float]:
    """Camera xyaxes looking from pos at target with world z up."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, [0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x, *y]


def box(body, size, pos, material=None, quat=None, name=None, collide=False, **kw):
    args = {"type": GEOM.mjGEOM_BOX, "size": list(size), "pos": list(pos)}
    if quat is not None:
        args["quat"] = list(quat)
    if material:
        args["material"] = material
    if name:
        args["name"] = name
    if not collide:
        args.update(VIS)
    args.update(kw)
    return body.add_geom(**args)


def rod(body, a, b, r, material, collide=False, name=None, kind=GEOM.mjGEOM_CYLINDER):
    args = {"type": kind, "size": [r, 0, 0], "fromto": [*a, *b], "material": material}
    if name:
        args["name"] = name
    if not collide:
        args.update(VIS)
    return body.add_geom(**args)


# --- Pieces ------------------------------------------------------------------------


def _add_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.45, 0.50, 0.56], rgb2=[0.10, 0.11, 0.13], width=512, height=3072)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.30, 0.30, 0.31], rgb2=[0.25, 0.25, 0.26], markrgb=[0.2, 0.2, 0.2],
                     width=300, height=300)
    spec.add_material(name="floor", textures=["", "floor"], texuniform=True, texrepeat=[4, 4],
                      reflectance=0.05)
    for name, rgba in COLORS.items():
        shiny = name in ("steel", "pneumatic", "roller")
        spec.add_material(name=name, rgba=rgba, specular=0.5 if shiny else 0.2,
                          shininess=0.6 if shiny else 0.3)
    names = ["pot", "sqpot", "rootball", "dibber", "wheel", "leaf", "plug"]
    names += [f"soil_{j}" for j in range(len(meshgen.SOIL_LAYERS) - 1)]
    names += [f"soil_{j}_holed" for j in range(1, len(meshgen.SOIL_LAYERS) - 1)]
    names += [f"ring_{j}" for j in range(L.TAMP_SECTORS)]
    for mesh in names:
        spec.add_mesh(name=mesh, file=str(meshgen.MESH_DIR / f"{mesh}.stl"))


def _add_environment(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.4, -0.8, 3.0], dir=[-0.15, 0.25, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=True, diffuse=[0.70, 0.70, 0.68], specular=[0.2, 0.2, 0.2])
    wb.add_light(name="fill", pos=[-1.5, 1.5, 2.2], dir=[0.6, -0.5, -0.7], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=False, diffuse=[0.30, 0.30, 0.33])
    wb.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[4, 4, 0.05], material="floor")

    pot3 = np.r_[L.station_xy(L.PLANT_STATION), L.WHEEL_Z + L.SOIL_Z]
    dr = L.RING_SHIFT
    cams = {
        "overview": ([-1.20, -2.25, 2.10], [0.15, -0.30, 0.80], 58),
        "line": ([2.30 + dr, -2.05, 2.30], [0.45 + dr, 0.05, 0.80], 47),
        "planting": (pot3 + [0.05, -0.85, 0.55], pot3 + [0.0, 0.0, 0.05], 40),
        "planting_close": (pot3 + [-0.10, -0.42, 0.30], pot3 + [0.0, 0.0, 0.0], 40),
        "pot_top": (pot3 + [0.0, -0.10, 0.55], pot3, 40),
        "robots": ([0.0, -2.00 - dr, 1.45], [0.0, -0.45 - dr, 0.85], 50),
        # Behind the two robots (the video's main views): the whole work area, and the planting station.
        "behind": ([0.30, -2.20, 1.75], [-0.30, -0.70, 0.86], 52),
        "behind_close": ([0.06, -1.50, 1.62], [-0.02, -0.55, 0.88], 36),
        "cart": ([-1.75, -0.35 - dr, 1.50], [-0.70, -0.70 - dr, 0.80], 50),
        "wheel_top": ([0.0, -0.30, 3.10], [0.0, 0.0, 0.70], 45),
        "pallet": ([1.10 + dr, -0.75, 1.55], [1.05 + dr, 0.15, 0.70], 50),
        "mulch": ([0.98 + dr, -0.52, 1.22], [L.CHIP_X, 0.02, 0.88], 45),
    }
    for name, (pos, target, fovy) in cams.items():
        wb.add_camera(name=name, pos=list(pos), xyaxes=look_at(pos, target), fovy=fovy)


def _add_wheel(spec: mujoco.MjSpec) -> None:
    """Potting wheel (indexing ring of nests), its drum-shaped base and the central soil table."""
    wb = spec.worldbody
    base = wb.add_body(name="wheel_base")
    z_top = L.WHEEL_Z - L.WHEEL_T
    rod(base, [0, 0, 0.02], [0, 0, z_top - 0.03], L.WHEEL_R - 0.06, "steel")       # drum
    rod(base, [0, 0, 0], [0, 0, 0.02], L.WHEEL_R - 0.04, "machine_dark")          # foot
    for z in (0.18, z_top - 0.06):                                                # drum bands
        rod(base, [0, 0, z], [0, 0, z + 0.012], L.WHEEL_R - 0.055, "machine")
    rod(base, [0, 0, z_top - 0.03], [0, 0, z_top - 0.002], L.WHEEL_R - 0.10, "machine")   # bearing ring
    rod(base, [0, 0, z_top - 0.03], [0, 0, L.WHEEL_Z + 0.05], L.HUB_R, "machine")  # fixed hub
    # Control box on the drum, at the back (start / stop buttons).
    a_cb = np.deg2rad(150.0)
    u_cb = np.array([np.cos(a_cb), np.sin(a_cb)])
    box(base, [0.05, 0.09, 0.07], [*((L.WHEEL_R - 0.03) * u_cb), 0.50], "machine", quat=yaw_quat(a_cb))
    for j, mat in enumerate(("motor", "frame")):
        base.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.012, 0.006],
                      pos=[*((L.WHEEL_R + 0.026) * u_cb + (j - 0.5) * 0.06 * np.array([-u_cb[1], u_cb[0]])), 0.52],
                      quat=quat_mul(yaw_quat(a_cb), axis_angle_quat([0, 1, 0], np.pi / 2)), material=mat, **VIS)

    # Soil table: a large fixed plate over the hub, its rim just inside the pot ring, heaped with soil.
    table = wb.add_body(name="soil_table")
    rod(table, [0, 0, L.WHEEL_Z + 0.05], [0, 0, L.TABLE_Z - 0.015], 0.10, "machine_dark")
    rod(table, [0, 0, L.TABLE_Z - 0.015], [0, 0, L.TABLE_Z], L.TABLE_R, "steel")
    rod(table, [0, 0, L.TABLE_Z], [0, 0, L.TABLE_Z + 0.006], L.TABLE_R - 0.025, "soil")
    rng = np.random.default_rng(3)
    for _ in range(22):                                           # low heap of potting soil
        r = L.TABLE_R * np.sqrt(rng.uniform(0.0, 0.55))
        a = rng.uniform(0, 2 * np.pi)
        h = 0.008 + 0.035 * (1 - r / L.TABLE_R) * rng.uniform(0.5, 1.0)
        table.add_geom(type=GEOM.mjGEOM_ELLIPSOID, size=[rng.uniform(0.10, 0.17), rng.uniform(0.07, 0.12), h],
                       pos=[r * np.cos(a), r * np.sin(a), L.TABLE_Z], quat=yaw_quat(rng.uniform(0, np.pi)),
                       material="soil" if rng.random() < 0.85 else "rootball", **VIS)

    wheel = wb.add_body(name="wheel", pos=[0, 0, L.WHEEL_Z])
    wheel.add_joint(name="wheel", type=JNT.mjJNT_HINGE, axis=[0, 0, 1], damping=30.0, armature=0.8)
    wheel.explicitinertial = True
    m_w = 14.0
    i_z = m_w * (L.WHEEL_R**2 + L.WHEEL_R_IN**2) / 2
    wheel.mass, wheel.ipos = m_w, [0, 0, -L.WHEEL_T / 2]
    wheel.inertia = [i_z / 2, i_z / 2, i_z]
    wheel.add_geom(type=GEOM.mjGEOM_MESH, meshname="wheel", material="machine", **VIS)
    t = 0.006
    h = L.NEST_GUIDE_H
    side = L.POT_R_BOT + L.NEST_CLEAR + t / 2
    for k in range(L.N_STATIONS):
        a = L.station_angle(k)
        c, s = np.cos(a), np.sin(a)
        rad, tan = np.array([c, s]), np.array([-s, c])
        ctr = L.STATION_R * rad
        q = yaw_quat(a)
        # Pad the pot stands on (box: flat multi-point contact), out to the rim for the ejection.
        r0, r1 = L.STATION_R - L.POT_R_BOT - 0.02, L.WHEEL_R - 0.008
        box(wheel, [(r1 - r0) / 2, L.POT_R_BOT + 0.012, 0.005], [*(((r0 + r1) / 2) * rad), -0.005], quat=q,
            collide=True, name=f"nest_{k}_pad", group=3, friction=[0.7, 0.01, 0.001])
        for sgn in (-1, 1):  # side guides (radial strips), and back posts, open towards the outside
            p = ctr + sgn * side * tan
            box(wheel, [0.035, t / 2, h / 2], [*p, h / 2], "yellow", quat=q, collide=True,
                name=f"nest_{k}_side_{'l' if sgn > 0 else 'r'}", friction=[0.4, 0.01, 0.001])
            p = ctr - side * rad + sgn * 0.035 * tan
            box(wheel, [t / 2, 0.012, h / 2], [*p, h / 2], "yellow", quat=q, collide=True,
                name=f"nest_{k}_back_{'l' if sgn > 0 else 'r'}", friction=[0.4, 0.01, 0.001])
        # station flag on the rim (like the reference line)
        box(wheel, [0.004, 0.03, 0.03], [*((L.WHEEL_R - 0.01) * rad + 0.11 * tan), 0.03], "yellow", quat=q)
    act = spec.add_actuator(name="wheel", target="wheel", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=3000.0, kv=300.0)
    act.forcerange = [-400, 400]
    act.forcelimited = True


def _add_stations(spec: mujoco.MjSpec) -> None:
    """Hub mast and gantry carrying the denester (S0), the soil plough (S1) and the dibber (S2);
    the ejector (S6) on the hub."""
    wb = spec.worldbody
    g = wb.add_body(name="gantry")
    top = 1.42
    rod(g, [0, 0, L.TABLE_Z], [0, 0, top], 0.045, "frame")
    for k in (L.S_DENEST, L.S_DIBBLE):  # gantry arms to the denester and the dibber
        xy = L.station_xy(k)
        box(g, [L.STATION_R / 2 + 0.03, 0.025, 0.03], [*(xy / 2), top - 0.03], "frame",
            quat=yaw_quat(L.station_angle(k)))

    # S0 denester: a tall stack of nested pots in a guide cage, release fingers at the bottom.
    xy0 = L.station_xy(L.S_DENEST)
    z_bot = L.WHEEL_Z + L.DENEST_DROP
    cage_top = stack_z(L.N_POTS + L.STACK_EXTRA - 1) + L.POT_H + 0.06
    for j in range(4):
        a = np.pi / 4 + j * np.pi / 2
        p = xy0 + (L.POT_LIP_R + 0.012) * np.array([np.cos(a), np.sin(a)])
        rod(g, [*p, z_bot + L.POT_H - 0.02], [*p, cage_top], 0.006, "steel")
    for z in (top - 0.06, cage_top - 0.03):
        rod(g, [*xy0, z], [*xy0, z + 0.03], L.POT_LIP_R + 0.02, "frame")
    rod(g, [*(xy0 * 1.25), 0.0], [*(xy0 * 1.25), cage_top], 0.03, "frame")      # cage column
    box(g, [0.06, 0.06, 0.01], [*(xy0 * 1.25), 0.01], "machine_dark")
    for z in (top - 0.045, cage_top - 0.015):
        box(g, [0.075, 0.015, 0.015], [*(xy0 * 1.125), z], "frame", quat=yaw_quat(L.station_angle(L.S_DENEST)))
    stack = wb.add_body(name="pot_stack", mocap=True, pos=[*xy0, stack_z(L.N_POTS)])
    for j in range(L.STACK_EXTRA):
        stack.add_geom(type=GEOM.mjGEOM_MESH, meshname="pot", material="pot", pos=[0, 0, j * L.NEST_PITCH],
                       quat=yaw_quat(0.7 * (j + L.N_POTS)), **VIS)
    den = wb.add_body(name="denester")
    for j in range(2):
        a = L.station_angle(L.S_DENEST) + np.pi / 2 + j * np.pi
        u = np.array([np.cos(a), np.sin(a)])
        p = xy0 + (L.POT_LIP_R + 0.03) * u
        box(den, [0.02, 0.015, 0.015], [*p, z_bot + L.POT_H - 0.01], "pneumatic", quat=yaw_quat(a))
        finger = den.add_body(name=f"denester/finger_{j}", pos=[*(xy0 + (L.POT_LIP_R + 0.008) * u), z_bot + L.POT_H - 0.01],
                              quat=yaw_quat(a))
        finger.add_joint(name=f"denester/finger_{j}", type=JNT.mjJNT_SLIDE, axis=[1, 0, 0], range=[0, 0.02],
                         damping=2.0)
        box(finger, [0.008, 0.006, 0.003], [-0.004, 0, 0], "steel")
        finger.explicitinertial = True
        finger.mass, finger.ipos, finger.inertia = 0.02, [0, 0, 0], [1e-6, 1e-6, 1e-6]
        act = spec.add_actuator(name=f"denester/finger_{j}", target=f"denester/finger_{j}",
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=200.0, kv=10.0)

    # S1 soil plough: a curved blade over the soil table, hung from the mast, pushes soil off the
    # table's rim into the pot (the soil stream is shown while filling).
    a1 = L.station_angle(L.S_FILL)
    u1, t1 = np.array([np.cos(a1), np.sin(a1)]), np.array([-np.sin(a1), np.cos(a1)])
    xy1 = L.station_xy(L.S_FILL)
    plough = wb.add_body(name="filler")
    zb = L.TABLE_Z + 0.006
    for j in range(4):   # blade in 4 straight pieces, from the middle of the table out to the rim
        r0, r1 = 0.12 + j * (L.TABLE_R - 0.13) / 4, 0.12 + (j + 1) * (L.TABLE_R - 0.13) / 4
        p0, p1 = r0 * u1 + 0.10 * (1 - j / 4) ** 1.5 * t1, r1 * u1 + 0.10 * (1 - (j + 1) / 4) ** 1.5 * t1
        mid, d = (p0 + p1) / 2, p1 - p0
        box(plough, [np.linalg.norm(d) / 2 + 0.004, 0.004, 0.035], [*mid, zb + 0.035], "frame",
            quat=yaw_quat(np.arctan2(d[1], d[0])))
        plough.add_geom(type=GEOM.mjGEOM_ELLIPSOID, size=[np.linalg.norm(d) / 2 + 0.01, 0.03, 0.03],
                        pos=[*(mid + 0.035 * t1), zb], quat=yaw_quat(np.arctan2(d[1], d[0])), material="soil", **VIS)
    rod(plough, [*(0.10 * u1), L.TABLE_Z + 0.30], [*(0.30 * u1), zb + 0.07], 0.012, "frame")
    rod(plough, [0, 0, L.TABLE_Z + 0.30], [*(0.10 * u1), L.TABLE_Z + 0.30], 0.012, "frame")
    plough.add_geom(name="soil_stream", type=GEOM.mjGEOM_CAPSULE, size=[0.024, 0, 0],
                    fromto=[*((L.TABLE_R - 0.01) * u1), zb + 0.01, *xy1, L.WHEEL_Z + 0.03], material="soil",
                    contype=0, conaffinity=0, group=HIDDEN_GROUP, density=0)

    # S2 dibber: pneumatic cylinder, the rod carries the dibber (the hole's shape).
    xy2 = L.station_xy(L.S_DIBBLE)
    dib = wb.add_body(name="dibber_frame")
    z_rest = L.WHEEL_Z + L.POT_H + 0.07      # dibber tip at rest
    rod(dib, [*xy2, z_rest + 0.30], [*xy2, top - 0.06], 0.03, "pneumatic")
    rod(dib, [*xy2, top - 0.06], [*xy2, top - 0.03], 0.04, "frame")
    tool = wb.add_body(name="dibber", pos=[*xy2, z_rest])
    tool.add_joint(name="dibber", type=JNT.mjJNT_SLIDE, axis=[0, 0, -1], range=[0, dibber_stroke()], damping=50.0)
    tool.add_geom(type=GEOM.mjGEOM_MESH, meshname="dibber", material="steel", **VIS)
    rod(tool, [0, 0, L.HOLE_DEPTH], [0, 0, 0.32], 0.012, "steel")
    tool.explicitinertial = True
    tool.mass, tool.ipos, tool.inertia = 0.8, [0, 0, 0.1], [1e-3, 1e-3, 1e-3]
    act = spec.add_actuator(name="dibber", target="dibber", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=3000.0, kv=200.0)

    # S6 ejector: horizontal cylinder on the hub (under the soil table), plate pushing the pot outwards.
    a5 = L.station_angle(L.S_EJECT)
    u5 = np.array([np.cos(a5), np.sin(a5)])
    ez = L.WHEEL_Z + L.EJECT_Z
    r_e = L.EJECT_REST_R
    ej = wb.add_body(name="ejector_frame")
    rod(ej, [*((r_e - 0.28) * u5), ez + 0.045], [*((r_e - 0.06) * u5), ez + 0.045], 0.03, "pneumatic")
    rod(ej, [*((r_e - 0.06) * u5), ez + 0.045], [*((r_e - 0.05) * u5), ez + 0.045], 0.034, "machine_dark")
    plate = wb.add_body(name="ejector", pos=[*(r_e * u5), ez], quat=yaw_quat(a5))
    plate.add_joint(name="ejector", type=JNT.mjJNT_SLIDE, axis=[1, 0, 0], range=[0, L.EJECT_STROKE], damping=40.0)
    box(plate, [0.005, 0.045, 0.02], [0, 0, 0], "frame", collide=True, name="ejector_plate",
        friction=[0.3, 0.01, 0.001])
    rod(plate, [-0.06, 0, 0.045], [0, 0, 0.045], 0.01, "steel")
    box(plate, [0.006, 0.01, 0.025], [-0.004, 0, 0.025], "frame")
    plate.explicitinertial = True
    plate.mass, plate.ipos, plate.inertia = 0.5, [0, 0, 0], [1e-3, 1e-3, 1e-3]
    act = spec.add_actuator(name="ejector", target="ejector", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=2000.0, kv=150.0)
    act.forcerange = [-120, 120]
    act.forcelimited = True


def dibber_stroke() -> float:
    return (L.WHEEL_Z + L.POT_H + 0.07) - (L.WHEEL_Z + L.HOLE_FLOOR_Z)


def stack_z(k: int) -> float:
    """Pot k's bottom in the denester stack (pot 0 at the bottom, released first)."""
    return L.WHEEL_Z + L.DENEST_DROP + k * L.NEST_PITCH


def _add_pot(spec: mujoco.MjSpec, k: int) -> None:
    """3 L nursery pot with its soil (shown as it is filled and dibbled)."""
    P = f"pot_{k}"
    pot = spec.worldbody.add_body(name=P, pos=[*L.station_xy(L.S_DENEST), stack_z(k)], quat=yaw_quat(0.7 * k))
    pot.add_freejoint(name=f"{P}_free")
    pot.explicitinertial = True
    pot.mass = L.POT_MASS
    pot.ipos = [0, 0, 0.06]
    pot.inertia = [1.2e-3, 1.2e-3, 1.0e-3]
    pot.add_geom(name=f"{P}/visual", type=GEOM.mjGEOM_MESH, meshname="pot", material="pot", **VIS)

    # Collision (switched on when the denester releases the pot): octagonal bottom + wall ring + lip.
    coll = {"group": 3, "contype": 0, "conaffinity": 0, "rgba": [0.1, 0.1, 0.1, 1]}
    half = (L.POT_R_BOT - 0.003) / np.sqrt(2)   # two squares 45 deg apart: corners on a 61 mm circle
    for a in (0.0, np.pi / 4):
        pot.add_geom(name=f"{P}/bottom_{round(np.rad2deg(a))}", type=GEOM.mjGEOM_BOX, size=[half, half, 0.003],
                     pos=[0, 0, 0.003], quat=yaw_quat(a), friction=[0.6, 0.01, 0.001], **coll)
    n_wall = 20
    taper = np.arctan2(L.POT_R_TOP - L.POT_R_BOT, L.POT_H)
    zc = L.POT_H / 2
    rc = L.pot_outer_radius(zc) - L.POT_WALL / 2
    half_w = np.pi * rc / n_wall * 1.05
    for w in range(n_wall):
        a = 2 * np.pi * w / n_wall
        q = quat_mul(yaw_quat(a), axis_angle_quat([0, 1, 0], taper))
        pot.add_geom(name=f"{P}/wall_{w}", type=GEOM.mjGEOM_BOX,
                     size=[L.POT_WALL / 2, half_w, (L.POT_H - 0.008) / 2 / np.cos(taper)],
                     pos=[rc * np.cos(a), rc * np.sin(a), zc], quat=q, friction=[0.6, 0.01, 0.001], **coll)
    n_lip = 16
    for w in range(n_lip):
        a = 2 * np.pi * (w + 0.5) / n_lip
        r = L.POT_LIP_R - 0.002
        pot.add_geom(name=f"{P}/lip_{w}", type=GEOM.mjGEOM_BOX,
                     size=[0.002, np.pi * r / n_lip * 1.05, L.POT_LIP_H / 2],
                     pos=[r * np.cos(a), r * np.sin(a), L.POT_H - L.POT_LIP_H / 2], quat=yaw_quat(a),
                     friction=[0.3, 0.01, 0.001], **coll)

    # Soil, hidden until filled: layers, the top ring in sectors (firmed by the fingertips), the plug
    # that the dibber pushes away, and the holed layers shown after dibbling.
    hidden = {"contype": 0, "conaffinity": 0, "group": HIDDEN_GROUP, "density": 0}
    for j in range(len(meshgen.SOIL_LAYERS) - 1):
        pot.add_geom(name=f"{P}/soil_{j}", type=GEOM.mjGEOM_MESH, meshname=f"soil_{j}", material="soil", **hidden)
        if j > 0:
            pot.add_geom(name=f"{P}/soil_{j}_holed", type=GEOM.mjGEOM_MESH, meshname=f"soil_{j}_holed",
                         material="soil", **hidden)
    pot.add_geom(name=f"{P}/plug", type=GEOM.mjGEOM_MESH, meshname="plug", material="soil", **hidden)
    # Top soil ring, in sectors: a thin visual piece each, and a thick collision box under its top
    # (contype 2 once filled: only the fingertips press on it; thick, so a fingertip cannot go through).
    r_in, r_out = L.TAMP_R
    da = 2 * np.pi / L.TAMP_SECTORS
    col_h = 0.04
    for j in range(L.TAMP_SECTORS):
        pot.add_geom(name=f"{P}/ring_{j}", type=GEOM.mjGEOM_MESH, meshname=f"ring_{j}", material="soil", **hidden)
        a = (j + 0.5) * da
        rc = (r_in + r_out) / 2
        pot.add_geom(name=f"{P}/ring_{j}_col", type=GEOM.mjGEOM_BOX,
                     size=[(r_out - r_in) / 2, rc * np.tan(da / 2), col_h / 2],
                     pos=[rc * np.cos(a), rc * np.sin(a), L.SOIL_Z - col_h / 2], quat=yaw_quat(a),
                     contype=0, conaffinity=0, group=3, density=0, priority=1, friction=[0.9, 0.01, 0.001],
                     rgba=[0.3, 0.2, 0.1, 1])
    # Wood-chip mulch, hidden until the pot passes under the mulcher: chips on the firmed soil ring
    # (their height follows the firming, see Line) and a few on the root ball top.
    rng = np.random.default_rng(100 + k)
    for i in range(L.N_CHIPS):
        on_rb = i % 4 == 3
        r = rng.uniform(0.011, L.RB_R_TOP - 0.004) if on_rb else rng.uniform(L.RB_R_TOP + 0.006, L.TAMP_R[1] - 0.002)
        a = rng.uniform(0, 2 * np.pi)
        q = quat_mul(yaw_quat(rng.uniform(0, 2 * np.pi)), axis_angle_quat([1, 0, 0], rng.uniform(-0.25, 0.25)))
        pot.add_geom(name=f"{P}/chip_{i}", type=GEOM.mjGEOM_BOX,
                     size=[rng.uniform(0.004, 0.0075), rng.uniform(0.0015, 0.003), 0.001],
                     pos=[r * np.cos(a), r * np.sin(a), L.SOIL_Z + (0.003 if on_rb else 0.001)], quat=q,
                     material=CHIP_MATERIALS[i % 3], **hidden)
    pot.add_geom(name=f"{P}/hole_floor", type=GEOM.mjGEOM_CYLINDER, size=[L.HOLE_R_BOT, 0.004],
                 pos=[0, 0, L.HOLE_FLOOR_Z - 0.004], contype=0, conaffinity=0, group=3, density=0,
                 solref=[0.01, 1])


def set_collision(m: mujoco.MjModel, geom: int, contype: int, conaffinity: int) -> None:
    """Switch a geom's collision bits at runtime. The body's bits (OR of its geoms, used by the
    broadphase) are set at compile time, so they must follow."""
    m.geom_contype[geom], m.geom_conaffinity[geom] = contype, conaffinity
    b = m.geom_bodyid[geom]
    m.body_contype[b] |= contype
    m.body_conaffinity[b] |= conaffinity


def pot_collision_bits(name: str) -> tuple[int, int] | None:
    """(contype, conaffinity) a pot geom gets once the pot is out of the denester (None: never collides)."""
    part = name.split("/", 1)[1] if "/" in name else ""
    if part.startswith(("bottom", "wall", "lip")):
        return 1, 1
    return None


def _add_plant(spec: mujoco.MjSpec, k: int) -> None:
    """Young tree in its root ball, standing in the cart's tray. Body origin at the root ball's bottom centre."""
    rng = np.random.default_rng(11 + k)
    _build_plant(spec, f"plant_{k}", L.cart_xy(k), rng, free=True)


def _add_tray_plants(spec: mujoco.MjSpec) -> None:
    """The rest of the crates on top of the trolley: the same trees, fixed (they wait there)."""
    for k in range(L.N_PLANTS, L.CRATES * L.CRATE_ROWS * L.CRATE_COLS):
        rng = np.random.default_rng(11 + k)
        _build_plant(spec, f"tray_plant_{k}", L.cart_xy(k), rng, free=False)


def _build_plant(spec: mujoco.MjSpec, P: str, xy, rng, free: bool) -> None:
    plant = spec.worldbody.add_body(name=P, pos=[*xy, L.CART_Z + 0.0005], quat=yaw_quat(rng.uniform(0, 2 * np.pi)))
    if free:
        plant.add_freejoint(name=f"{P}_free")
        plant.explicitinertial = True
        plant.mass = L.RB_MASS
        plant.ipos = [0, 0, 0.040]
        plant.inertia = [3.4e-4, 3.4e-4, 2.4e-4]
    plant.add_geom(name=f"{P}/rootball", type=GEOM.mjGEOM_MESH, meshname="rootball", material="rootball",
                   contype=4, conaffinity=4, condim=4, friction=[1.0, 0.01, 0.001])
    plant.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[L.RB_R_TOP - 0.002, 0.0015], pos=[0, 0, L.RB_H - 0.001],
                   material="soil", **VIS)
    for _ in range(26):  # white roots showing through the side of the root ball
        a = rng.uniform(0, 2 * np.pi)
        z0, z1 = rng.uniform(0.005, 0.07, 2)
        if abs(z1 - z0) < 0.01:
            continue
        r0, r1 = L.rb_radius(z0) + 0.0003, L.rb_radius(z1) + 0.0003
        a1 = a + rng.uniform(-0.5, 0.5)
        plant.add_geom(type=GEOM.mjGEOM_CAPSULE, size=[0.0008, 0, 0],
                       fromto=[r0 * np.cos(a), r0 * np.sin(a), z0, r1 * np.cos(a1), r1 * np.sin(a1), z1],
                       material="root", **VIS)
    if free:
        plant.add_site(name=f"{P}/top", pos=[0, 0, L.RB_H], size=[0.004, 0, 0], group=4)

    # Trunk: chain of capsules with two bending hinges per segment (stiffness from the stem's EI / l);
    # a fixed tree gets the same segments, slightly bent, without hinges.
    parent, pos, E = plant, [0, 0, L.RB_H - 0.01], 1.0e9
    for j in range(L.TREE_SEGMENTS):
        r0 = L.TREE_R_BASE + (L.TREE_R_TOP - L.TREE_R_BASE) * j / (L.TREE_SEGMENTS - 1)
        stiff = E * (np.pi * r0**4 / 4) / L.TREE_SEG_LEN
        inertia = 0.02 * (L.TREE_SEG_LEN * (L.TREE_SEGMENTS - j)) ** 2 / 3 + 2e-4
        damping = 0.3 * np.sqrt(stiff * inertia)
        if free:
            seg = parent.add_body(name=f"{P}/seg_{j}", pos=pos)
            for ax_name, ax in (("x", [1, 0, 0]), ("y", [0, 1, 0])):
                seg.add_joint(name=f"{P}/bend_{j}_{ax_name}", type=JNT.mjJNT_HINGE, axis=ax,
                              stiffness=stiff, damping=damping, armature=2e-4, range=[-0.6, 0.6])
        else:
            seg = parent.add_body(name=f"{P}/seg_{j}", pos=pos,
                                  quat=axis_angle_quat([*rng.normal(0, 1, 2), 0.0], rng.uniform(0.0, 0.05)))
        seg.add_geom(name=f"{P}/stem_{j}", type=GEOM.mjGEOM_CAPSULE, size=[r0, 0, 0],
                     fromto=[0, 0, 0.0 if j else 0.01, 0, 0, L.TREE_SEG_LEN], material="bark", density=700,
                     friction=[0.6, 0.01, 0.001])
        if j >= L.TREE_SEGMENTS - 4:
            _add_foliage(seg, j, rng)
        parent, pos = seg, [0, 0, L.TREE_SEG_LEN]


def _add_foliage(seg, j: int, rng) -> None:
    """Visual-only twigs and leaves on the upper trunk segments."""
    n_twigs = 2 if j < L.TREE_SEGMENTS - 2 else 4
    for t in range(n_twigs):
        z = rng.uniform(0.01, L.TREE_SEG_LEN)
        yaw = rng.uniform(0, 2 * np.pi) + t * 2.4
        lift = rng.uniform(0.6, 1.1)
        length = rng.uniform(0.05, 0.10)
        d = np.array([np.cos(yaw) * np.sin(lift), np.sin(yaw) * np.sin(lift), np.cos(lift)])
        end = np.array([0, 0, z]) + length * d
        seg.add_geom(type=GEOM.mjGEOM_CAPSULE, size=[0.0015, 0, 0], fromto=[0, 0, z, *end], material="bark", **VIS)
        for s in (0.35, 0.7, 1.0):
            p = np.array([0, 0, z]) + s * length * d
            for side in (-1, 1):
                leaf_yaw = yaw + side * rng.uniform(0.5, 1.2)
                q = quat_mul(yaw_quat(leaf_yaw), axis_angle_quat([0, 1, 0], -rng.uniform(0.2, 0.7)))
                q = quat_mul(q, axis_angle_quat([1, 0, 0], rng.uniform(-0.6, 0.6)))
                seg.add_geom(type=GEOM.mjGEOM_MESH, meshname="leaf", pos=p, quat=q,
                             material="leaf" if rng.random() < 0.6 else "leaf2", **VIS)


def _crate(body, center, z_floor: float, material: str = "crate") -> None:
    """Plastic crate for CRATE_ROWS x CRATE_COLS square pots: floor, walls (collision: an arm touching
    them is caught; the root balls do not touch them) and a rim, handle slots on the ends."""
    hx = L.CRATE_ROWS * L.CELL / 2 + L.CRATE_WALL
    hy = L.CRATE_COLS * L.CELL / 2 + L.CRATE_WALL
    x, y = center
    w, h = L.CRATE_WALL / 2, L.CRATE_H
    box(body, [hx, hy, 0.004], [x, y, z_floor - 0.006], material)
    for sx in (-1, 1):
        box(body, [w, hy, h / 2], [x + sx * (hx - w), y, z_floor - 0.002 + h / 2], material, collide=True,
            contype=1, conaffinity=1)
        box(body, [w + 0.003, hy + 0.003, 0.004], [x + sx * (hx - w), y, z_floor - 0.002 + h], material)
    for sy in (-1, 1):
        box(body, [hx, w, h / 2], [x, y + sy * (hy - w), z_floor - 0.002 + h / 2], material, collide=True,
            contype=1, conaffinity=1)
        box(body, [hx + 0.003, w + 0.003, 0.004], [x, y + sy * (hy - w), z_floor - 0.002 + h], material)
        box(body, [0.05, w + 0.0005, 0.012], [x, y + sy * (hy - w), z_floor + h - 0.025], "rubber")   # handle slot


def _sqpot(body, xy, z_root: float) -> None:
    """Square nursery pot around a root ball standing at z_root (visual)."""
    body.add_geom(type=GEOM.mjGEOM_MESH, meshname="sqpot", pos=[*xy, z_root - 0.0015], quat=yaw_quat(np.pi / 4),
                  material="pot", **VIS)


def _add_cart(spec: mujoco.MjSpec) -> None:
    """Nursery trolley on the left of arm A: crates of young trees in square pots, on two levels,
    and a camera on a mast over the top crates (it locates the plants for arm A)."""
    c = spec.worldbody.add_body(name="cart")
    hx_c = L.CRATE_ROWS * L.CELL / 2 + L.CRATE_WALL
    hy_c = L.CRATE_COLS * L.CELL / 2 + L.CRATE_WALL
    centres = [(L.tray_xy(0, 0, j) + L.tray_xy(L.CRATE_ROWS - 1, L.CRATE_COLS - 1, j)) / 2 for j in range(L.CRATES)]
    x0, x1 = centres[0][0] - hx_c - 0.03, centres[0][0] + hx_c + 0.03
    y0, y1 = centres[0][1] + hy_c + 0.03, centres[-1][1] - hy_c - 0.03
    yc, xc = (y0 + y1) / 2, (x0 + x1) / 2
    hy, hx = (y0 - y1) / 2, (x1 - x0) / 2
    z_top = L.CART_Z - 0.01                                                   # trolley deck under the crates
    # The root balls stand on the pot bottoms (collision: a flat plate at CART_Z, not drawn).
    box(c, [hx, hy, 0.01], [xc, yc, L.CART_Z - 0.01], "cart_tray", collide=True, name="cart_tray", group=3,
        contype=1, conaffinity=5, friction=[0.8, 0.01, 0.001])
    box(c, [hx, hy, 0.004], [xc, yc, z_top - 0.004], "cart_tray")
    box(c, [hx + 0.01, hy + 0.01, 0.015], [xc, yc, z_top - 0.025], "cart")
    for ctr in centres:
        _crate(c, ctr, L.CART_Z)
    for k in range(L.CRATES * L.CRATE_ROWS * L.CRATE_COLS):
        _sqpot(c, L.cart_xy(k), L.CART_Z)
    # Lower shelf: two more crates, the next batch (younger trees, scenery).
    z_low = 0.235
    box(c, [hx + 0.01, hy + 0.01, 0.01], [xc, yc, z_low - 0.03], "cart")
    box(c, [hx, hy, 0.004], [xc, yc, z_low - 0.016], "cart_tray")
    rng = np.random.default_rng(7)
    for j, ctr in enumerate(centres):
        _crate(c, ctr, z_low)
        for r in range(L.CRATE_ROWS):
            for col in range(L.CRATE_COLS):
                x, y = L.tray_xy(r, col, j)
                _sqpot(c, (x, y), z_low)
                c.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[L.SQPOT_TOP - 0.004, 0.002], pos=[x, y, z_low + L.RB_H - 0.002],
                           material="soil", **VIS)
                top = np.array([x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01), z_low + L.RB_H + 0.22])
                rod(c, [x, y, z_low + L.RB_H - 0.01], top, 0.0035, "bark", kind=GEOM.mjGEOM_CAPSULE)
                for i in range(6):
                    p = np.array([x, y, z_low + L.RB_H]) + (0.35 + 0.65 * i / 5) * (top - [x, y, z_low + L.RB_H])
                    qq = quat_mul(yaw_quat(rng.uniform(0, 2 * np.pi) + 2.4 * i),
                                  axis_angle_quat([0, 1, 0], -rng.uniform(0.2, 0.6)))
                    c.add_geom(type=GEOM.mjGEOM_MESH, meshname="leaf", pos=list(p), quat=qq,
                               material="leaf" if rng.random() < 0.6 else "leaf2", **VIS)
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = [xc + sx * hx, yc + sy * hy]
            rod(c, [*p, 0.07], [*p, z_top - 0.02], 0.012, "cart")
            c.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.05, 0.015], pos=[p[0], p[1], 0.05],
                       quat=axis_angle_quat([1, 0, 0], np.pi / 2), material="rubber", **VIS)
    for sx in (-1, 1):  # push handle at the far end
        rod(c, [xc + sx * hx * 0.8, y1, z_top - 0.02], [xc + sx * hx * 0.8, y1 - 0.06, z_top + 0.25], 0.01, "steel")
    rod(c, [xc - hx * 0.8, y1 - 0.06, z_top + 0.25], [xc + hx * 0.8, y1 - 0.06, z_top + 0.25], 0.012, "rubber")

    # Fixed camera over the first crate, on a mast beside the trolley (far side from the robots).
    cam_pos = np.array([centres[0][0], centres[0][1], 1.85])
    mast = np.array([x0 - 0.10, centres[0][1]])
    cm = spec.worldbody.add_body(name="tray_camera")
    rod(cm, [*mast, 0.0], [*mast, cam_pos[2] + 0.03], 0.012, "steel")
    box(cm, [0.08, 0.08, 0.006], [*mast, 0.006], "machine_dark")
    rod(cm, [*mast, cam_pos[2] + 0.03], [cam_pos[0], cam_pos[1], cam_pos[2] + 0.03], 0.008, "steel")
    box(cm, [0.045, 0.012, 0.012], cam_pos + [0, 0, 0.005], "rubber")
    cm.add_camera(name="tray_cam", pos=list(cam_pos), xyaxes=[0, -1, 0, 1, 0, 0], fovy=45)


PAD_BACK = 0.1347   # YAM linear_4310: pad body origin this far behind grasp_site, along the approach


def _foot_pose(side: str, fwd: float, out: float, down: float) -> tuple[list[float], list[float]]:
    """Pose in a pad body's frame of a piece placed in the tilted push hand's frame: `fwd` along the
    horizontal approach, `out` outwards from the pad's inner face, `down` below the grasp site."""
    c, s = np.cos(L.PUSH_TILT), np.sin(L.PUSH_TILT)
    x_s, z_s = -s * fwd + c * down, c * fwd + s * down          # in grasp_site coordinates
    sx = 1.0 if side == "left" else -1.0                         # pad axes: x = +-x_site, y, z = -z_site
    pos = [sx * x_s, -out, -(z_s + PAD_BACK)]
    f, d = np.array([-sx * s, 0.0, -c]), np.array([sx * c, 0.0, -s])
    R = np.column_stack([f, np.cross(d, f), d])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return pos, list(q)


def _add_tamper_feet(yam: mujoco.MjSpec) -> None:
    """Arm B's fingertips adapted for pushing and firming soil: a flat sole under each pad (level
    in the tilted push hand), held by a clamp block on the pad's outer side."""
    for side in ("left", "right"):
        pad = yam.body(f"{side}_pad")
        pos, q = _foot_pose(side, L.FOOT_FWD, L.FOOT_W / 2, L.FOOT_TOP + L.FOOT_T / 2)
        pad.add_geom(name=f"{side}_foot", type=GEOM.mjGEOM_BOX, size=[L.FOOT_LEN / 2, L.FOOT_W / 2, L.FOOT_T / 2],
                     pos=pos, quat=q, rgba=[0.72, 0.74, 0.77, 1], group=2, contype=1, conaffinity=1, condim=4,
                     priority=1, friction=[1.0, 0.01, 0.001], solref=[0.01, 1], density=0)
        pos, q = _foot_pose(side, L.FOOT_FWD + 0.003, 0.0035 + 0.008, L.FOOT_TOP - 0.0035)
        pad.add_geom(type=GEOM.mjGEOM_BOX, size=[0.008, 0.008, 0.0035], pos=pos, quat=q,
                     rgba=[0.93, 0.42, 0.10, 1], **VIS)


def _add_arm(spec: mujoco.MjSpec, prefix: str, base_xy, base_z: float, yaw: float, camera: bool,
             tamper: bool = False) -> None:
    yam = mujoco.MjSpec.from_file(str(YAM_XML))
    for mesh in yam.meshes:  # make mesh paths independent of the parent's meshdir
        mesh.file = str((YAM_XML.parent / "assets" / mesh.file).resolve())
    yam.meshdir = ""
    yam.compiler.meshdir = ""
    while yam.keys:
        yam.delete(yam.keys[0])
    if not camera:
        # Both arms work with the hand tilted 45 deg, the trunk passing in front of the gripper: the
        # wrist D405 (on top of the hand) would sit on the trunk. No wrist camera; a fixed camera
        # over the plant tray locates the plants.
        for ex in list(yam.excludes):
            if "wrist_camera" in (ex.bodyname1, ex.bodyname2):
                yam.delete(ex)
        yam.delete(yam.body("wrist_camera"))
    if tamper:
        _add_tamper_feet(yam)
    for g in yam.geoms:  # the robots also touch the root balls and the top soil
        if g.contype:
            g.conaffinity = 1 | 2 | 4
    frame = spec.worldbody.add_frame(pos=[*base_xy, base_z], quat=yaw_quat(yaw))
    spec.attach(yam, prefix=prefix, frame=frame)
    ped = spec.worldbody.add_body(name=f"{prefix}pedestal")
    box(ped, [0.09, 0.09, 0.008], [*base_xy, base_z - 0.008], "pedestal")
    box(ped, [0.06, 0.06, (base_z - 0.04) / 2], [*base_xy, (base_z - 0.016) / 2 + 0.008], "pedestal")
    box(ped, [0.16, 0.16, 0.01], [*base_xy, 0.01], "pedestal")


def _roller_section(spec, name: str, xs, y: float, z_top: float) -> None:
    frame = spec.worldbody.add_body(name=f"{name}_frame")
    half = L.LANE_W / 2 - L.ROLLER_R
    for i, x in enumerate(xs):
        r = frame.add_body(name=f"{name}/roller_{i}", pos=[x, y, z_top - L.ROLLER_R])
        jn = f"{name}/roller_{i}"
        r.add_joint(name=jn, type=JNT.mjJNT_HINGE, axis=[0, 1, 0], damping=1e-4, armature=1e-5)
        r.add_geom(type=GEOM.mjGEOM_CAPSULE, size=[L.ROLLER_R, 0, 0], fromto=[0, -half, 0, 0, half, 0],
                   material="roller", friction=[0.8, 0.01, 0.001])
        act = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_velocity(kv=0.01)
        act.forcerange = [-L.DRIVE_TORQUE, L.DRIVE_TORQUE]
        act.forcelimited = True
    x0, x1 = xs[0] - 0.02, xs[-1] + 0.02
    for side in (-1, 1):
        yy = y + side * (L.LANE_W / 2 + 0.008)
        box(frame, [(x1 - x0) / 2, 0.006, 0.02], [(x0 + x1) / 2, yy, z_top - L.ROLLER_R], "machine_dark")
        for x in (x0 + 0.03, x1 - 0.03):
            box(frame, [0.015, 0.015, (z_top - 0.04) / 2], [x, yy, (z_top - 0.04) / 2], "machine_dark")


def _add_outfeed(spec: mujoco.MjSpec) -> None:
    """Outfeed conveyor from the ejector to the row former, the sweep and the pallet."""
    wb = spec.worldbody
    y, zt = L.CONV_Y, L.CONV_Z
    xs = L.CONV_X[0] + L.ROLLER_PITCH * np.arange(int((L.CONV_X[1] - L.CONV_X[0]) / L.ROLLER_PITCH + 1e-9) + 1)
    _roller_section(spec, "conveyor", xs, y, zt)
    fixed = wb.add_body(name="outfeed_fixed")
    # Dead plate bridging the wheel rim and the first roller.
    x_dp0, x_dp1 = L.WHEEL_R + 0.004, xs[0] - L.ROLLER_R - 0.002
    box(fixed, [(x_dp1 - x_dp0) / 2, L.LANE_W / 2, 0.005], [(x_dp0 + x_dp1) / 2, y, zt - 0.005], "steel",
        collide=True, name="dead_plate", friction=[0.3, 0.01, 0.001])
    # Guide rails along the first part of the conveyor (the row former is open on the pallet side).
    xr0, xr1 = xs[0] - 0.01, L.row_x(L.ROW_N - 1) - L.POT_LIP_R - 0.02
    for side in (-1, 1):
        yy = y + side * (L.POT_LIP_R + 0.012)
        box(fixed, [(xr1 - xr0) / 2, 0.004, 0.012], [(xr0 + xr1) / 2, yy, zt + 0.10], "yellow", collide=True,
            name=f"guide_{'l' if side > 0 else 'r'}", friction=[0.2, 0.01, 0.001])
        for x in (xr0 + 0.02, xr1 - 0.02):
            rod(fixed, [x, yy + side * 0.01, zt - 0.01], [x, yy + side * 0.01, zt + 0.11], 0.005, "steel")
    # Row stop at the conveyor end, and the row sensor (light barrier) at the third place.
    box(fixed, [0.006, L.LANE_W / 2, 0.08], [L.ROW_STOP_X + 0.006, y, zt + 0.08], "yellow", collide=True,
        name="row_stop", friction=[0.2, 0.01, 0.001])
    xe = L.row_x(L.ROW_N - 1)
    ye = y - L.LANE_W / 2 - 0.02
    box(fixed, [0.008, 0.008, 0.012], [xe, ye, zt + 0.06], "machine_dark")
    rod(fixed, [xe, ye, zt - 0.01], [xe, ye, zt + 0.05], 0.004, "steel")
    fixed.add_site(name="row_eye", pos=[xe, ye + 0.009, zt + 0.06], size=[0.003, 0, 0], group=4)
    # Motor.
    fixed.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.03, 0.04], pos=[xs[-1] - 0.05, y - L.LANE_W / 2 - 0.06, zt - 0.05],
                   quat=axis_angle_quat([1, 0, 0], np.pi / 2), material="motor", **VIS)

    # Sweep: a bar on the -y side pushes the finished row across onto the pallet.
    x0, x1 = L.row_x(L.ROW_N - 1) - L.POT_LIP_R + 0.01, L.ROW_STOP_X - 0.005
    y_rest = y - L.LANE_W / 2 - 0.035
    sw = wb.add_body(name="sweep", pos=[(x0 + x1) / 2, y_rest, zt + L.SWEEP_Z])
    sw.add_joint(name="sweep", type=JNT.mjJNT_SLIDE, axis=[0, 1, 0], range=[0, L.SWEEP_STROKE], damping=60.0)
    box(sw, [(x1 - x0) / 2, 0.008, 0.025], [0, 0, 0], "yellow", collide=True, name="sweep_bar",
        friction=[0.2, 0.01, 0.001])
    for sx in (-1, 1):
        rod(sw, [sx * 0.15, -0.008, 0.0], [sx * 0.15, -0.25, 0.0], 0.008, "steel")
    sw.explicitinertial = True
    sw.mass, sw.ipos, sw.inertia = 2.0, [0, 0, 0], [0.05, 0.01, 0.05]
    for sx in (-1, 1):
        rod(fixed, [(x0 + x1) / 2 + sx * 0.15, y_rest - 0.10, zt + L.SWEEP_Z],
            [(x0 + x1) / 2 + sx * 0.15, y_rest - 0.30, zt + L.SWEEP_Z], 0.02, "pneumatic")
    box(fixed, [0.20, 0.02, 0.03], [(x0 + x1) / 2, y_rest - 0.31, zt + L.SWEEP_Z], "machine_dark")
    box(fixed, [0.02, 0.02, (zt + L.SWEEP_Z) / 2], [(x0 + x1) / 2, y_rest - 0.31, (zt + L.SWEEP_Z) / 2],
        "machine_dark")
    act = spec.add_actuator(name="sweep", target="sweep", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=3000.0, kv=300.0)
    act.forcerange = [-250, 250]
    act.forcelimited = True

    # Pallet on a stand.
    pal = wb.add_body(name="pallet")
    px0, px1 = L.PALLET_X
    py0, py1 = L.PALLET_Y
    pz = L.PALLET_Z
    box(pal, [(px1 - px0) / 2, (py1 - py0) / 2, 0.01], [(px0 + px1) / 2, (py0 + py1) / 2, pz - 0.01], "pallet",
        collide=True, name="pallet_top", friction=[L.PALLET_FRICTION, 0.01, 0.001])
    for j in range(5):  # deck boards
        yj = py0 + (j + 0.5) * (py1 - py0) / 5
        box(pal, [(px1 - px0) / 2, (py1 - py0) / 12, 0.002], [(px0 + px1) / 2, yj, pz + 0.0005], "pallet")
    for yj in (py0 + 0.04, (py0 + py1) / 2, py1 - 0.04):
        box(pal, [(px1 - px0) / 2, 0.04, 0.035], [(px0 + px1) / 2, yj, pz - 0.055], "pallet")
    box(pal, [(px1 - px0) / 2 + 0.02, (py1 - py0) / 2 + 0.02, (pz - 0.09) / 2],
        [(px0 + px1) / 2, (py0 + py1) / 2, (pz - 0.09) / 2], "machine_dark")
    box(pal, [0.006, (py1 - py0) / 2, 0.06], [px1 + 0.006, (py0 + py1) / 2, pz + 0.06], "yellow",
        collide=True, name="pallet_stop_x")
    box(pal, [(px1 - px0) / 2, 0.006, 0.03], [(px0 + px1) / 2, py1 + 0.006, pz + 0.03], "yellow",
        collide=True, name="pallet_stop_y")


def _add_mulcher(spec: mujoco.MjSpec) -> None:
    """Wood-chip mulcher beside the outfeed: a big bag of chips, an inclined screw conveyor up to
    a hopper on a column, and a side spout that drops chips on each pot rolling by. Visual only:
    the line controller shows the chip stream and the chips on the pot (line.py)."""
    wb = spec.worldbody
    mu = wb.add_body(name="mulcher")
    x, zt = L.CHIP_X, L.CONV_Z
    col = np.array([x, 0.30])            # on the far side of the lane, before the pallet
    hop = np.array([x, 0.21, 1.17])
    rod(mu, [*col, 0.0], [*col, 1.36], 0.035, "frame")                                # column
    box(mu, [0.12, 0.12, 0.012], [*col, 0.012], "machine_dark")                      # foot
    box(mu, [0.025, 0.05, 0.025], [col[0], (col[1] + hop[1]) / 2, 1.33], "frame")    # arm to the hopper
    box(mu, [0.09, 0.08, 0.075], hop + [0, 0, 0.02], "hopper")                        # hopper
    box(mu, [0.095, 0.085, 0.006], hop + [0, 0, 0.10], "frame")
    for j in range(3):                                                                # chips heaped in it
        box(mu, [0.085 - 0.012 * j, 0.075 - 0.012 * j, 0.008], hop + [0, 0, 0.108 + 0.01 * j], CHIP_MATERIALS[j],
            quat=yaw_quat(0.25 * j))
    rod(mu, hop + [0, 0, -0.055], hop + [0, 0, -0.09], 0.045, "hopper")              # vibrating feeder
    box(mu, [0.03, 0.03, 0.02], hop + [0.07, 0, -0.07], "motor")
    spout = np.array([x, L.CONV_Y + L.CHIP_SPOUT_Y, zt + L.POT_H + 0.055])
    rod(mu, hop + [0, 0, -0.09], spout + [0, 0, 0.01], 0.022, "frame")               # chute down to the spout
    rod(mu, spout + [0, 0, 0.01], spout - [0, 0, 0.008], 0.026, "frame")
    mu.add_geom(name="chip_stream", type=GEOM.mjGEOM_CYLINDER, size=[0.012, 0, 0],
                fromto=[*(spout - [0, 0, 0.008]), *(spout[:2]), zt + L.SOIL_Z], material="chips",
                contype=0, conaffinity=0, group=HIDDEN_GROUP, density=0)
    box(mu, [0.008, 0.008, 0.012], [x - L.CHIP_ZONE - 0.01, L.CONV_Y - L.LANE_W / 2 - 0.02, zt + 0.06],
        "machine_dark")                                                                # pot sensor
    # Big bag of chips on a stand behind the cell, and the inclined screw conveyor feeding the hopper.
    bag = np.array([0.50 + L.RING_SHIFT, 0.87])
    hb = 0.19
    box(mu, [hb + 0.03, hb + 0.03, 0.09], [*bag, 0.09], "pallet")
    box(mu, [hb, hb, 0.23], [*bag, 0.41], "bigbag")
    box(mu, [hb - 0.02, hb - 0.02, 0.012], [*bag, 0.645], "chips")
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = bag + [sx * hb, sy * hb]
            rod(mu, [*p, 0.18], [*p, 0.86], 0.012, "frame")
            rod(mu, [*p, 0.86], [*(bag + [sx * 0.05, sy * 0.05]), 0.66], 0.006, "bigbag")
    for sx in (-1, 1):
        rod(mu, [bag[0] + sx * hb, bag[1] - hb, 0.86], [bag[0] + sx * hb, bag[1] + hb, 0.86], 0.012, "frame")
    feed = np.array([bag[0] + 0.10, bag[1] - hb - 0.06, 0.45])
    rod(mu, feed, hop + [0.0, 0.05, 0.06], 0.04, "pneumatic")                          # screw conveyor
    box(mu, [0.05, 0.05, 0.05], feed - [0, 0, 0.06], "motor")


def _add_task_constraints(spec: mujoco.MjSpec) -> None:
    weld = mujoco.mjtEq.mjEQ_WELD
    body = mujoco.mjtObj.mjOBJ_BODY
    # The second body is a placeholder: the controllers point it at the actual plant when they
    # switch the constraint on (constraints.retarget).
    spec.add_equality(name="grasp_a", type=weld, objtype=body, name1="a/link_6", name2="plant_0",
                      active=False, solref=[0.005, 1])
    for k in range(L.N_POTS):
        spec.add_equality(name=f"denest_{k}", type=weld, objtype=body, name1="world", name2=f"pot_{k}",
                          active=False, solref=[0.02, 1])
        spec.add_equality(name=f"planted_{k}", type=weld, objtype=body, name1=f"pot_{k}", name2="plant_0",
                          active=False, solref=[0.004, 1])


def build_spec() -> mujoco.MjSpec:
    if not (meshgen.MESH_DIR / "sqpot.stl").exists():
        meshgen.generate()
    spec = mujoco.MjSpec()
    spec.modelname = "rempotage_cell"
    spec.compiler.autolimits = True
    spec.compiler.degree = False  # all angles in this file are radians
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    # Constraint softness from the exact inverse inertia: the pots' mass changes as they are filled.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_DIAGEXACT
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.global_.azimuth = 235
    spec.visual.global_.elevation = -30
    spec.visual.quality.shadowsize = 8192
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.stat.center = [0.1, -0.2, 0.9]
    spec.stat.extent = 1.4

    _add_assets(spec)
    _add_environment(spec)
    _add_wheel(spec)
    _add_stations(spec)
    for k in range(L.N_POTS):
        _add_pot(spec, k)
    for k in range(L.N_PLANTS):
        _add_plant(spec, k)
    _add_tray_plants(spec)
    _add_cart(spec)
    _add_arm(spec, "a/", L.ARM_A_BASE, L.ARM_A_Z, L.ARM_A_YAW, camera=False)
    _add_arm(spec, "b/", L.ARM_B_BASE, L.ARM_B_Z, L.ARM_B_YAW, camera=False, tamper=True)
    _add_outfeed(spec)
    _add_mulcher(spec)
    _add_task_constraints(spec)
    return spec


def build_model() -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_spec()
    return spec.compile(), spec


def export_xml(path: Path) -> Path:
    """Write the cell as a standalone MJCF (e.g. for MuJoCo's `simulate` app).

    Mesh paths are written relative to the output file. The pots start in the denester stack with
    collisions off (the line controller switches them on as it releases each pot).
    """
    path = Path(path).resolve()
    spec = build_spec()
    model = spec.compile()
    qpos, ctrl = model.qpos0.copy(), np.zeros(model.nu)
    for prefix in ("a/", "b/"):
        for k, q in enumerate(L.ARM_HOME_Q):
            qpos[model.jnt_qposadr[model.joint(f"{prefix}joint{k + 1}").id]] = q
            ctrl[model.actuator(f"{prefix}joint{k + 1}").id] = q
        ctrl[model.actuator(f"{prefix}gripper").id] = L.GRIPPER_OPEN
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)
    spec.compile()
    xml = spec.to_xml()
    for mesh in spec.meshes:  # to_xml() recompiles, so relativize in the text
        xml = xml.replace(f'file="{mesh.file}"', f'file="{os.path.relpath(mesh.file, path.parent)}"')
    path.write_text(xml)
    return path
