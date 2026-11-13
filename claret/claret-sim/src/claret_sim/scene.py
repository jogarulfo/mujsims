"""Builds the MuJoCo model of the Claret packing cell with MjSpec.

Cell contents, after the photo of the Claret line:
  - a blue production machine with a hood and a rubber strip curtain. Inside, finished parts
    land on a flat outlet plate and an ejector pushes them out onto the chute;
  - a wide stainless outlet tray on blue legs, sloping down towards the robot to a rubber end
    stop. Parts end up anywhere on it. A "tray full" light barrier near the machine;
  - one I2RT YAM arm (prefix ``r/``) on a pedestal, with narrow bore jaws under its fingers (it
    grips the parts from inside their bore) and a retractable vacuum cup on the side of its
    gripper;
  - a cardboard carton lined with a plastic bag, on a roller stand;
  - a magazine of cardboard layer sheets;
  - the parts (ribbed rubber-metal bushings). They wait inside the machine until it "makes" them.

Equality constraints that the controllers switch at runtime (all inactive at start). Their
second body is set when they are switched on (constraints.retarget):
  grasp    weld  r/link_6 <-> part    grasp assist (optional)
  vacuum   weld  cup      <-> sheet   suction cup holding a sheet
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from claret_sim import layout as L
from claret_sim import meshgen

ASSETS = Path(__file__).parent / "assets"
YAM_XML = ASSETS / "yam" / "yam_linear_4310.xml"
ARM = "r/"

GEOM = mujoco.mjtGeom
JNT = mujoco.mjtJoint

VIS = {"contype": 0, "conaffinity": 0, "group": 2, "density": 0}
COL = {"group": 3, "rgba": [0.5, 0.5, 0.5, 1]}

COLORS = {
    "machine_blue": [0.13, 0.42, 0.78, 1],
    "machine_blue_dark": [0.09, 0.30, 0.58, 1],
    "hood": [0.55, 0.42, 0.30, 1],
    "rubber": [0.10, 0.10, 0.10, 1],
    "steel": [0.72, 0.74, 0.76, 1],
    "steel_dark": [0.42, 0.44, 0.47, 1],
    "part": [0.07, 0.07, 0.075, 1],
    "cardboard": [0.78, 0.62, 0.42, 1],
    "cardboard_edge": [0.70, 0.54, 0.35, 1],
    "sheet": [0.84, 0.70, 0.50, 1],
    "liner": [0.92, 0.94, 0.96, 0.33],
    "yellow": [0.95, 0.78, 0.10, 1],
    "black": [0.05, 0.05, 0.05, 1],
    "pedestal": [0.30, 0.31, 0.33, 1],
    "sensor": [0.20, 0.55, 0.95, 1],
    "lamp_off": [0.25, 0.25, 0.25, 1],
    "pneumatic": [0.78, 0.80, 0.83, 1],
}
LAMPS = {"green": [0.1, 0.9, 0.2, 1], "amber": [1.0, 0.65, 0.05, 1], "blue": [0.15, 0.45, 1.0, 1]}


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
    """Shorthand for a box geom: visual only unless collide=True."""
    args = {"type": GEOM.mjGEOM_BOX, "size": list(size), "pos": list(pos)}
    if quat is not None:
        args["quat"] = list(quat)
    if name:
        args["name"] = name
    if material:
        args["material"] = material
    if not collide:
        args.update(VIS)
    args.update(kw)
    return body.add_geom(**args)


def rod(body, a, b, r, material, collide=False, name=None):
    args = {"type": GEOM.mjGEOM_CYLINDER, "size": [r, 0, 0], "fromto": [*a, *b], "material": material}
    if name:
        args["name"] = name
    if not collide:
        args.update(VIS)
    return body.add_geom(**args)


# --- Assets / environment ----------------------------------------------------------------


def _add_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.60, 0.66, 0.70], rgb2=[0.16, 0.17, 0.19], width=512, height=3072)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.42, 0.50, 0.44], rgb2=[0.42, 0.50, 0.44], markrgb=[0.38, 0.45, 0.40],
                     width=300, height=300)  # green-grey epoxy floor, as in the photo
    spec.add_material(name="floor", textures=["", "floor"], texuniform=True, texrepeat=[2, 2],
                      reflectance=0.0, specular=0.05, shininess=0.1)
    shiny = ("steel", "steel_dark", "pneumatic", "machine_blue")
    for name, rgba in COLORS.items():
        spec.add_material(name=name, rgba=rgba, specular=0.5 if name in shiny else 0.15,
                          shininess=0.6 if name in shiny else 0.2)
    spec.add_material(name="lamp_glow", rgba=[1, 1, 1, 1], emission=0.8)
    for mesh in ("bushing", "bushing_col", "bore_seg", "cup"):
        spec.add_mesh(name=mesh, file=str(meshgen.MESH_DIR / f"{mesh}.stl"))


def _add_environment(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.6, 0.4, 3.0], dir=[-0.2, -0.1, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=True, diffuse=[0.75, 0.75, 0.72], specular=[0.2, 0.2, 0.2])
    wb.add_light(name="fill", pos=[2.0, -1.5, 2.2], dir=[-0.6, 0.5, -0.6], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=False, diffuse=[0.35, 0.35, 0.38])
    wb.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[4, 4, 0.05], material="floor")

    nest = np.array([L.CHUTE_X, L.STOP_Y + 0.06, L.NEST_Z])
    box_c = np.array([*L.BOX_CENTER, L.BOX_FLOOR_Z])
    cams = {
        "overview": ([1.75, -1.35, 1.75], [0.15, 0.35, 0.65], 45),
        "photo": ([0.95, -0.55, 1.65], [0.05, 0.55, 0.70], 55),       # roughly the photo's viewpoint
        "front": ([1.55, 0.25, 1.30], [0.15, 0.15, 0.75], 50),
        "machine": ([0.75, 0.05, 1.20], [0.0, 1.05, 0.85], 50),
        "tray": (nest + [0.62, -0.30, 0.52], nest + [0.0, 0.08, 0.0], 38),
        "tray_top": (nest + [0.05, -0.12, 0.75], nest + [0.0, 0.08, 0.0], 45),
        "box": (box_c + [0.42, -0.30, 0.55], box_c + [0.0, 0.0, 0.03], 50),
        "box_top": (box_c + [0.10, 0.0, 0.85], box_c + [0.0, 0.0, 0.0], 40),
        "box_close": (box_c + [0.24, -0.30, 0.66], box_c + [0.03, 0.03, 0.03], 42),
        "magazine": ([0.55, -0.95, 1.20], [-0.03, -0.30, 0.68], 45),
    }
    for name, (pos, target, fovy) in cams.items():
        wb.add_camera(name=name, pos=list(pos), xyaxes=look_at(pos, target), fovy=fovy)


# --- Robot ------------------------------------------------------------------------------


def _add_pedestal(spec: mujoco.MjSpec) -> None:
    ped = spec.worldbody.add_body(name="pedestal")
    box(ped, [0.20, 0.20, 0.008], [0, 0, 0.008], "pedestal", name="pedestal_foot", collide=True)
    ped.add_geom(name="pedestal_column", type=GEOM.mjGEOM_CYLINDER, size=[0.085, (L.BASE_Z - 0.04) / 2],
                 pos=[0, 0, (L.BASE_Z - 0.04) / 2 + 0.016], material="pedestal")
    box(ped, [0.11, 0.11, 0.012], [0, 0, L.BASE_Z - 0.012], "pedestal", name="pedestal_top", collide=True)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):  # anchor bolts
        ped.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.008, 0.006], pos=[0.16 * sx, 0.16 * sy, 0.022],
                     material="steel", **VIS)
    # Stack light on the pedestal's back corner.
    p = np.array([-0.09, -0.09])
    rod(ped, [*p, L.BASE_Z], [*p, L.BASE_Z + 0.30], 0.008, "steel_dark")
    for k, lamp in enumerate(("blue", "amber", "green")):
        ped.add_geom(name=f"lamp_{lamp}", type=GEOM.mjGEOM_CYLINDER, size=[0.022, 0.022],
                     pos=[*p, L.BASE_Z + 0.33 + 0.046 * k], material="lamp_glow", rgba=COLORS["lamp_off"], **VIS)
    ped.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.023, 0.006], pos=[*p, L.BASE_Z + 0.33 + 0.046 * 3 - 0.016],
                 material="black", **VIS)


def _add_arm(spec: mujoco.MjSpec) -> None:
    yam = mujoco.MjSpec.from_file(str(YAM_XML))
    for mesh in yam.meshes:  # make mesh paths independent of the parent's meshdir
        mesh.file = str((YAM_XML.parent / "assets" / mesh.file).resolve())
    yam.meshdir = ""
    yam.compiler.meshdir = ""
    while yam.keys:
        yam.delete(yam.keys[0])
    # No wrist camera in this cell: the vacuum cup takes the side of the gripper.
    for ex in list(yam.excludes):
        if "wrist_camera" in (ex.bodyname1, ex.bodyname2):
            yam.delete(ex)
    yam.delete(yam.body("wrist_camera"))
    frame = spec.worldbody.add_frame(pos=[0, 0, L.BASE_Z])
    spec.attach(yam, prefix=ARM, frame=frame)
    _add_bore_jaws(spec)
    _add_vacuum_tool(spec)


def _add_bore_jaws(spec: mujoco.MjSpec) -> None:
    """Narrow aluminium jaws under the stock pads, for gripping a part from inside its bore.

    Pad body frame: z = -(0.1347 + depth along the approach axis), the pad face (towards the
    other finger) at y = 0, the finger behind it at y < 0. Each jaw lies behind the pad face
    plane, so the closed jaws are 2 * JAW_T thick and their outer faces grip the bore.
    The jaws collide only with the bore lining of the parts (contype 4).
    """
    top, tip = -0.002, L.JAW_TIP
    zc, hz = -(L.GRASP_SITE_Z + (top + tip) / 2), (tip - top) / 2
    for side in ("left", "right"):
        pad = spec.body(f"{ARM}{side}_pad")
        pad.add_geom(name=f"{ARM}{side}_jaw", type=GEOM.mjGEOM_BOX, size=[L.JAW_W / 2, L.JAW_T / 2, hz],
                     pos=[0, -L.JAW_T / 2, zc], group=3, contype=4, conaffinity=4, condim=6, priority=2,
                     friction=[1.0, 0.01, 0.002], density=2700, rgba=[0.7, 0.7, 0.7, 1])  # rubber-faced
        box(pad, [L.JAW_W / 2, L.JAW_T / 2, hz], [0, -L.JAW_T / 2, zc], "steel")
        box(pad, [L.JAW_W / 2 - 0.001, 0.0004, L.JAW_INSERT / 2], [0, -L.JAW_T - 0.0004, -(L.GRASP_SITE_Z + tip
            - L.JAW_INSERT / 2)], "rubber")  # grip pad on the outer face
        box(pad, [L.JAW_W / 2 + 0.002, L.JAW_T / 2 + 0.002, 0.004], [0, -L.JAW_T / 2 - 0.001,
            -(L.GRASP_SITE_Z - 0.004)], "steel_dark")  # clamp block on the stock finger
    spec.body(ARM + "link_6").add_site(name=ARM + "grip_site", pos=[0, 0, L.GRASP_SITE_Z + L.GRIP_SITE],
                                       quat=[1, 0, 0, -1], size=[0.004, 0, 0], group=4)


def _add_vacuum_tool(spec: mujoco.MjSpec) -> None:
    """Pneumatic cylinder + suction cup on the side of the gripper (grasp-frame +x side).

    link_6 frame: z is the approach axis, grasp_site at z = 0.1347, grasp-frame +x = link_6 -y.
    """
    link6 = spec.body(ARM + "link_6")
    gz = 0.1347
    y = -L.CUP_OFFSET
    lip_retracted = gz + L.CUP_RETRACTED
    barrel_top, barrel_bot = -0.055, lip_retracted - 0.050
    link6.add_geom(name="cup_barrel", type=GEOM.mjGEOM_CYLINDER, size=[0.011, 0, 0],
                   fromto=[0, y, barrel_top, 0, y, barrel_bot], material="pneumatic", **VIS)
    link6.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.013, 0, 0], fromto=[0, y, barrel_top, 0, y, barrel_top + 0.006],
                   material="black", **VIS)
    link6.add_geom(type=GEOM.mjGEOM_BOX, size=[0.012, 0.016, 0.006], pos=[0, y + 0.020, 0.030],
                   material="steel_dark", **VIS)  # bracket to the wrist flange
    link6.add_geom(type=GEOM.mjGEOM_CAPSULE, size=[0.0022, 0, 0],
                   fromto=[0.006, y - 0.004, barrel_top + 0.004, 0.02, y - 0.03, barrel_top - 0.03],
                   rgba=[0.15, 0.35, 0.75, 1],
                   **VIS)  # air hose
    cup = link6.add_body(name="cup", pos=[0, y, lip_retracted], gravcomp=1)
    cup.add_joint(name="cup_slide", type=JNT.mjJNT_SLIDE, axis=[0, 0, 1],
                  range=[0, L.CUP_EXTENDED - L.CUP_RETRACTED], armature=0.01, damping=2.0)
    flip = axis_angle_quat([1, 0, 0], np.pi)  # cup mesh lip at z = 0, stem up: point the lip along +z
    cup.add_geom(type=GEOM.mjGEOM_MESH, meshname="cup", quat=flip, material="black", **VIS)
    cup.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.005, 0, 0], fromto=[0, 0, -0.03, 0, 0, -0.16],
                 material="steel", **VIS)
    cup.add_geom(name="cup_lip", type=GEOM.mjGEOM_CYLINDER, size=[L.CUP_R, 0.003], pos=[0, 0, -0.003],
                 rgba=[0.1, 0.1, 0.1, 1], group=3, friction=[1.0, 0.01, 0.001])
    cup.add_site(name="cup_site", pos=[0, 0, 0], size=[0.004, 0, 0], group=4)
    cup.explicitinertial = True
    cup.mass, cup.ipos, cup.inertia = 0.05, [0, 0, -0.03], [2e-5, 2e-5, 1e-5]
    act = spec.add_actuator(name="cup/cylinder", target="cup_slide", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=300.0, kv=15.0)
    act.ctrlrange = [0, L.CUP_EXTENDED - L.CUP_RETRACTED]
    act.ctrllimited = True
    act.forcerange = [-40, 40]
    act.forcelimited = True


# --- Machine + chute ------------------------------------------------------------------------


def _add_machine(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    m = wb.add_body(name="machine")
    F = L.MACHINE_FRONT_Y
    W, D, H = L.MACHINE_W, L.MACHINE_D, L.MACHINE_H
    oz = L.OUTLET_Z
    back = F + 0.12  # main cabinet front face (the outlet housing sticks out in front of it)
    box(m, [W / 2, D / 2, H / 2], [0, back + D / 2, H / 2], "machine_blue")
    # Upper hood: bronze door with a handle above the curtain, framed in blue.
    hood_y = F + 0.05
    box(m, [0.26, 0.07, 0.30], [0, hood_y + 0.07, oz + 0.47], "machine_blue")
    box(m, [0.20, 0.004, 0.18], [0, hood_y - 0.004, oz + 0.43], "hood")
    box(m, [0.025, 0.012, 0.022], [0, hood_y - 0.02, oz + 0.47], "steel")
    box(m, [0.012, 0.010, 0.012], [-0.02, hood_y - 0.012, oz + 0.47], "steel")
    box(m, [0.012, 0.010, 0.012], [0.02, hood_y - 0.012, oz + 0.47], "steel")
    for sx in (-1, 1):  # door hinges
        box(m, [0.008, 0.008, 0.025], [sx * 0.205, hood_y - 0.008, oz + 0.55], "steel_dark")
    # Warning sign (yellow triangle on the hood frame) and a side control box.
    box(m, [0.045, 0.002, 0.06], [-0.30, back - 0.002, oz + 0.52], "yellow")
    box(m, [0.03, 0.003, 0.012], [-0.30, back - 0.004, oz + 0.50], "black")
    box(m, [0.09, 0.07, 0.12], [0.36, back - 0.07, oz + 0.25], "machine_blue_dark")
    for k, c in enumerate(([0.1, 0.8, 0.2, 1], [0.9, 0.1, 0.1, 1])):
        m.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.012, 0.006], pos=[0.33 + 0.05 * k, back - 0.142, oz + 0.27],
                   quat=axis_angle_quat([1, 0, 0], np.pi / 2), rgba=c, **VIS)
    # Outlet tunnel: side cheeks, roof, and the lower housing under the outlet plate.
    for sx in (-1, 1):
        box(m, [0.04, 0.10, 0.09], [sx * (L.CHUTE_W / 2 + 0.04), F + 0.10, oz + 0.09], "machine_blue",
            name=f"outlet_cheek_{'l' if sx > 0 else 'r'}", collide=True)
    box(m, [0.26, 0.10, 0.02], [0, F + 0.10, oz + 0.165], "machine_blue")
    box(m, [0.26, 0.15, (oz - 0.30) / 2], [0, F + 0.07, 0.30 + (oz - 0.30) / 2 - 0.006], "machine_blue")
    box(m, [0.10, 0.004, 0.05], [0.10, F - 0.084, oz - 0.18], "steel_dark")  # nameplate
    # Flat outlet plate inside the machine (the part lands here), low friction like the chute.
    box(m, [L.CHUTE_W / 2, L.OUTLET_DEPTH / 2, 0.005], [0, F + L.OUTLET_DEPTH / 2, oz - 0.005], "steel",
        name="outlet_plate", collide=True, friction=[L.CHUTE_FRICTION, 0.005, 0.0001], priority=1)
    box(m, [L.CHUTE_W / 2, 0.01, 0.06], [0, F + L.OUTLET_DEPTH + 0.01, oz + 0.06], "steel_dark",
        name="outlet_back", collide=True)

    # Ejector: a pusher plate on a pneumatic slide, pushes the fresh part over the lip.
    ej = m.add_body(name="ejector", pos=[0, L.SPAWN_Y + L.PART_RIB_R + 0.012, oz])
    ej.add_joint(name="ejector", type=JNT.mjJNT_SLIDE, axis=[0, -1, 0], range=[0, L.EJECT_STROKE],
                 armature=0.05, damping=5.0)
    box(ej, [L.CHUTE_W / 2 - 0.012, 0.006, 0.022], [0, 0, 0.034], "steel_dark", name="ejector_plate", collide=True,
        friction=[0.3, 0.005, 0.0001], priority=1)
    for x in (-0.08, 0.08):
        rod(ej, [x, 0.006, 0.03], [x, 0.20, 0.03], 0.008, "steel")
    ej.explicitinertial = True
    ej.mass, ej.ipos, ej.inertia = 0.3, [0, 0.05, 0.03], [1e-3, 1e-4, 1e-3]
    act = spec.add_actuator(name="ejector/cylinder", target="ejector", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=400.0, kv=40.0)
    act.ctrlrange = [0, L.EJECT_STROKE]
    act.ctrllimited = True
    act.forcerange = [-30, 30]
    act.forcelimited = True

    # Rubber strip curtain over the outlet (each strip swings when a part goes through).
    n_strips, strip_w, strip_h = 8, 0.04, 0.135
    top = oz + 0.15
    for k in range(n_strips):
        x = (k - (n_strips - 1) / 2) * strip_w
        s = wb.add_body(name=f"curtain_{k}", pos=[x, L.CURTAIN_Y, top])
        s.add_joint(name=f"curtain_{k}", type=JNT.mjJNT_HINGE, axis=[1, 0, 0], stiffness=0.02, damping=0.004,
                    range=[-1.3, 1.3], armature=1e-4)
        s.add_geom(name=f"curtain_{k}_strip", type=GEOM.mjGEOM_BOX, size=[strip_w / 2 - 0.001, 0.0015, strip_h / 2],
                   pos=[0, 0, -strip_h / 2], material="rubber", density=1200, friction=[0.05, 0.005, 0.0001], priority=3,
                   contype=2, conaffinity=2)
    box(m, [0.17, 0.012, 0.008], [0, L.CURTAIN_Y, top + 0.008], "steel_dark")


def _add_chute(spec: mujoco.MjSpec) -> None:
    """Wide outlet tray: no guide lane, parts end up anywhere between the side walls."""
    ch = spec.worldbody.add_body(name="chute")
    th = 0.004
    fric = {"friction": [L.CHUTE_FRICTION, 0.005, 0.0001], "priority": 1}
    a = L.CHUTE_SLOPE
    q = axis_angle_quat([1, 0, 0], a)
    n = np.array([0, -np.sin(a), np.cos(a)])           # tray floor normal
    t_ = np.array([0, np.cos(a), np.sin(a)])           # up the slope

    def on_slope(y, h=0.0, x=0.0):
        """Point at height h above the tray floor (normal to it), at horizontal position y."""
        return np.array([x, y, L.chute_z(y)]) + h * n

    # Floor: one sloped sheet from just below the end stop up to the outlet plate (flush there).
    y0, y1 = L.STOP_Y - 0.02, L.MACHINE_FRONT_Y
    run = (y1 - y0) / np.cos(a)
    c = np.array([L.CHUTE_X, (y0 + y1) / 2, L.NEST_Z + ((y0 + y1) / 2 - L.STOP_Y) * np.tan(a)]) - n * th / 2
    box(ch, [L.CHUTE_W / 2, run / 2, th / 2], c, "steel", quat=q, name="chute_floor", collide=True, **fric)
    # Tray side walls (the tall stainless sides of the photo).
    mid_y = (y0 + y1) / 2
    for sx in (-1, 1):
        box(ch, [0.002, run / 2, L.CHUTE_WALL_H / 2], on_slope(mid_y, L.CHUTE_WALL_H / 2, sx * (L.CHUTE_W / 2 + 0.002)),
            "steel", quat=q, name=f"tray_wall_{'l' if sx > 0 else 'r'}", collide=True, **fric)
        rod(ch, on_slope(y0, L.CHUTE_WALL_H, sx * (L.CHUTE_W / 2 + 0.002)),
            on_slope(y1, L.CHUTE_WALL_H, sx * (L.CHUTE_W / 2 + 0.002)), 0.004, "steel")  # rolled top edge
    # End stop across the whole width (square to the tray floor) with a rubber bumper (soft
    # contact). Kept low (26 mm): the part's centre of mass is at 28 mm.
    box(ch, [L.CHUTE_W / 2 + 0.004, 0.004, 0.015], on_slope(L.STOP_Y, 0.011) - 0.014 * t_, "steel", quat=q)
    box(ch, [L.CHUTE_W / 2, 0.005, 0.012], on_slope(L.STOP_Y, 0.014) - 0.005 * t_, "rubber", quat=q,
        name="end_stop", collide=True, friction=[0.6, 0.005, 0.0001], priority=2, solref=[0.03, 0.7])
    # "Tray full" light barrier near the machine (emitter + receiver on the side walls), and the
    # camera over the pick zone that locates the parts.
    for sx in (-1, 1):
        box(ch, [0.008, 0.012, 0.015], on_slope(L.CHUTE_FULL_Y, 0.03, sx * (L.CHUTE_W / 2 + 0.012)),
            "sensor", quat=q, name=f"full_eye_{'tx' if sx > 0 else 'rx'}")
    # (on a pole at the far side of the tray, away from the robot and the carton)
    cam_p = np.array([-(L.CHUTE_W / 2 + 0.07), L.STOP_Y + 0.12, L.NEST_Z + 0.52])
    rod(ch, [cam_p[0], cam_p[1], 0.0], cam_p + [0, 0, 0.03], 0.012, "steel_dark")
    rod(ch, cam_p + [0, 0, 0.02], cam_p + [0.16, 0, 0.02], 0.008, "steel_dark")
    box(ch, [0.03, 0.022, 0.018], cam_p + [0.17, 0, 0.0], "black")
    ch.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.010, 0.004], pos=cam_p + [0.17, 0, -0.020],
                material="sensor", **VIS)
    # Legs (blue tubes), cross bars and the small shelf seen under the tray in the photo.
    for y in (L.STOP_Y + 0.03, L.MACHINE_FRONT_Y - 0.08):
        for sx in (-1, 1):
            x = sx * (L.CHUTE_W / 2 - 0.02)
            rod(ch, [x, y, 0.0], [x, y, L.chute_z(y) - 0.004], 0.014, "machine_blue")
            ch.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.025, 0.004], pos=[x, y, 0.004], material="steel_dark",
                        **VIS)
        rod(ch, [-(L.CHUTE_W / 2 - 0.02), y, 0.25], [L.CHUTE_W / 2 - 0.02, y, 0.25], 0.010, "machine_blue")
    yl = (L.STOP_Y + 0.03, L.MACHINE_FRONT_Y - 0.08)
    for sx in (-1, 1):
        x = sx * (L.CHUTE_W / 2 - 0.02)
        rod(ch, [x, yl[0], 0.25], [x, yl[1], 0.25], 0.010, "machine_blue")
    box(ch, [L.CHUTE_W / 2 - 0.03, 0.10, 0.004], [0, (yl[0] + yl[1]) / 2, 0.26], "steel_dark")


# --- Carton + layer sheets -------------------------------------------------------------------


def _add_carton(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    cx, cy = L.BOX_CENTER
    ix, iy = L.BOX_IN / 2
    t, H, fz = L.BOX_WALL, L.BOX_H, L.BOX_FLOOR_Z
    # Roller stand.
    st = wb.add_body(name="carton_stand")
    sx_, sy_ = ix + t + 0.04, iy + t + 0.03
    top = fz - t
    for px in (-1, 1):
        for py in (-1, 1):
            rod(st, [cx + px * (sx_ - 0.02), cy + py * (sy_ - 0.02), 0], [cx + px * (sx_ - 0.02), cy + py * (sy_ - 0.02),
                                                                        top - 0.03], 0.016, "steel_dark")
    for py in (-1, 1):
        box(st, [sx_, 0.02, 0.02], [cx, cy + py * (sy_ - 0.02), top - 0.045], "steel_dark")
    for k in range(9):  # rollers across x
        y = cy - sy_ + 0.04 + k * (2 * sy_ - 0.08) / 8
        st.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.016, sx_ - 0.03], pos=[cx, y, top - 0.016],
                    quat=axis_angle_quat([0, 1, 0], np.pi / 2), material="steel", **VIS)
    # Side guide (the carton is pushed against it when loaded).
    box(st, [0.01, sy_, 0.03], [cx + sx_ + 0.01, cy, top + 0.01], "yellow")

    # The carton.
    c = wb.add_body(name="carton", pos=[cx, cy, fz])
    box(c, [ix + t, iy + t, t / 2], [0, 0, -t / 2], "cardboard")
    box(c, [ix + t, iy + t, 0.015], [0, 0, -0.015], name="carton_floor", collide=True, rgba=[0, 0, 0, 0], group=3,
        friction=[L.BOX_FRICTION, 0.005, 0.0001])
    for sx in (-1, 1):
        box(c, [t / 2, iy + t, H / 2], [sx * (ix + t / 2), 0, H / 2], "cardboard",
            name=f"carton_wall_x{'+' if sx > 0 else '-'}", collide=True)
        box(c, [ix, t / 2, H / 2], [0, sx * (iy + t / 2), H / 2], "cardboard",
            name=f"carton_wall_y{'+' if sx > 0 else '-'}", collide=True)
    # Flaps folded outwards and hanging down the sides, slightly splayed.
    flap = ix
    beta = np.deg2rad(18)
    for sx in (-1, 1):
        d = np.array([sx * np.sin(beta), 0, -np.cos(beta)])
        box(c, [iy + t - 0.004, 0.002, flap / 2], [sx * (ix + t + 0.002), 0, H] + d * flap / 2, "cardboard_edge",
            quat=quat_mul(axis_angle_quat([0, 1, 0], np.pi - sx * beta), yaw_quat(np.pi / 2)))
        d = np.array([0, sx * np.sin(beta), -np.cos(beta)])
        box(c, [ix + t - 0.004, 0.002, flap / 2], [0, sx * (iy + t + 0.002), H] + d * flap / 2, "cardboard_edge",
            quat=axis_angle_quat([1, 0, 0], np.pi + sx * beta))
    # Plastic liner bag: inside the walls, over the rim and draped down outside in loose folds.
    lt = 0.0008
    for sx in (-1, 1):
        box(c, [lt, iy, H / 2], [sx * (ix - 0.0012), 0, H / 2], "liner")
        box(c, [ix, lt, H / 2], [0, sx * (iy - 0.0012), H / 2], "liner")
    rng = np.random.default_rng(7)
    for sx in (-1, 1):
        for k in range(5):
            w = (iy + t) / 2.5
            y = -iy + w * (k + 0.5) * 2 * iy / (5 * w)
            tilt = np.deg2rad(rng.uniform(8, 22))
            drop = rng.uniform(0.10, 0.17)
            pos = [sx * (ix + t + 0.012 + np.sin(tilt) * drop / 2), y, H + 0.012 - np.cos(tilt) * drop / 2]
            box(c, [lt, w * 0.62, drop / 2], pos, "liner",
                quat=quat_mul(axis_angle_quat([0, 1, 0], -sx * tilt), axis_angle_quat([1, 0, 0], rng.uniform(-0.15, 0.15))))
        for k in range(4):
            w = (ix + t) / 2
            x = -ix + w * (k + 0.5)
            tilt = np.deg2rad(rng.uniform(8, 22))
            drop = rng.uniform(0.10, 0.17)
            pos = [x, sx * (iy + t + 0.012 + np.sin(tilt) * drop / 2), H + 0.012 - np.cos(tilt) * drop / 2]
            box(c, [w * 0.62, lt, drop / 2], pos, "liner",
                quat=quat_mul(axis_angle_quat([1, 0, 0], sx * tilt), axis_angle_quat([0, 1, 0], rng.uniform(-0.15, 0.15))))
        # Rolled rim of the bag over the wall tops.
        rod(c, [sx * (ix + t / 2), -iy - t, H + 0.004], [sx * (ix + t / 2), iy + t, H + 0.004], 0.009, "liner")
        rod(c, [-ix - t, sx * (iy + t / 2), H + 0.004], [ix + t, sx * (iy + t / 2), H + 0.004], 0.009, "liner")


def _add_magazine(spec: mujoco.MjSpec) -> None:
    """Sheet magazine: tray with corner guides, sheets stacked flat, long side along x."""
    mz = spec.worldbody.add_body(name="magazine")
    cx, cy = L.MAG_CENTER
    hx, hy = L.SHEET[1] / 2, L.SHEET[0] / 2  # rotated: long side along x
    gap = 0.006
    box(mz, [hx + 0.03, hy + 0.03, 0.006], [cx, cy, L.MAG_BASE_Z - 0.006], "steel_dark", name="magazine_tray",
        collide=True)
    for px in (-1, 1):
        for py in (-1, 1):
            rod(mz, [cx + px * (hx - 0.02), cy + py * (hy - 0.02), 0], [cx + px * (hx - 0.02), cy + py * (hy - 0.02),
                                                                      L.MAG_BASE_Z - 0.012], 0.014, "machine_blue")
            # L-shaped corner guide.
            gx, gy = cx + px * (hx + gap), cy + py * (hy + gap)
            box(mz, [0.002, 0.03, 0.04], [gx + px * 0.002, gy - py * 0.026, L.MAG_BASE_Z + 0.04], "steel",
                collide=True)
            box(mz, [0.03, 0.002, 0.04], [gx - px * 0.026, gy + py * 0.002, L.MAG_BASE_Z + 0.04], "steel",
                collide=True)
    for k in range(L.N_SHEETS):
        s = spec.worldbody.add_body(name=f"sheet_{k}", pos=[cx, cy, L.MAG_BASE_Z + (k + 0.5) * L.SHEET_T + 0.0002 * k])
        s.add_freejoint(name=f"sheet_{k}_free")
        s.add_geom(name=f"sheet_{k}_geom", type=GEOM.mjGEOM_BOX, size=[hx, hy, L.SHEET_T / 2], material="sheet",
                   mass=L.SHEET_MASS, friction=[L.BOX_FRICTION, 0.005, 0.0001])
        # Corrugation flutes visible on the edges.
        box(s, [hx, 0.0006, L.SHEET_T / 2 * 0.6], [0, hy, 0], "cardboard_edge")
        box(s, [hx, 0.0006, L.SHEET_T / 2 * 0.6], [0, -hy, 0], "cardboard_edge")


# --- Parts -------------------------------------------------------------------------------


def park_pos(k: int) -> list[float]:
    """Where part k waits (hidden inside the machine) until the machine makes it."""
    col, row, lay = k % 8, (k // 8) % 4, k // 32
    return [(col - 3.5) * 0.09, L.MACHINE_FRONT_Y + 0.40 + row * 0.10, 0.95 + lay * 0.10]


def _add_part(spec: mujoco.MjSpec, k: int) -> None:
    p = spec.worldbody.add_body(name=f"part_{k}", pos=park_pos(k), gravcomp=1)
    p.add_freejoint(name=f"part_{k}_free")
    p.explicitinertial = True
    p.mass = L.PART_MASS
    p.ipos = [0, 0, L.PART_H / 2]
    r2 = L.PART_R**2 + L.PART_BORE_R**2
    p.inertia = [L.PART_MASS * (3 * r2 + L.PART_H**2) / 12] * 2 + [L.PART_MASS * r2 / 2]
    p.add_geom(name=f"part_{k}_vis", type=GEOM.mjGEOM_MESH, meshname="bushing", material="part", **VIS)
    # Collision: a solid convex 24-sided prism for everything (flat ends sit well on the tray,
    # sheets and other parts) ...
    p.add_geom(name=f"part_{k}_col", type=GEOM.mjGEOM_MESH, meshname="bushing_col", group=3,
               rgba=[0.1, 0.1, 0.1, 1], friction=[L.PART_FRICTION, 0.005, 0.0001], contype=3, conaffinity=3)
    # ... and a lining of the bore made of 8 convex segments, which only the jaws touch.
    for j in range(8):
        p.add_geom(name=f"part_{k}_bore{j}", type=GEOM.mjGEOM_MESH, meshname="bore_seg", quat=yaw_quat(j * np.pi / 4),
                   group=3, rgba=[0.3, 0.3, 0.3, 1], friction=[1.0, 0.005, 0.0001], contype=4, conaffinity=4,
                   density=0)
    p.add_site(name=f"part_{k}_top", pos=[0, 0, L.PART_H], size=[0.003, 0, 0], group=4)


def _add_task_constraints(spec: mujoco.MjSpec) -> None:
    weld = mujoco.mjtEq.mjEQ_WELD
    body = mujoco.mjtObj.mjOBJ_BODY
    spec.add_equality(name="grasp", type=weld, objtype=body, name1=ARM + "link_6", name2="part_0",
                      active=False, solref=[0.005, 1])
    spec.add_equality(name="vacuum", type=weld, objtype=body, name1="cup", name2="sheet_0",
                      active=False, solref=[0.005, 1])


def n_parts(layers: int = L.N_LAYERS) -> int:
    return L.GRID[0] * L.GRID[1] * layers


def build_spec(layers: int = L.N_LAYERS) -> mujoco.MjSpec:
    meshgen.generate()
    spec = mujoco.MjSpec()
    spec.modelname = "claret_packing_cell"
    spec.compiler.autolimits = True
    spec.compiler.degree = False  # all angles in this file are radians
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.global_.azimuth = 215
    spec.visual.global_.elevation = -25
    spec.visual.quality.shadowsize = 8192
    spec.visual.map.shadowclip = 2.5  # shadow map covers the whole cell (no speckles on the floor)
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.stat.center = [0.15, 0.35, 0.7]
    spec.stat.extent = 1.2

    _add_assets(spec)
    _add_environment(spec)
    _add_pedestal(spec)
    _add_arm(spec)
    _add_machine(spec)
    _add_chute(spec)
    _add_carton(spec)
    _add_magazine(spec)
    for k in range(n_parts(layers)):
        _add_part(spec, k)
    _add_task_constraints(spec)
    return spec


def build_model(layers: int = L.N_LAYERS) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_spec(layers)
    return spec.compile(), spec


def export_xml(path: Path, layers: int = L.N_LAYERS) -> Path:
    """Write the cell as a standalone MJCF (e.g. for MuJoCo's `simulate` app).

    Mesh paths are written relative to the output file. The task logic (IK, machine,
    vacuum) lives in Python, so the exported model is the static cell.
    """
    path = Path(path).resolve()
    spec = build_spec(layers)
    model = spec.compile()
    qpos, ctrl = model.qpos0.copy(), np.zeros(model.nu)
    for k, q in enumerate(L.ARM_HOME_Q):
        qpos[model.jnt_qposadr[model.joint(f"{ARM}joint{k + 1}").id]] = q
        ctrl[model.actuator(f"{ARM}joint{k + 1}").id] = q
    ctrl[model.actuator(f"{ARM}gripper").id] = L.GRIPPER_OPEN
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)
    spec.compile()
    xml = spec.to_xml()
    for mesh in spec.meshes:  # to_xml() recompiles, so relativize in the text
        xml = xml.replace(f'file="{mesh.file}"', f'file="{os.path.relpath(mesh.file, path.parent)}"')
    path.write_text(xml)
    return path
