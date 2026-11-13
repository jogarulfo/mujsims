"""Builds the MuJoCo model of the Accuwat rack-filling cell with MjSpec.

Cell contents, after the photos of the Accuwat setup:
  - a light grey worktop;
  - one I2RT YAM arm (prefix ``r/``) bolted on the table. Its linear gripper is replaced by an
    adapter, a stand-off and a 20 mm holding electromagnet;
  - two open cardboard boxes of 21700 cells, 10 x 10 cells standing in rows with cardboard
    dividers: one with the cells + up (left, +y), one with the cells - up (right, -y);
  - the rack: a base plate the cells stand on, and a light grey holder plate with 15 x 8
    staggered holes on spacers. Tape and +/- marks on the front edge show the three blocks
    (+ up, - up, + up).

Collision groups (contype / conaffinity):
  cells        2 / 1   cells never touch each other (in the boxes and in the rack they stand
                       apart; this keeps the 200 cells cheap)
  cell world   1 / 2   table, boxes, rack: they hold the cells and the arm hits them
  arm          1 / 1   (YAM defaults) hits the cells and the cell world

Equality constraint switched at runtime (inactive at start, second body set when it is
switched on, see constraints.retarget):
  magnet   weld  r/link_6 <-> cell   the electromagnet holding a cell
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from accuwat_sim import layout as L
from accuwat_sim import meshgen

ASSETS = Path(__file__).parent / "assets"
YAM_XML = ASSETS / "yam" / "yam_linear_4310.xml"
ARM = "r/"

GEOM = mujoco.mjtGeom
JNT = mujoco.mjtJoint

VIS = {"contype": 0, "conaffinity": 0, "group": 2, "density": 0}
ENV = {"contype": 1, "conaffinity": 2, "group": 3}
CELL_COL = {"contype": 2, "conaffinity": 1, "group": 3}

COLORS = {
    "worktop": [0.66, 0.67, 0.66, 1],
    "table_frame": [0.30, 0.31, 0.33, 1],
    "steel": [0.72, 0.74, 0.76, 1],
    "steel_dark": [0.42, 0.44, 0.47, 1],
    "alu": [0.80, 0.81, 0.83, 1],
    "zinc": [0.78, 0.79, 0.74, 1],
    "black": [0.05, 0.05, 0.05, 1],
    "holder": [0.80, 0.80, 0.78, 1],
    "cardboard": [0.72, 0.56, 0.38, 1],
    "cardboard_in": [0.80, 0.66, 0.48, 1],
    "cardboard_edge": [0.64, 0.49, 0.32, 1],
    "label": [0.95, 0.95, 0.93, 1],
    "plus_red": [0.85, 0.10, 0.10, 1],
    "minus_black": [0.08, 0.08, 0.10, 1],
    "minus_blue": [0.10, 0.25, 0.75, 1],
    "wrap": [0.20, 0.55, 0.85, 1],      # blue PVC sleeve, as on the cells of the photo
    "insulator": [0.90, 0.90, 0.86, 1],
    "terminal": [0.66, 0.67, 0.70, 1],
    "wire_red": [0.80, 0.08, 0.08, 1],
    "wire_black": [0.06, 0.06, 0.06, 1],
}


def axis_angle_quat(axis, angle: float) -> list[float]:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return [np.cos(angle / 2), *(np.sin(angle / 2) * axis)]


def yaw_quat(yaw: float) -> list[float]:
    return axis_angle_quat([0, 0, 1], yaw)


def look_at(pos, target) -> list[float]:
    """Camera xyaxes looking from pos at target with world z up."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, [0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return [*x, *y]


def box(body, size, pos, material=None, quat=None, name=None, collide=False, **kw):
    """Shorthand for a box geom: visual only unless collide=True (then in the cell world group)."""
    args = {"type": GEOM.mjGEOM_BOX, "size": list(size), "pos": list(pos)}
    if quat is not None:
        args["quat"] = list(quat)
    if name:
        args["name"] = name
    if material:
        args["material"] = material
    args.update(ENV if collide else VIS)
    args.update(kw)
    return body.add_geom(**args)


def cyl(body, r, z0, z1, material, xy=(0.0, 0.0), collide=False, name=None, **kw):
    """Vertical cylinder from z0 to z1 at xy."""
    args = {"type": GEOM.mjGEOM_CYLINDER, "size": [r, (z1 - z0) / 2, 0], "pos": [xy[0], xy[1], (z0 + z1) / 2],
            "material": material}
    if name:
        args["name"] = name
    args.update(ENV if collide else VIS)
    args.update(kw)
    return body.add_geom(**args)


def rod(body, a, b, r, material, **kw):
    args = {"type": GEOM.mjGEOM_CAPSULE, "size": [r, 0, 0], "fromto": [*a, *b], "material": material}
    args.update(VIS)
    args.update(kw)
    return body.add_geom(**args)


def plus_sign(body, center, size, material, normal_axis: int = 2, t=0.0006):
    """A '+' made of two bars, lying in the plane normal to `normal_axis` (0: x, 2: z)."""
    c = np.asarray(center, float)
    w = size * 0.18
    if normal_axis == 2:
        box(body, [size / 2, w / 2, t], c, material)
        box(body, [w / 2, size / 2, t], c, material)
    else:
        box(body, [t, size / 2, w / 2], c, material)
        box(body, [t, w / 2, size / 2], c, material)


def minus_sign(body, center, size, material, normal_axis: int = 2, t=0.0006):
    c = np.asarray(center, float)
    w = size * 0.18
    if normal_axis == 2:
        box(body, [w / 2, size / 2, t], c, material)    # along y (reads as '-' from the front, +x)
    else:
        box(body, [t, size / 2, w / 2], c, material)


# --- Assets / environment ----------------------------------------------------------------


def _add_assets(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.66, 0.70], rgb2=[0.16, 0.17, 0.19], width=512, height=3072)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                     rgb1=[0.46, 0.47, 0.49], rgb2=[0.43, 0.44, 0.46], markrgb=[0.38, 0.39, 0.41],
                     width=300, height=300)
    spec.add_material(name="floor", textures=["", "floor"], texuniform=True, texrepeat=[3, 3],
                      reflectance=0.0, specular=0.05, shininess=0.1)
    shiny = ("steel", "steel_dark", "alu", "zinc", "terminal")
    for name, rgba in COLORS.items():
        spec.add_material(name=name, rgba=rgba, specular=0.3 if name in shiny else 0.08,
                          shininess=0.5 if name in shiny else 0.2)
    spec.add_material(name="led", rgba=[1, 1, 1, 1], emission=0.8)
    for mesh in ("holder", "hole_seg"):
        spec.add_mesh(name=mesh, file=str(meshgen.MESH_DIR / f"{mesh}.stl"))


def _add_environment(spec: mujoco.MjSpec) -> None:
    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.8, 0.5, 3.0], dir=[-0.25, -0.15, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=True, diffuse=[0.50, 0.50, 0.48], specular=[0.2, 0.2, 0.2])
    wb.add_light(name="fill", pos=[2.0, -1.5, 2.2], dir=[-0.6, 0.5, -0.6], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 castshadow=False, diffuse=[0.20, 0.20, 0.22])
    wb.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[4, 4, 0.05], material="floor")

    rack = np.array([*L.RACK_CENTER, L.RACK_BASE_Z])
    bp = np.array([*L.BOX_PLUS_CENTER, L.BOX_FLOOR_Z])
    bm = np.array([*L.BOX_MINUS_CENTER, L.BOX_FLOOR_Z])
    cams = {
        "overview": ([1.35, -0.95, 1.55], [0.22, 0.0, 0.80], 45),
        "photo": ([1.15, 0.12, 1.22], [0.22, 0.0, 0.90], 50),          # roughly the photo's viewpoint (main view)
        "front": ([1.20, 0.0, 1.30], [0.20, 0.0, 0.80], 50),
        "side": ([0.35, -1.25, 1.20], [0.25, 0.0, 0.82], 45),
        "rack": (rack + [0.38, -0.30, 0.34], rack + [0.0, 0.0, 0.03], 50),
        "rack_top": (rack + [0.001, 0.0, 0.62], rack, 40),
        "rack_close": (rack + [0.22, -0.16, 0.20], rack + [0.0, 0.0, 0.05], 45),
        "box_plus": (bp + [0.40, 0.18, 0.40], bp + [0.0, 0.0, 0.05], 45),
        "box_minus": (bm + [0.40, -0.18, 0.40], bm + [0.0, 0.0, 0.05], 45),
    }
    for k, (c0, c1, _) in enumerate(L.BLOCKS):   # a close-up of each block, from the operator side
        y = (L.rack_hole_xy(0, c0)[1] + L.rack_hole_xy(1, c1 - 1)[1]) / 2
        tgt = np.array([L.RACK_CENTER[0] + 0.01, y, L.RACK_BASE_Z + 0.05])
        cams[f"block{k + 1}"] = (tgt + [0.30, -0.10 if y >= 0 else 0.10, 0.24], tgt, 45)
    for name, (pos, target, fovy) in cams.items():
        wb.add_camera(name=name, pos=list(pos), xyaxes=look_at(pos, target), fovy=fovy)


def _add_table(spec: mujoco.MjSpec) -> None:
    t = spec.worldbody.add_body(name="table")
    hx, hy = L.TABLE_SIZE / 2
    cx, cy = L.TABLE_CENTER
    th = 0.03
    box(t, [hx, hy, th / 2], [cx, cy, L.TABLE_Z - th / 2], "worktop", name="worktop", collide=True)
    box(t, [hx, hy, th / 2], [cx, cy, L.TABLE_Z - th / 2], "worktop")
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * (hx - 0.05), cy + sy * (hy - 0.05)
            box(t, [0.025, 0.025, (L.TABLE_Z - th) / 2], [x, y, (L.TABLE_Z - th) / 2], "table_frame")
    for sy in (-1, 1):
        box(t, [hx - 0.05, 0.02, 0.04], [cx, cy + sy * (hy - 0.05), L.TABLE_Z - th - 0.04], "table_frame")
    for sx in (-1, 1):
        box(t, [0.02, hy - 0.05, 0.04], [cx + sx * (hx - 0.05), cy, L.TABLE_Z - th - 0.04], "table_frame")


# --- Robot + electromagnet ----------------------------------------------------------------


def _strip_gripper(yam: mujoco.MjSpec) -> None:
    """Remove the linear_4310 gripper (fingers, housing, actuator) and the wrist camera."""
    while yam.keys:
        yam.delete(yam.keys[0])
    for ex in list(yam.excludes):
        if {ex.bodyname1, ex.bodyname2} & {"left_finger", "right_finger", "left_pad", "right_pad", "wrist_camera"}:
            yam.delete(ex)
    for eq in list(yam.equalities):
        yam.delete(eq)
    yam.delete(yam.actuator("gripper"))
    for name in ("left_finger", "right_finger", "wrist_camera"):
        yam.delete(yam.body(name))
    link6 = yam.body("link_6")
    for g in list(link6.geoms):
        if g.name == "gripper_housing" or g.meshname == "gripper":
            yam.delete(g)
    for mesh in list(yam.meshes):
        if mesh.name in ("gripper", "tip_left", "tip_right", "d405_wrist_linear_4310"):
            yam.delete(mesh)


def _add_arm(spec: mujoco.MjSpec) -> None:
    yam = mujoco.MjSpec.from_file(str(YAM_XML))
    for mesh in yam.meshes:  # make mesh paths independent of the parent's meshdir
        mesh.file = str((YAM_XML.parent / "assets" / mesh.file).resolve())
    yam.meshdir = ""
    yam.compiler.meshdir = ""
    _strip_gripper(yam)
    # Steel riser block bolted on the table (welded tube with top and bottom plates).
    mount = spec.worldbody.add_body(name="arm_mount")
    h = L.RISER_H
    box(mount, [0.07, 0.07, h / 2], [0, 0, L.TABLE_Z + h / 2], "steel_dark", name="arm_riser", collide=True)
    box(mount, [0.06, 0.06, (h - 0.02) / 2], [0, 0, L.TABLE_Z + h / 2], "steel_dark")
    for z in (L.TABLE_Z + 0.005, L.BASE_Z - 0.005):
        box(mount, [0.075, 0.075, 0.005], [0, 0, z], "steel")
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(mount, 0.005, L.BASE_Z, L.BASE_Z + 0.004, "black", xy=(0.06 * sx, 0.06 * sy))
            cyl(mount, 0.005, L.TABLE_Z + 0.01, L.TABLE_Z + 0.014, "black", xy=(0.065 * sx, 0.065 * sy))
    frame = spec.worldbody.add_frame(pos=[0, 0, L.BASE_Z])
    spec.attach(yam, prefix=ARM, frame=frame)
    spec.body(ARM + "link_1").sleep = mujoco.mjtSleepPolicy.mjSLEEP_NEVER  # the arm is always under control
    _add_magnet_tool(spec)


def _add_magnet_tool(spec: mujoco.MjSpec) -> None:
    """Adapter disc + stand-off + holding electromagnet on the wrist flange (link_6 frame:
    z along the approach, flange face at FLANGE_Z)."""
    l6 = spec.body(ARM + "link_6")
    z0 = L.FLANGE_Z
    z_ad = z0 + L.ADAPTER_T
    z_mag = L.MAGNET_FACE_Z - L.MAGNET_H
    face = L.MAGNET_FACE_Z
    arm_col = {"contype": 1, "conaffinity": 1, "group": 3}
    cyl(l6, L.ADAPTER_R, z0, z_ad, "alu", name=ARM + "adapter", **arm_col)
    cyl(l6, L.ADAPTER_R, z0, z_ad, "alu")
    for k in range(4):  # cap screws
        a = np.pi / 4 + k * np.pi / 2
        cyl(l6, 0.003, z_ad, z_ad + 0.002, "black", xy=(0.022 * np.cos(a), 0.022 * np.sin(a)))
    cyl(l6, L.STEM_R, z_ad, z_mag, "steel")
    cyl(l6, 0.011, z_mag - 0.006, z_mag, "steel_dark")                   # lock nut on the magnet's thread
    cyl(l6, L.MAGNET_R, z_mag, face, "zinc", name=ARM + "magnet", **arm_col, friction=[0.6, 0.005, 0.0001])
    cyl(l6, L.MAGNET_R, z_mag, face - 0.0003, "zinc")
    cyl(l6, L.MAGNET_R - 0.0005, face - 0.0006, face - 0.0001, "black")  # potting ring on the face
    cyl(l6, 0.0042, face - 0.0006, face, "steel")                        # pole
    # Coil leads from the magnet up along the stand-off to the wrist.
    for k, mat in enumerate(("wire_red", "wire_black")):
        dx = 0.0025 * (1 - 2 * k)
        rod(l6, [dx, L.MAGNET_R - 0.002, z_mag + 0.003], [dx, L.STEM_R + 0.002, z_mag - 0.010], 0.0012, mat)
        rod(l6, [dx, L.STEM_R + 0.002, z_mag - 0.010], [dx, L.STEM_R + 0.002, z_ad + 0.002], 0.0012, mat)
        rod(l6, [dx, L.STEM_R + 0.002, z_ad + 0.002], [dx, L.ADAPTER_R - 0.004, z0 - 0.004], 0.0012, mat)
    # "Magnet on" LED on the adapter.
    l6.add_geom(name="magnet_led", type=GEOM.mjGEOM_SPHERE, size=[0.003, 0, 0], pos=[0, -(L.ADAPTER_R - 0.006), z_ad + 0.0015],
                material="led", rgba=[0.25, 0.08, 0.08, 1], **VIS)
    l6.add_site(name=ARM + "magnet_site", pos=[0, 0, face], quat=[1, 0, 0, -1], size=[0.003, 0, 0], group=4)


# --- Cell boxes ----------------------------------------------------------------------------


def _add_cell_box(spec: mujoco.MjSpec, name: str, center, polarity: int) -> None:
    wb = spec.worldbody
    cx, cy = center
    ix, iy = L.BOX_IN / 2
    t, H = L.BOX_WALL, L.BOX_H
    b = wb.add_body(name=name, pos=[cx, cy, L.BOX_FLOOR_Z])
    box(b, [ix + t, iy + t, L.BOX_FLOOR_T / 2], [0, 0, -L.BOX_FLOOR_T / 2], "cardboard", name=f"{name}_floor",
        collide=True, friction=[L.BOX_FRICTION, 0.005, 0.0001])
    box(b, [ix + t, iy + t, L.BOX_FLOOR_T / 2], [0, 0, -L.BOX_FLOOR_T / 2], "cardboard_in")
    for sx in (-1, 1):
        box(b, [t / 2, iy + t, H / 2], [sx * (ix + t / 2), 0, H / 2], "cardboard", name=f"{name}_wall_{'xp' if sx > 0 else 'xm'}",
            collide=True)
        box(b, [t / 2, iy + t, H / 2], [sx * (ix + t / 2), 0, H / 2], "cardboard")
        box(b, [ix, t / 2, H / 2], [0, sx * (iy + t / 2), H / 2], "cardboard", collide=True)
        box(b, [ix, t / 2, H / 2], [0, sx * (iy + t / 2), H / 2], "cardboard")
    # Lid removed (as in the photo): a rolled top edge on the walls.
    for sx in (-1, 1):
        box(b, [t / 2 + 0.0005, iy + t, 0.002], [sx * (ix + t / 2), 0, H - 0.002], "cardboard_edge")
        box(b, [ix, t / 2 + 0.0005, 0.002], [0, sx * (iy + t / 2), H - 0.002], "cardboard_edge")
    # Dividers between the rows (cardboard strips along y, just below the cell tops).
    for r in range(1, L.BOX_GRID[0]):
        x = (L.BOX_GRID[0] / 2 - r) * L.BOX_PITCH_X
        box(b, [0.0005, iy, 0.029], [x, 0, 0.029], "cardboard_in")
    # Label on the front wall (+x side, facing the camera): white, with a big + or - and a
    # coloured band. The terminal on top is the one printed.
    x = ix + t + 0.0008
    box(b, [0.0006, 0.07, 0.024], [x, 0, H * 0.52], "label")
    band = "plus_red" if polarity > 0 else "minus_blue"
    box(b, [0.0007, 0.07, 0.004], [x + 0.0001, 0, H * 0.52 - 0.020], band)
    mark = (plus_sign if polarity > 0 else minus_sign)
    mark(b, [x + 0.0002, -0.035, H * 0.52 + 0.002], 0.026, band, normal_axis=0, t=0.0007)
    for k in range(3):  # "text" lines
        box(b, [0.0007, 0.032, 0.0018], [x + 0.0001, 0.022, H * 0.52 + 0.010 - k * 0.008], "black")


# --- Rack ---------------------------------------------------------------------------------------


def _add_rack(spec: mujoco.MjSpec) -> None:
    rk = spec.worldbody.add_body(name="rack", pos=[*L.RACK_CENTER, 0.0])
    px, py = L.RACK_PLATE / 2
    front_extra = 0.03                      # the base plate sticks out in front (+x), with the +/- marks
    bx0, bx1 = -px - 0.012, px + 0.012 + front_extra
    by = py + 0.012
    zb = L.RACK_BASE_Z
    box(rk, [(bx1 - bx0) / 2, by, L.RACK_BASE_T / 2], [(bx0 + bx1) / 2, 0, zb - L.RACK_BASE_T / 2], "alu",
        name="rack_base", collide=True, friction=[L.RACK_FRICTION, 0.005, 0.0001])
    box(rk, [(bx1 - bx0) / 2, by, L.RACK_BASE_T / 2], [(bx0 + bx1) / 2, 0, zb - L.RACK_BASE_T / 2], "alu")
    zt = L.HOLDER_TOP
    zh = zt - L.HOLDER_T
    for sx in (-1, 0, 1):                   # spacers between the base and the holder
        for sy in (-1, 1):
            xy = (sx * (px - 0.007), sy * (py - 0.007))
            cyl(rk, 0.005, zb, zh, "steel", xy=xy, collide=True)
            cyl(rk, 0.005, zb, zh, "steel", xy=xy)
            cyl(rk, 0.0045, zt, zt + 0.002, "steel", xy=xy)        # screw heads
    rk.add_geom(name="holder_vis", type=GEOM.mjGEOM_MESH, meshname="holder", pos=[0, 0, zh], material="holder", **VIS)
    # Collision of the holes: 6 convex segments each (the plate between them never touches anything).
    # Their flat outer faces look at the neighbouring holes (30 + k * 60 deg), so the segment corners
    # sit in the webs between three holes and never reach into a neighbour.
    for r in range(L.RACK_ROWS):
        for c in range(L.RACK_COLS):
            x, y = L.rack_hole_xy(r, c) - L.RACK_CENTER
            for k in range(6):
                rk.add_geom(type=GEOM.mjGEOM_MESH, meshname="hole_seg", pos=[x, y, zh],
                            quat=yaw_quat(np.pi / 6 + k * np.pi / 3),
                            friction=[L.RACK_FRICTION, 0.005, 0.0001], rgba=[0.6, 0.6, 0.6, 1], **ENV)
    # Front edge: a tape strip per block and its +/- mark on the base plate.
    for c0, c1, pol in L.BLOCKS:
        ya = L.rack_hole_xy(0, c0)[1] - L.RACK_CENTER[1] + L.RACK_PITCH / 2 - 0.002
        yb = L.rack_hole_xy(1, c1 - 1)[1] - L.RACK_CENTER[1] - L.RACK_PITCH / 2 + 0.002
        mat = "plus_red" if pol > 0 else "minus_black"
        box(rk, [0.004, (ya - yb) / 2, 0.0003], [px - 0.0045, (ya + yb) / 2, zt + 0.0003], mat)
        mark = plus_sign if pol > 0 else minus_sign
        mark(rk, [px + 0.012 + front_extra / 2, (ya + yb) / 2, zb + 0.0004], 0.018, mat)


# --- Cells --------------------------------------------------------------------------------------


def _add_cell(spec: mujoco.MjSpec, k: int, xy, polarity: int) -> None:
    """A 21700 cell standing in its box. Body frame at the cell centre, +z towards the positive
    terminal (so a '- up' cell is turned upside down)."""
    h = L.CELL_H / 2
    quat = [1, 0, 0, 0] if polarity > 0 else [0, 1, 0, 0]
    c = spec.worldbody.add_body(name=f"cell_{k}", pos=[xy[0], xy[1], L.BOX_FLOOR_Z + h], quat=quat)
    c.add_freejoint(name=f"cell_{k}_free")
    c.explicitinertial = True
    c.mass = L.CELL_MASS
    c.ipos, c.iquat = [0, 0, 0], [1, 0, 0, 0]
    c.inertia = [L.CELL_MASS * (3 * L.CELL_R**2 + L.CELL_H**2) / 12] * 2 + [L.CELL_MASS * L.CELL_R**2 / 2]
    c.add_geom(name=f"cell_{k}", type=GEOM.mjGEOM_CYLINDER, size=[L.CELL_R, h, 0], rgba=[0.2, 0.5, 0.8, 1],
               friction=[L.CELL_FRICTION, 0.005, 0.0001], **CELL_COL)
    # Sleeve, positive end (white insulating ring + button cap), negative end (flat can bottom).
    cyl(c, L.CELL_R, -h + 0.0004, h - 0.0008, "wrap")
    cyl(c, 0.0092, h - 0.0010, h - 0.0005, "insulator")
    cyl(c, 0.0064, h - 0.0010, h, "terminal")
    cyl(c, 0.0030, h - 0.0001, h + 0.00005, "steel")
    cyl(c, 0.0097, -h, -h + 0.0006, "terminal")


def _add_task_constraints(spec: mujoco.MjSpec) -> None:
    spec.add_equality(name="magnet", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1=ARM + "link_6", name2="cell_0", active=False, solref=[0.005, 1])


def box_cells() -> list[tuple[str, int, int, int]]:
    """(box, row, col, polarity) of every cell, in body order: the + box first."""
    out = []
    for name, pol in (("box_plus", +1), ("box_minus", -1)):
        for r in range(L.BOX_GRID[0]):
            for c in range(L.BOX_GRID[1]):
                out.append((name, r, c, pol))
    return out


def build_spec() -> mujoco.MjSpec:
    meshgen.generate()
    spec = mujoco.MjSpec()
    spec.modelname = "accuwat_rack_cell"
    spec.compiler.autolimits = True
    spec.compiler.degree = False  # all angles in this file are radians
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.global_.azimuth = 200
    spec.visual.global_.elevation = -30
    spec.visual.quality.shadowsize = 8192
    spec.visual.map.shadowclip = 1.8
    spec.visual.headlight.ambient = [0.16, 0.16, 0.16]
    spec.visual.headlight.diffuse = [0.16, 0.16, 0.16]
    spec.stat.center = [0.25, 0.0, 0.8]
    spec.stat.extent = 0.9

    _add_assets(spec)
    _add_environment(spec)
    _add_table(spec)
    _add_arm(spec)
    _add_cell_box(spec, "box_plus", L.BOX_PLUS_CENTER, +1)
    _add_cell_box(spec, "box_minus", L.BOX_MINUS_CENTER, -1)
    _add_rack(spec)
    for k, (name, r, c, pol) in enumerate(box_cells()):
        center = L.BOX_PLUS_CENTER if pol > 0 else L.BOX_MINUS_CENTER
        _add_cell(spec, k, L.box_cell_xy(center, r, c), pol)
    _add_task_constraints(spec)
    return spec


def build_model() -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_spec()
    return spec.compile(), spec


def export_xml(path: Path) -> Path:
    """Write the cell as a standalone MJCF (e.g. for MuJoCo's `simulate` app).

    Mesh paths are written relative to the output file. The task logic (IK, magnet) lives in
    Python, so the exported model is the static cell.
    """
    path = Path(path).resolve()
    spec = build_spec()
    model = spec.compile()
    qpos, ctrl = model.qpos0.copy(), np.zeros(model.nu)
    for k, q in enumerate(L.ARM_HOME_Q):
        qpos[model.jnt_qposadr[model.joint(f"{ARM}joint{k + 1}").id]] = q
        ctrl[model.actuator(f"{ARM}joint{k + 1}").id] = q
    spec.add_key(name="home", qpos=qpos, ctrl=ctrl)
    spec.compile()
    xml = spec.to_xml()
    for mesh in spec.meshes:  # to_xml() recompiles, so relativize in the text
        xml = xml.replace(f'file="{mesh.file}"', f'file="{os.path.relpath(mesh.file, path.parent)}"')
    path.write_text(xml)
    return path
