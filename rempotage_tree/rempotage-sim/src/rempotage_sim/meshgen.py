"""Procedural 3D assets (binary STL) for the repotting cell.

Run ``rempotage-sim-assets`` (or ``python -m rempotage_sim.meshgen``) to regenerate the meshes in
``assets/meshes``. Every mesh is built from the dimensions in :mod:`rempotage_sim.layout`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rempotage_sim import layout as L

MESH_DIR = Path(__file__).parent / "assets" / "meshes"


# --- Geometry helpers ------------------------------------------------------------


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ccw(poly) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    return poly if _signed_area(poly) > 0 else poly[::-1].copy()


def triangulate(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple CCW polygon."""
    idx = list(range(len(poly)))
    tris: list[tuple[int, int, int]] = []

    def inside(p, a, b, c):
        def cross(o, u, v):
            return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue  # reflex or degenerate corner
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break  # no ear found (numerical trouble): fall back to a fan
    if len(idx) >= 3:
        tris += [(idx[0], idx[j], idx[j + 1]) for j in range(1, len(idx) - 1)]
    return tris


def extrude(poly, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a simple 2D polygon between z0 and z1 into a closed solid."""
    poly = _ccw(poly)
    n = len(poly)
    verts = np.vstack([np.c_[poly, np.full(n, z0)], np.c_[poly, np.full(n, z1)]])
    faces = []
    for a, b, c in triangulate(poly):
        faces.append((n + a, n + b, n + c))  # top, CCW seen from +z
        faces.append((c, b, a))              # bottom
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    return verts, np.array(faces)


def band(polyline, width: float, cap_segments: int = 6) -> np.ndarray:
    """Outline polygon of a thick curve (constant width, rounded ends)."""
    p = np.asarray(polyline, dtype=float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.c_[-t[:, 1], t[:, 0]]
    h = width / 2
    left, right = p + h * nrm, p - h * nrm

    def cap(center, direction, start_normal):
        ang0 = np.arctan2(start_normal[1], start_normal[0])
        turn = start_normal[0] * direction[1] - start_normal[1] * direction[0]
        sweep = np.pi if turn > 0 else -np.pi
        angs = ang0 + np.linspace(0, sweep, cap_segments + 2)[1:-1]
        return center + h * np.c_[np.cos(angs), np.sin(angs)]

    end_cap = cap(p[-1], t[-1], nrm[-1])
    start_cap = cap(p[0], -t[0], -nrm[0])
    return np.vstack([left, end_cap, right[::-1], start_cap])


def smooth_polyline(ctrl, samples: int = 40) -> np.ndarray:
    """Catmull-Rom spline through control points."""
    c = np.asarray(ctrl, dtype=float)
    c = np.vstack([2 * c[0] - c[1], c, 2 * c[-1] - c[-2]])
    out = []
    per = max(2, samples // (len(c) - 3))
    for i in range(1, len(c) - 2):
        p0, p1, p2, p3 = c[i - 1], c[i], c[i + 1], c[i + 2]
        for s in np.linspace(0, 1, per, endpoint=False):
            s2, s3 = s * s, s * s * s
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    out.append(c[-2])
    return np.array(out)


def revolve(profile, segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Revolve a closed (r, z) profile (CCW in the r-z plane) about the z axis."""
    prof = np.asarray(profile, dtype=float)
    m = len(prof)
    angs = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = np.array([[r * np.cos(a), r * np.sin(a), z] for a in angs for r, z in prof])
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        for i in range(m):
            j = (i + 1) % m
            a, b = s * m + i, s * m + j
            c, d = s2 * m + j, s2 * m + i
            faces.append((a, c, b))  # outward normals for a CCW profile
            faces.append((a, d, c))
    return verts, np.array(faces)


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    keep = area > 1e-14  # drop degenerate faces (e.g. on the revolution axis)
    tri, nrm = tri[keep], nrm[keep] / area[keep, None]
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(b"rempotage_sim procedural mesh".ljust(80, b" "))
        f.write(np.uint32(len(rec)).tobytes())
        f.write(rec.tobytes())


# --- Pot, soil, root ball ----------------------------------------------------------------------
POT_BOTTOM_T = L.POT_WALL + 0.003       # pot floor thickness
TOP_RING_T = L.TAMP_MAX + 0.004         # top soil ring (the part the fingertips firm), thickness
SOIL_LAYERS = (POT_BOTTOM_T, L.HOLE_FLOOR_Z, L.HOLE_FLOOR_Z + 0.019, L.HOLE_FLOOR_Z + 0.038,
               L.HOLE_FLOOR_Z + 0.057, L.SOIL_Z - TOP_RING_T)


def pot_inner_radius(z: float) -> float:
    return L.pot_outer_radius(z) - L.POT_WALL


def hole_radius(z: float) -> float:
    """Planting hole radius at height z above the pot bottom."""
    f = (z - L.HOLE_FLOOR_Z) / L.HOLE_DEPTH
    return L.HOLE_R_BOT + (L.HOLE_R_TOP - L.HOLE_R_BOT) * f


def pot_profile() -> list[tuple[float, float]]:
    return [
        (0.0, 0.0), (L.POT_R_BOT - 0.004, 0.0), (L.POT_R_BOT, 0.004),
        (L.POT_R_TOP, L.POT_H - L.POT_LIP_H), (L.POT_LIP_R, L.POT_H - L.POT_LIP_H),
        (L.POT_LIP_R, L.POT_H), (L.POT_R_TOP - L.POT_WALL, L.POT_H),
        (L.POT_R_BOT - L.POT_WALL, POT_BOTTOM_T), (0.0, POT_BOTTOM_T),
    ]


def sqpot_profile() -> list[tuple[float, float]]:
    """Square nursery pot, revolved with 4 segments (radii are half diagonals; turn it 45 deg)."""
    d = np.sqrt(2.0)
    b, t, w, tb = L.SQPOT_BOT * d, L.SQPOT_TOP * d, 0.0015 * d, 0.0015
    return [(0.0, 0.0), (b, 0.0), (t, L.SQPOT_H), (t - w, L.SQPOT_H), (b - w, tb), (0.0, tb)]


def soil_layer(z0: float, z1: float, holed: bool) -> list[tuple[float, float]]:
    """(r, z) profile of a soil slab between z0 and z1, following the pot taper (0.3 mm clear)."""
    r0, r1 = pot_inner_radius(z0) - 0.0003, pot_inner_radius(z1) - 0.0003
    if not holed:
        return [(0.0, z0), (r0, z0), (r1, z1), (0.0, z1)]
    return [(hole_radius(z0), z0), (r0, z0), (r1, z1), (hole_radius(z1), z1)]


def ring_sector(r_in: float, r_out: float, a0: float, a1: float, z0: float, z1: float, n: int = 8):
    """Annular sector prism (one of the top soil ring pieces that the fingertips firm)."""
    a = np.linspace(a0, a1, n)
    outline = np.vstack([np.c_[r_out * np.cos(a), r_out * np.sin(a)],
                         np.c_[r_in * np.cos(a[::-1]), r_in * np.sin(a[::-1])]])
    return extrude(outline, z0, z1)


def rootball_profile() -> list[tuple[float, float]]:
    """Root ball, origin at its bottom centre. Slightly rounded bottom edge."""
    return [(0.0, 0.0), (L.RB_R_BOT - 0.004, 0.0), (L.RB_R_BOT, 0.004), (L.RB_R_TOP, L.RB_H), (0.0, L.RB_H)]


def dibber_profile() -> list[tuple[float, float]]:
    """Dibber tool: the planting hole's shape, origin at its tip."""
    return [(0.0, 0.0), (L.HOLE_R_BOT - 0.004, 0.0), (L.HOLE_R_BOT, 0.004), (L.HOLE_R_TOP, L.HOLE_DEPTH),
            (L.HOLE_R_TOP + 0.004, L.HOLE_DEPTH + 0.006), (0.0, L.HOLE_DEPTH + 0.006)]


def wheel_profile() -> list[tuple[float, float]]:
    return [(L.WHEEL_R_IN, -L.WHEEL_T), (L.WHEEL_R, -L.WHEEL_T), (L.WHEEL_R, 0.0), (L.WHEEL_R_IN, 0.0)]


def leaf_outline(length: float = 0.05, width: float = 0.022, n: int = 14) -> np.ndarray:
    t = np.linspace(0, 1, n)
    half = 0.5 * width * np.sin(np.pi * t) ** 0.8 * (1 - 0.35 * t)
    top = np.c_[t * length, half]
    bot = np.c_[t * length, -half][::-1][1:-1]
    return np.vstack([top, bot])



def generate(out_dir: Path = MESH_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "pot": revolve(pot_profile(), 64),
        "sqpot": revolve(sqpot_profile(), 4),
        "rootball": revolve(rootball_profile(), 40),
        "dibber": revolve(dibber_profile(), 40),
        "wheel": revolve(wheel_profile(), 128),
        "leaf": extrude(leaf_outline(0.06, 0.026), -0.0004, 0.0004),
        "plug": revolve(soil_layer(L.SOIL_Z - TOP_RING_T, L.SOIL_Z, False), 40),
    }
    for j in range(len(SOIL_LAYERS) - 1):
        z0, z1 = SOIL_LAYERS[j], SOIL_LAYERS[j + 1]
        jobs[f"soil_{j}"] = revolve(soil_layer(z0, z1, False), 48)
        if j > 0:
            jobs[f"soil_{j}_holed"] = revolve(soil_layer(z0, z1, True), 48)
    z0, z1 = L.SOIL_Z - TOP_RING_T, L.SOIL_Z
    r_out = pot_inner_radius(z1) - 0.0005
    da = 2 * np.pi / L.TAMP_SECTORS
    for j in range(L.TAMP_SECTORS):
        jobs[f"ring_{j}"] = ring_sector(L.HOLE_R_TOP, r_out, j * da, (j + 1) * da, z0, z1)
    paths = []
    for name, (v, f) in jobs.items():
        p = out_dir / f"{name}.stl"
        write_stl(p, v, f)
        paths.append(p)
    return paths


def main() -> None:
    for p in generate():
        print("wrote", p)


if __name__ == "__main__":
    main()
