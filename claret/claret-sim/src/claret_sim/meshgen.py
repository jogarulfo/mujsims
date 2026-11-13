"""Procedural 3D assets (binary STL) for the Claret packing cell.

Run ``claret-sim-assets`` (or ``python -m claret_sim.meshgen``) to regenerate the meshes in
``assets/meshes``. Every mesh is built from the dimensions in :mod:`claret_sim.layout`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from claret_sim import layout as L

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
        f.write(b"claret_sim procedural mesh".ljust(80, b" "))
        f.write(np.uint32(len(rec)).tobytes())
        f.write(rec.tobytes())


# --- Cell meshes ---------------------------------------------------------------------


def bushing_profile(ribs: int = 4) -> list[tuple[float, float]]:
    """Half cross-section (r, z) of the ribbed bushing, CCW in the r-z plane."""
    R, Rr, Rb, H = L.PART_R, L.PART_RIB_R, L.PART_BORE_R, L.PART_H
    c = 0.002  # chamfer
    outer = [(Rb + c, 0.0), (R - c, 0.0), (R, c)]
    band_h = (H - 0.016) / ribs
    for k in range(ribs):  # moulded ribs on the outside
        z0 = 0.008 + k * band_h
        outer += [(R, z0 + 0.15 * band_h), (Rr, z0 + 0.3 * band_h), (Rr, z0 + 0.7 * band_h),
                  (R, z0 + 0.85 * band_h)]
    outer += [(R, H - c), (R - c, H), (Rb + 0.006, H), (Rb + 0.004, H - 0.003)]  # top face + collar
    bore = [(Rb + 0.002, H - 0.004), (Rb, H - 0.008), (Rb, 0.006), (Rb + c, 0.002)]
    return outer + bore


def bore_segment(n: int = 8, wall: float = 0.007) -> list[tuple[float, float]]:
    """45 deg ring sector lining the bore (polygon in the xy plane). Its flat inner face sits
    on the bore radius, so a jaw pressing on it stops where the visual bore is."""
    half = np.pi / n
    r_in = (L.PART_BORE_R - 0.0002) / np.cos(half)
    r_out = L.PART_BORE_R + wall
    pts = [(r_in * np.cos(-half), r_in * np.sin(-half))]
    pts += [(r_out * np.cos(a), r_out * np.sin(a)) for a in (-half, 0.0, half)]
    pts += [(r_in * np.cos(half), r_in * np.sin(half))]
    return pts


def chamfered(mesh, c: float):
    """Move the top inner edge of a bore segment outwards by c (lead-in chamfer)."""
    v, f = mesh
    v = v.copy()
    r = np.hypot(v[:, 0], v[:, 1])
    top = v[:, 2] > v[:, 2].max() - 1e-9
    inner = r < r.min() + 1e-6
    sel = top & inner
    v[sel, :2] *= ((r[sel] + c) / r[sel])[:, None]
    return v, f


def cup_profile() -> list[tuple[float, float]]:
    """Bellows suction cup, lip at z = 0, stem upwards."""
    r = L.CUP_R
    return [(0.004, 0.0), (r, 0.0), (r, 0.002), (0.6 * r, 0.008), (0.85 * r, 0.012), (0.55 * r, 0.018),
            (0.7 * r, 0.022), (0.35 * r, 0.028), (0.35 * r, 0.034), (0.004, 0.034)]


def generate(out_dir: Path = MESH_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "bushing": revolve(bushing_profile(), 40),
        "cup": revolve(cup_profile(), 32),
        # Collision hull of the part: a 24-sided prism circumscribing the body (ribs included).
        "bushing_col": extrude([(L.PART_COL_R * np.cos(a), L.PART_COL_R * np.sin(a))
                                for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)], 0.0, L.PART_H),
        # One of the 8 convex segments lining the bore (only the gripper jaws touch them).
        "bore_seg": chamfered(extrude(bore_segment(), L.BORE_FLOOR, L.PART_H), L.BORE_CHAMFER),
    }
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
