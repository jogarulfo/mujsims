"""Procedural 3D assets (binary STL) for the Accuwat rack-filling cell.

Run ``accuwat-sim-assets`` (or ``python -m accuwat_sim.meshgen``) to regenerate the meshes in
``assets/meshes``. Every mesh is built from the dimensions in :mod:`accuwat_sim.layout`.

  holder    visual mesh of the rack's holder plate: a rounded plate with the 120 staggered
            holes, each with its lead-in chamfer. Built tile by tile: the honeycomb lattice's
            hexagonal cells tile the plane, so each hole gets its own hexagon-with-a-hole and
            the hexagons of the lattice points without a hole are clipped to the plate outline.
  hole_seg  one of the 6 convex segments lining a hole (collisions): a flat inner face on the
            hole radius, chamfered at the top.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from accuwat_sim import layout as L

MESH_DIR = Path(__file__).parent / "assets" / "meshes"


class TriMesh:
    """Triangle soup. Each triangle is wound to face along a hint direction, so the winding
    (and the shading) is right without bookkeeping."""

    def __init__(self):
        self.tris: list[np.ndarray] = []

    def tri(self, a, b, c, hint) -> None:
        t = np.array([a, b, c], float)
        n = np.cross(t[1] - t[0], t[2] - t[0])
        if np.linalg.norm(n) < 1e-14:
            return
        if n @ np.asarray(hint, float) < 0:
            t = t[[0, 2, 1]]
        self.tris.append(t)

    def quad(self, a, b, c, d, hint) -> None:
        self.tri(a, b, c, hint)
        self.tri(a, c, d, hint)

    def fan(self, poly, z: float, hint) -> None:
        """Convex polygon (2D, in order) at height z."""
        p = [np.r_[v, z] for v in poly]
        for k in range(1, len(p) - 1):
            self.tri(p[0], p[k], p[k + 1], hint)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        v = np.vstack(self.tris)
        return v, np.arange(len(v)).reshape(-1, 3)


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(nrm, axis=1)
    keep = area > 1e-14
    tri, nrm = tri[keep], nrm[keep] / area[keep, None]
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tri
    with open(path, "wb") as f:
        f.write(b"accuwat_sim procedural mesh".ljust(80, b" "))
        f.write(np.uint32(len(rec)).tobytes())
        f.write(rec.tobytes())


# --- 2D helpers -------------------------------------------------------------------------------


def rounded_rect(hx: float, hy: float, r: float, n: int = 6) -> np.ndarray:
    """Convex CCW outline of a rectangle with rounded corners."""
    pts = []
    for cx, cy, a0 in ((hx - r, hy - r, 0.0), (-hx + r, hy - r, np.pi / 2), (-hx + r, -hy + r, np.pi),
                       (hx - r, -hy + r, 1.5 * np.pi)):
        for a in np.linspace(a0, a0 + np.pi / 2, n + 1):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def clip_convex(poly: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: part of the polygon inside a convex CCW clip polygon."""
    out = [np.asarray(p, float) for p in poly]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        e = b - a

        def side(p, a=a, e=e):
            return e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])

        src, out = out, []
        for j in range(len(src)):
            p, q = src[j], src[(j + 1) % len(src)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                out.append(p + (q - p) * sp / (sp - sq))
        if not out:
            return np.zeros((0, 2))
    return np.array(out)


def plate_outline() -> np.ndarray:
    return rounded_rect(L.RACK_PLATE[0] / 2, L.RACK_PLATE[1] / 2, 0.012)


# --- Meshes -------------------------------------------------------------------------------------


def holder_plate(n_seg: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Holder plate with its holes, z from 0 (bottom) to HOLDER_T (top), centred on the rack."""
    T, c, rh = L.HOLDER_T, L.HOLE_CHAMFER, L.HOLE_R
    inr = L.RACK_PITCH / 2                       # hexagon (lattice cell) inradius
    rc = inr / np.cos(np.pi / 6)                 # ... and circumradius
    outline = plate_outline()
    m = TriMesh()
    up, down = (0, 0, 1), (0, 0, -1)
    angs = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    assert n_seg % 6 == 0                        # the hexagon corners are among the samples

    def hex_r(a):
        k = np.round((a - np.pi / 6) / (np.pi / 3))
        return inr / np.cos(a - (np.pi / 6 + k * np.pi / 3))

    for row in range(-2, L.RACK_ROWS + 2):
        for col in range(-2, L.RACK_COLS + 2):
            ctr = L.rack_hole_xy(row, col) - L.RACK_CENTER
            if 0 <= row < L.RACK_ROWS and 0 <= col < L.RACK_COLS:
                # Hexagon with a chamfered hole.
                for k in range(n_seg):
                    a0, a1 = angs[k], angs[(k + 1) % n_seg]
                    u0, u1 = np.array([np.cos(a0), np.sin(a0)]), np.array([np.cos(a1), np.sin(a1)])
                    o0, o1 = ctr + hex_r(a0) * u0, ctr + hex_r(a1) * u1
                    ct0, ct1 = ctr + (rh + c) * u0, ctr + (rh + c) * u1
                    h0, h1 = ctr + rh * u0, ctr + rh * u1
                    um = -(u0 + u1) / 2                    # towards the hole axis
                    m.quad(np.r_[ct0, T], np.r_[o0, T], np.r_[o1, T], np.r_[ct1, T], up)
                    m.quad(np.r_[h0, 0], np.r_[o0, 0], np.r_[o1, 0], np.r_[h1, 0], down)
                    m.quad(np.r_[ct0, T], np.r_[ct1, T], np.r_[h1, T - c], np.r_[h0, T - c], (*um, 1.0))
                    m.quad(np.r_[h0, T - c], np.r_[h1, T - c], np.r_[h1, 0], np.r_[h0, 0], (*um, 0.0))
            else:
                hexagon = np.array([ctr + rc * np.array([np.cos(a), np.sin(a)]) for a in np.arange(6) * np.pi / 3])
                poly = clip_convex(hexagon, outline)
                if len(poly) >= 3:
                    m.fan(poly, T, up)
                    m.fan(poly, 0.0, down)
    # Outer edge.
    for i in range(len(outline)):
        a, b = outline[i], outline[(i + 1) % len(outline)]
        mid = (a + b) / 2
        m.quad(np.r_[a, 0], np.r_[b, 0], np.r_[b, T], np.r_[a, T], (*mid, 0.0))
    return m.arrays()


def hole_segment() -> tuple[np.ndarray, np.ndarray]:
    """60 deg sector lining a hole (convex), its flat inner face on the hole radius, from the
    plate bottom (z = 0) to its top (z = HOLDER_T), chamfered at the top. Centred on +x."""
    half = np.pi / 6
    T, c = L.HOLDER_T, L.HOLE_CHAMFER
    r_in = L.HOLE_R / np.cos(half)
    r_out = (L.HOLE_R + 0.0012) / np.cos(half)
    r_ch = (L.HOLE_R + c) / np.cos(half)
    m = TriMesh()
    pts = []
    for z, r_i in ((0.0, r_in), (T - c, r_in), (T, r_ch)):
        for r, a in ((r_i, -half), (r_out, -half), (r_out, half), (r_i, half)):
            pts.append(np.array([r * np.cos(a), r * np.sin(a), z]))
    P = np.array(pts).reshape(3, 4, 3)          # [level][corner]
    ctr = P.reshape(-1, 3).mean(axis=0)
    # Faces of the convex solid, wound outward from its centre.
    for lvl in range(2):
        for i in range(4):
            j = (i + 1) % 4
            a, b, cc, d = P[lvl, i], P[lvl, j], P[lvl + 1, j], P[lvl + 1, i]
            m.quad(a, b, cc, d, (a + b + cc + d) / 4 - ctr)
    for lvl, hint in ((0, (0, 0, -1)), (2, (0, 0, 1))):
        m.quad(*P[lvl], hint)
    return m.arrays()


def generate(out_dir: Path = MESH_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {"holder": holder_plate(), "hole_seg": hole_segment()}
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
