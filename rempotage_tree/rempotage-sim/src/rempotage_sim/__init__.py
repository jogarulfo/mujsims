"""Repotting line: two I2RT YAM arms plant young trees into 3 L pots on a potting wheel.

    rempotage-sim                     interactive viewer at real speed
    rempotage-sim --speed 3           3x faster than real time
    rempotage-sim --video demo.mp4    headless render to MP4 (needs ffmpeg), main view + whole line +
                                      the planting station
    rempotage-sim --export-xml f.xml  write the cell as a standalone MJCF
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="rempotage-sim", description="Bimanual YAM repotting line (MuJoCo).")
    p.add_argument("--speed", type=float, default=1.0, help="viewer / video real-time factor")
    p.add_argument("--no-assist", action="store_true", help="arm A carries the root ball by friction only")
    p.add_argument("--video", type=Path, help="render headless to this MP4 file instead of opening a viewer")
    p.add_argument("--video-speed", type=float, default=None, help="video speed-up (default: --speed)")
    p.add_argument("--camera", default="auto",
                   help="video camera: auto (behind the robots), behind, behind_close, overview, line, planting, "
                        "planting_close, pot_top, robots, cart, wheel_top, pallet, mulch, tray_cam")
    p.add_argument("--no-pip", action="store_true", help="video without the line / planting station insets")
    p.add_argument("--size", default="1920x1080", help="video size WxH")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--report", type=Path, help="write the final report (JSON) here")
    p.add_argument("--export-xml", type=Path, help="write the cell MJCF and exit")
    return p.parse_args(argv)


def _lines(sim) -> list[tuple[str, str]]:
    ln = sim.line
    k = ln.pot_ready
    pot = f"pot {k + 1}/{len(ln.state)}" if k is not None else "-"
    return [
        ("Robots", "2 x I2RT YAM + linear_4310 gripper"),
        ("Robots do", sim.status.split(": ", 1)[-1] if sim.status else "-"),
        ("Line", ln.status),
        ("Planting", (f"{pot}   potted: {len(sim.potted)}/{len(ln.state)}   "
                      f"mulched: {int((ln.mulch >= 1).sum())}   pallet: {len(ln.on_pallet)}")),
        ("Time", f"{sim.data.time:6.1f} s"),
    ]


def run_viewer(args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    from rempotage_sim.sim import CellSim

    sim = CellSim(assist=not args.no_assist)
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.lookat[:] = [0.15, -0.25, 0.85]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.6, -125, -32
        wall0, sim0 = time.perf_counter(), sim.data.time
        rtf, rtf_wall, rtf_sim = 1.0, wall0, sim0
        while v.is_running():
            with v.lock():
                sim.step()
            now = time.perf_counter()
            if now - rtf_wall > 1.0:
                rtf = (sim.data.time - rtf_sim) / (now - rtf_wall)
                rtf_wall, rtf_sim = now, sim.data.time
            rows = _lines(sim) + [("Speed", f"{rtf:4.2f}x real time")]
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)))
            v.sync()
            lag = (sim.data.time - sim0) / args.speed - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:  # cannot keep up: resync instead of fast-forwarding
                wall0, sim0 = time.perf_counter(), sim.data.time


class _AutoCamera:
    """Main video camera: always behind the robots, the wide view while arm A fetches a plant and the
    line runs, the close one on the planting station while the robots plant. Each shot is held a while."""

    SHOTS: ClassVar[dict[str, str]] = {"insert": "behind_close", "push": "behind_close", "tamp": "behind_close"}

    def __init__(self, hold: float = 2.5):
        self.cam, self.since, self.hold = "behind", -1e9, hold

    def __call__(self, sim) -> str:
        want = self.SHOTS.get(sim.phase, "behind")
        if want != self.cam and sim.data.time - self.since >= self.hold:
            self.cam, self.since = want, sim.data.time
        return self.cam


def run_video(args: argparse.Namespace) -> None:
    import mujoco
    import numpy as np

    from rempotage_sim.sim import CellSim

    width, height = (int(v) for v in args.size.lower().split("x"))
    speed = args.video_speed or args.speed
    sim = CellSim(assist=not args.no_assist, verbose=False)
    main = mujoco.Renderer(sim.model, height, width)
    pw, ph = width // 4, height // 4
    pip = mujoco.Renderer(sim.model, ph, pw)
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default(size=max(14, height // 46))
        small = ImageFont.load_default(size=max(11, height // 60))
    except ImportError:  # captions are optional
        Image = None
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", str(args.video)]
    try:
        ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("ffmpeg not found: install it to export videos")
    auto = _AutoCamera()
    next_frame, frames, t_end = 0.0, 0, None
    orange = (240, 110, 30, 255)
    wall0 = time.perf_counter()
    while True:
        sim.step()
        if sim.done and t_end is None:
            t_end = sim.data.time + 2.0
            print(f"routine done at {sim.data.time:.1f} s")
        if t_end is not None and sim.data.time >= t_end:
            break
        if sim.data.time < next_frame:
            continue
        next_frame += speed / args.fps
        cam = auto(sim) if args.camera == "auto" else args.camera
        main.update_scene(sim.data, camera=cam)
        frame = main.render().copy()
        insets = []
        if not args.no_pip:
            m = height // 60
            for k, (name, label) in enumerate((("overview", "Whole line"), ("planting_close", "Planting station"))):
                pip.update_scene(sim.data, camera=name)
                img = pip.render()
                y0 = m + k * (ph + 2 * m)
                frame[y0:y0 + ph, width - pw - m:width - m] = img
                insets.append((width - pw - m, y0, label))
        if Image is not None:
            im = Image.fromarray(frame)
            dr = ImageDraw.Draw(im, "RGBA")
            for x0, y0, label in insets:
                dr.rectangle([x0 - 2, y0 - 2, x0 + pw + 1, y0 + ph + 1], outline=orange, width=3)
                box = dr.textbbox((0, 0), label, font=small)
                dr.rectangle([x0, y0, x0 + box[2] + 12, y0 + box[3] + 8], fill=(0, 0, 0, 160))
                dr.text((x0 + 6, y0 + 3), label, font=small, fill=(255, 255, 255, 255))
            rows = _lines(sim)
            pad = height // 70
            col = max(dr.textbbox((0, 0), r[0], font=font)[2] for r in rows) + 2 * pad
            lh = dr.textbbox((0, 0), "Ag", font=font)[3] + pad // 2
            w = col + max(dr.textbbox((0, 0), r[1], font=font)[2] for r in rows) + 2 * pad
            dr.rectangle([0, 0, w, len(rows) * lh + 2 * pad], fill=(0, 0, 0, 165))
            for i, (a, b) in enumerate(rows):
                dr.text((pad, pad + i * lh), a, font=font, fill=(170, 170, 170, 255))
                dr.text((col, pad + i * lh), b, font=font, fill=(255, 255, 255, 255))
            frame = np.asarray(im)
        ffmpeg.stdin.write(frame.tobytes())
        frames += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    main.close()
    pip.close()
    print(f"wrote {args.video} ({frames} frames, {sim.data.time:.1f} s simulated, "
          f"{time.perf_counter() - wall0:.0f} s wall)")
    _print_report(sim, args.report)


def _print_report(sim, path: Path | None) -> None:
    rep = sim.report()
    from rempotage_sim import layout as L

    for r in rep["plants"]:
        firmed = (f"soil firmed {r['firmed_mm']:.1f} mm (least {r['firmed_min_mm']:.1f}), " if L.TAMP_SIDES
                  else "")
        print(f"plant {r['plant'] + 1} -> pot {r['pot'] + 1}: {r['off_axis_mm']:.1f} mm off the hole axis, "
              f"{r['above_floor_mm']:.1f} mm above the hole floor, tilt {r['tilt_deg']:.1f} deg, {firmed}"
              f"mulched {r['mulched'] * 100:.0f} %, pot {r['state']}")
    for p, k, f, gap, t in rep["push"]:
        print(f"push plant {p + 1}: {f:.1f} N at the end, {t:.1f} s")
    if rep["cycle_times"]:
        ct = rep["cycle_times"]
        print(f"planting station: {sum(ct) / len(ct):.1f} s per plant on average (max {max(ct):.1f})")
    if rep["arm_hits"]:
        print("arm contacts outside the work:", ", ".join(f"{k} ({v * 1000:.1f} mm)" for k, v in rep["arm_hits"].items()))
    else:
        print("no arm contacts outside the work")
    if path:
        path.write_text(json.dumps(rep, indent=1, default=float))


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.export_xml:
        from rempotage_sim.scene import export_xml
        print("wrote", export_xml(args.export_xml))
        return
    if args.video:
        run_video(args)
    else:
        run_viewer(args)


def assets_main() -> None:
    from rempotage_sim import meshgen
    meshgen.main()
