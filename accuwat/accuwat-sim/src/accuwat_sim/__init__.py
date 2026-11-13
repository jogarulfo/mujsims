"""Accuwat rack-filling cell: one I2RT YAM arm with an electromagnet fills a cell rack.

    accuwat-sim                     interactive viewer at real speed: the robot takes 21700 cells one
                                    by one from the "+ up" and "- up" boxes and fills the 15 x 8 rack
                                    in three blocks: + up, - up, + up
    accuwat-sim --cells 30          stop after 30 cells
    accuwat-sim --video demo.mp4    headless render to MP4 (needs ffmpeg)
    accuwat-sim --export-xml f.xml  write the cell as a standalone MJCF
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from accuwat_sim import layout as L

N_HOLES = L.RACK_ROWS * L.RACK_COLS


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="accuwat-sim", description="Accuwat rack-filling cell (MuJoCo).")
    p.add_argument("--cells", type=int, default=N_HOLES, help=f"cells to place (default: the full rack, {N_HOLES})")
    p.add_argument("--speed", type=float, default=1.0, help="viewer real-time factor")
    p.add_argument("--no-marks", action="store_true",
                   help="hide the polarity check marks (red disc: + up, blue disc: - up) over the placed cells")
    p.add_argument("--no-sleep", action="store_true",
                   help="keep every cell awake (slow: the resting cells' contacts are computed every step)")
    p.add_argument("--video", type=Path, help="render headless to this MP4 file instead of opening a viewer")
    p.add_argument("--camera", default="auto",
                   help="video camera: auto, overview, photo, front, side, rack, rack_top, rack_close, block1,"
                        " block2, block3, box_plus, box_minus")
    p.add_argument("--video-speed", type=float, default=1.0, help="video playback speed, e.g. 4 for a 4x video")
    p.add_argument("--size", default="1280x720", help="video size WxH")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--export-xml", type=Path, help="write the cell MJCF and exit")
    args = p.parse_args(argv)
    if not 1 <= args.cells <= N_HOLES:
        p.error(f"--cells must be between 1 and {N_HOLES}")
    if args.video_speed <= 0:
        p.error("--video-speed must be positive")
    return args


def _make_sim(args, verbose: bool = True):
    from accuwat_sim.sim import CellSim
    return CellSim(n_cells=args.cells, verbose=verbose, sleep=not args.no_sleep)


def _status_lines(sim, rtf: float | None = None) -> tuple[str, str]:
    c0, c1, pol = L.BLOCKS[sim.block]
    cyc = sim.cycle_times
    left = "Task\nBlock\nRack\nBoxes\nCycle\nSim time" + ("\nSpeed" if rtf is not None else "")
    right = (f"{sim.status}\n"
             f"{sim.block + 1}/3: columns {c0 + 1}-{c1}, {'+' if pol > 0 else '-'} up\n"
             f"{sim.rack_count()} cells\n"
             f"'+ up' {sim.box_left[+1]} left, '- up' {sim.box_left[-1]} left\n"
             f"{(sum(cyc) / len(cyc)) if cyc else 0:.1f} s per cell\n"
             f"{sim.data.time:6.1f} s" + (f"\n{rtf:4.2f}x real time" if rtf is not None else ""))
    return left, right


def run_viewer(args: argparse.Namespace):
    import mujoco
    import mujoco.viewer

    sim = _make_sim(args)
    print("Viewer: close the window to quit.")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.lookat[:] = [0.25, 0.0, 0.80]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -32
        wall0, sim0 = time.perf_counter(), sim.data.time
        rtf, rtf_wall, rtf_sim = 1.0, wall0, sim0
        while v.is_running():
            with v.lock():
                sim.step()
                v.user_scn.ngeom = 0
                sim.draw(v.user_scn, marks=not args.no_marks)
            now = time.perf_counter()
            if now - rtf_wall > 1.0:
                rtf = (sim.data.time - rtf_sim) / (now - rtf_wall)
                rtf_wall, rtf_sim = now, sim.data.time
            left, right = _status_lines(sim, rtf)
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, left, right))
            v.sync()
            lag = (sim.data.time - sim0) / args.speed - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:  # cannot keep up: resync instead of fast-forwarding
                wall0, sim0 = time.perf_counter(), sim.data.time
    return sim


class _AutoCamera:
    """Stays on the main view (`photo`) the whole time; once the rack is done, one cut to the top
    view to show the polarity check."""

    def __call__(self, sim) -> str:
        return "rack_top" if sim.phase == "done" else "photo"


def run_video(args: argparse.Namespace):
    import mujoco
    import numpy as np

    width, height = (int(v) for v in args.size.lower().split("x"))
    sim = _make_sim(args)
    renderer = mujoco.Renderer(sim.model, height, width)
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default(size=max(14, height // 36))
        small = ImageFont.load_default(size=max(11, height // 54))
    except ImportError:  # captions are optional
        Image = None
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.video)]
    try:
        ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("ffmpeg not found: install it to export videos")
    auto = _AutoCamera()
    next_frame, frames = 0.0, 0
    t_end = None
    while True:
        sim.step()
        if sim.done and t_end is None:
            t_end = sim.data.time + 6.0  # keep filming a little after the routine ends (top view)
        if t_end is not None and sim.data.time >= t_end:
            break
        if sim.data.time < next_frame:
            continue
        next_frame += args.video_speed / args.fps
        cam = auto(sim) if args.camera == "auto" else args.camera
        renderer.update_scene(sim.data, camera=cam)
        sim.draw(renderer.scene, marks=not args.no_marks)
        frame = renderer.render()
        if Image is not None:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img, "RGBA")
            pad = height // 60
            pol = L.BLOCKS[sim.block][2]
            text = (f"{sim.status}   |   rack {sim.rack_count()}   |   block {sim.block + 1}/3"
                    f" ({'+' if pol > 0 else '-'} up)   |   boxes +{sim.box_left[+1]} / -{sim.box_left[-1]}")
            box = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([0, height - box[3] - 3 * pad, width, height], fill=(0, 0, 0, 150))
            draw.text((2 * pad, height - box[3] - 1.5 * pad), text, font=font, fill=(255, 255, 255, 255))
            label = (f"Accuwat rack filling  -  t = {sim.data.time:5.1f} s  -  camera: {cam}"
                     + (f"  -  {args.video_speed:g}x" if args.video_speed != 1 else ""))
            box = draw.textbbox((0, 0), label, font=small)
            draw.rectangle([0, 0, box[2] + 3 * pad, box[3] + 2 * pad], fill=(0, 0, 0, 120))
            draw.text((1.5 * pad, pad), label, font=small, fill=(255, 255, 255, 255))
            frame = np.asarray(img)
        ffmpeg.stdin.write(frame.tobytes())
        frames += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    renderer.close()
    print(f"wrote {args.video} ({frames} frames, {sim.data.time:.1f} s simulated, {len(sim.placed)} cells placed)")
    return sim


def _report(sim) -> None:
    import numpy as np
    rep = sim.rack_report()
    if not len(rep["offsets"]):
        return
    blocks = ", ".join(f"block {k + 1} ({'+' if L.BLOCKS[k][2] > 0 else '-'} up): {ok}/{n} right way up"
                       for k, (ok, n) in rep["per_block"].items() if n)
    print(f"Rack: {len(sim.placed)} cells; {blocks}")
    print(f"  off the hole centre: median {np.median(rep['offsets']) * 1000:.2f} mm, max {rep['offsets'].max() * 1000:.2f} mm"
          f" (0.45 mm play in the hole); largest tilt {rep['tilts'].max():.2f} deg")
    print(f"  magnet off the cell axis when it caught it: median {np.median(sim.magnet.pick_offsets) * 1000:.2f} mm,"
          f" max {max(sim.magnet.pick_offsets) * 1000:.2f} mm (0.85 mm before it reaches a neighbour)")
    print(f"  largest sideways drift of a cell lifted out of its box: {sim.lift_dev * 1000:.2f} mm (0.3 mm gap"
          " along a row, 0.75 mm to the dividers)")
    if sim.magnet.grabbed_wrong:
        print(f"  the magnet caught a neighbouring cell {len(sim.magnet.grabbed_wrong)} time(s)")
    if sim.cycle_times:
        print(f"  cycle: {np.mean(sim.cycle_times):.2f} s per cell on average (max {max(sim.cycle_times):.2f} s)")
    if sim.arm_hits:
        print("  arm contacts:", sim.arm_hits)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.export_xml:
        from accuwat_sim.scene import export_xml
        print("wrote", export_xml(args.export_xml))
        return
    sim = run_video(args) if args.video else run_viewer(args)
    _report(sim)


def assets_main() -> None:
    from accuwat_sim import meshgen
    meshgen.main()
