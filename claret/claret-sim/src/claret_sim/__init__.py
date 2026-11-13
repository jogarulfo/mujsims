"""Claret packing cell: one I2RT YAM arm unloads a production machine into a cardboard carton.

    claret-sim                     interactive viewer at real speed: the machine releases parts at
                                   random times onto its outlet tray, the robot takes each one by its
                                   bore, sets it down in the carton and pushes it tight against its
                                   neighbours, layer by layer, with cardboard sheets between layers
    claret-sim --no-sheets         same, without the cardboard layer sheets
    claret-sim --video demo.mp4    headless render to MP4 (needs ffmpeg)
    claret-sim --export-xml f.xml  write the cell as a standalone MJCF
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from claret_sim import layout as L


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="claret-sim", description="Claret machine-unloading and carton-packing cell (MuJoCo).")
    p.add_argument("--no-sheets", action="store_true", help="do not lay cardboard sheets (bottom, between layers, top)")
    p.add_argument("--layers", type=int, default=L.N_LAYERS, help=f"layers per carton (default {L.N_LAYERS})")
    p.add_argument("--cycle", type=float, default=L.MACHINE_CYCLE,
                   help=f"mean machine cycle time in seconds per part (default {L.MACHINE_CYCLE:g})")
    p.add_argument("--seed", type=int, default=None, help="random seed for the machine (default: random)")
    p.add_argument("--no-assist", action="store_true",
                   help="disable the grasp-assist weld (pure friction grasps; experimental: parts turn in the jaws"
                        " when the hand leans near the walls, and the packing degrades)")
    p.add_argument("--speed", type=float, default=1.0, help="viewer real-time factor")
    p.add_argument("--video", type=Path, help="render headless to this MP4 file instead of opening a viewer")
    p.add_argument("--camera", default="auto",
                   help="video camera: auto, overview, photo, front, machine, tray, tray_top, box, box_top,"
                        " box_close, magazine")
    p.add_argument("--video-speed", type=float, default=1.0, help="video playback speed, e.g. 2 for a 2x video")
    p.add_argument("--size", default="1280x720", help="video size WxH")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--export-xml", type=Path, help="write the cell MJCF and exit")
    args = p.parse_args(argv)
    if not 1 <= args.layers <= 3:
        p.error("--layers must be 1, 2 or 3 (the carton is 22 cm deep)")
    if args.video_speed <= 0:
        p.error("--video-speed must be positive")
    if args.layers + 1 > L.N_SHEETS and not args.no_sheets:
        p.error(f"the sheet magazine holds {L.N_SHEETS} sheets")
    return args


def _make_sim(args, verbose: bool = True):
    from claret_sim.sim import CellSim
    if args.no_assist:
        print("warning: --no-assist is experimental here: with friction alone the parts turn in the jaws when the"
              " hand leans near the carton walls, and the packing degrades.")
    print("Planning the hand poses for the carton slots...", flush=True)
    return CellSim(sheets=not args.no_sheets, layers=args.layers, cycle=args.cycle, assist=not args.no_assist,
                   verbose=verbose, seed=args.seed)


def _status_lines(sim, rtf: float | None = None) -> tuple[str, str]:
    m = sim.machine
    left = "Task\nMachine\nTray\nCarton\nSheets\nSim time" + ("\nSpeed" if rtf is not None else "")
    right = (f"{sim.status}\n"
             f"{m.made}/{len(m.parts)} parts made, {sim.machine.cycle:g} s mean cycle"
             f"{'  (holding: tray full)' if m.waiting_full else ''}\n"
             f"{m.queue_length()} parts, {len(m.pickable())} ready to pick\n"
             f"{sim.carton_count()} parts\n"
             f"{(str(sim.sheets_laid) + ' laid') if sim.sheets else 'off'}\n"
             f"{sim.data.time:6.1f} s" + (f"\n{rtf:4.2f}x real time" if rtf is not None else ""))
    return left, right


def run_viewer(args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    sim = _make_sim(args)
    print("Viewer: close the window to quit.")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.lookat[:] = [0.15, 0.25, 0.70]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.4, 215, -28
        wall0, sim0 = time.perf_counter(), sim.data.time
        rtf, rtf_wall, rtf_sim = 1.0, wall0, sim0
        while v.is_running():
            with v.lock():
                sim.step()
                v.user_scn.ngeom = 0
                sim.draw(v.user_scn)
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


_SHOTS = {"start": "photo", "pick": "photo", "place": "box", "push": "box_close", "sheet_pick": "magazine",
          "sheet_place": "box", "done": "overview"}


class _AutoCamera:
    """Follows the work: chute when picking, carton when placing, magazine for the sheets.
    Holds each shot at least a couple of seconds so the cut rate stays calm."""

    def __init__(self, min_hold: float = 2.5):
        self.cam, self.since, self.min_hold = "photo", -1e9, min_hold
        self.n_place = 0
        self._last_phase = None

    def __call__(self, sim) -> str:
        phase = sim.phase
        if phase != self._last_phase:
            self._last_phase = phase
            if phase == "place":
                self.n_place += 1
        want = _SHOTS.get(phase, "overview")
        if phase == "pick" and self.n_place % 3 == 1:
            want = "tray"         # now and then a close-up of the tray
        if phase in ("place", "push") and self.n_place % 4 == 2:
            want = "box_top"
        if want != self.cam and sim.data.time - self.since >= self.min_hold:
            self.cam, self.since = want, sim.data.time
        return self.cam


def run_video(args: argparse.Namespace) -> None:
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
            t_end = sim.data.time + 3.0  # keep filming a little after the routine ends
        if t_end is not None and sim.data.time >= t_end:
            break
        if sim.data.time < next_frame:
            continue
        next_frame += args.video_speed / args.fps
        cam = auto(sim) if args.camera == "auto" else args.camera
        renderer.update_scene(sim.data, camera=cam)
        sim.draw(renderer.scene)
        frame = renderer.render()
        if Image is not None:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img, "RGBA")
            pad = height // 60
            m = sim.machine
            text = (f"{sim.status}   |   machine {m.made}/{len(m.parts)}   |   tray {m.queue_length()}"
                    f"   |   carton {sim.carton_count()}" + (f"   |   sheets {sim.sheets_laid}" if sim.sheets else ""))
            box = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([0, height - box[3] - 3 * pad, width, height], fill=(0, 0, 0, 150))
            draw.text((2 * pad, height - box[3] - 1.5 * pad), text, font=font, fill=(255, 255, 255, 255))
            label = (f"Claret packing cell  -  t = {sim.data.time:5.1f} s  -  camera: {cam}"
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
    print(f"wrote {args.video} ({frames} frames, {sim.data.time:.1f} s simulated, {sim.packed} parts packed)")


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.export_xml:
        from claret_sim.scene import export_xml
        print("wrote", export_xml(args.export_xml, layers=args.layers))
        return
    if args.video:
        run_video(args)
    else:
        run_viewer(args)


def assets_main() -> None:
    from claret_sim import meshgen
    meshgen.main()
