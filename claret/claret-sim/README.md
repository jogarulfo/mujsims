# Claret packing cell: machine unloading into a carton (MuJoCo)

Demo simulation for the Claret Group use case. The cell takes the parts that come out of a
production machine and packs them tight in a cardboard box. Optionally, it adds a cardboard
layer sheet automatically.

It is built after the photo of the Claret line (`../claret_machine_photo.png`):

- a blue machine with a hood and a rubber strip curtain;
- a wide stainless outlet tray on blue legs;
- black ribbed rubber-metal bushings;
- a lined carton with a cardboard sheet on its floor, with the parts standing upright and touching.

One I2RT YAM arm replaces the operator. The packing strategy is the one of the lejunior
rollout (`store_pvc_pieces`, SO-101): grip the part by its bore, set it down next to its
place, then push it in tight against its neighbours.

![cell](docs/cell.png)

## The cell

| Element | What it does |
|---|---|
| **Machine** | Finishes a part at random intervals: 9 s on average (`--cycle`), anywhere between 4 and 14 s. It drops the part at a random spot on its outlet plate, behind the rubber curtain, and a pneumatic ejector pushes it out onto the tray. It holds the next part while the tray is full (6 parts, or a part standing at the "tray full" light barrier). |
| **Outlet tray** | Wide stainless tray (34 cm) sloping 9.5° down towards the robot, ending at a rubber end stop. No guide lane: each part slides down and stops wherever it stops, against the end stop or against the parts already lying there. A camera over the tray reports the parts lying still. |
| **Robot** (YAM on a pedestal) | Takes the part nearest the end stop and packs it in the next place of the carton. |
| **Bore jaws** | Two narrow aluminium jaws (10 x 4 mm, 52 mm long) under the gripper pads. They go into the part's bore closed, then open and press on the bore from the inside. Nothing sticks out around the part, so it can go right against its neighbours. |
| **Vacuum cup** | A pneumatic cylinder on the side of the gripper carries a suction cup. Retracted, the cup stays out of the way. Extended, it reaches 1 cm below the jaw tips to pick up a sheet. |
| **Carton** | 26 x 32.3 x 22 cm inside, lined with a plastic bag, on a roller stand. It holds 4 x 5 parts per layer, touching each other and the walls (8 mm of play per direction), 3 layers by default (`--layers 1..3`). |
| **Sheet magazine** | Holds 6 cardboard layer sheets, stacked flat between corner guides. |
| **Stack light** | Green: working. Amber: robot waiting for a part. Blinking blue: carton full, swap it. |

### Packing cycle

1. **Bottom sheet** (with sheets on, the default): the robot extends the cup, takes the top sheet of the magazine, turns it 90° on the way and lays it on the carton floor.
2. For each place, far row first, starting in the far left corner:
   - **pick**: the robot picks the part nearest the end stop (any part it can reach without touching the tray walls or the other parts). It aligns the closed jaws with the bore, goes in along the jaws' length, opens them, and lifts the part out;
   - **place**: it swings to above the carton, goes down slowly, corrects the last millimetres from where the part really is, and sets the part down 12 mm short of its place;
   - **push**: it lets go, then pushes the part with the jaws towards the far left corner of its pocket, until the part stops against its neighbours (or the walls). The jaws come out of the bore along their length.
3. **Between the layers and on top** (with sheets on), a cardboard sheet as in step 1.
4. **Carton full**: the robot goes home and the stack light blinks blue.

With `--no-sheets`, the parts of the next layer stand directly on the ones below.

![tray](docs/tray.png)

*Parts lying where they stopped on the tray. The robot's jaws go into the bore of the nearest one.*

![push](docs/push.png)

*Set down short of its place, then pushed into the corner of its pocket.*

![layer](docs/layer.png)

*A finished layer, from above: 20 parts touching each other and the walls.*

### Measured (default settings, 3 layers + 4 sheets)

- All 60 parts packed and standing upright (largest tilt 0.2°).
- Tight: neighbouring parts touch (median gap -0.2 mm, largest 5.9 mm); the first row and
  column touch the walls (within 0.2 mm). All play ends up along the near and -y walls.
- Every part within 1.3 mm of its ideal position on median (7.7 mm at worst, the last
  corner of a layer).
- The arm stays clear of the carton, the tray and the magazine, except for one 0.1 mm
  touch of the gripper housing on the near wall at the last corner of the first layer.
- Robot cycle: 10.8 s per part; each sheet takes about 9 s. The robot is a bit slower than
  the machine (9 s on average), so the machine sometimes holds with a full tray.
- Full carton (60 parts, 4 sheets) in about 11 min.
- The simulation alone runs about 3x faster than real time.
- Other random seeds (one layer each): all 20 parts packed and upright, largest gap 4.5 to
  6.9 mm.

## Run

```bash
uv sync
uv run claret-sim                       # interactive viewer at real speed
uv run claret-sim --no-sheets           # no cardboard layer sheets
uv run claret-sim --layers 2            # 2 layers (40 parts, 3 sheets)
uv run claret-sim --cycle 12            # slower machine: the robot waits (amber light)
uv run claret-sim --seed 3              # repeatable random machine (default: random)
uv run claret-sim --speed 3             # 3x faster than real time
uv run claret-sim --video demo.mp4 --video-speed 2   # headless 2x MP4, auto camera + captions (needs ffmpeg)
uv run claret-sim --video box.mp4 --camera box_top --size 1920x1080
uv run claret-sim --export-xml cell.xml # standalone MJCF of the cell
uv run claret-sim-assets                # regenerate the procedural meshes
```

Headless rendering on a machine without a display: `MUJOCO_GL=egl uv run claret-sim --video demo.mp4`.

Cameras: `overview`, `photo` (roughly the viewpoint of the Claret photo), `front`, `machine`,
`tray`, `tray_top`, `box`, `box_top`, `box_close`, `magazine`.

## Code map

```
src/claret_sim/
  layout.py     all dimensions and positions (part, jaws, tray, carton, packing grid, sheets, cup)
  meshgen.py    procedural meshes -> assets/meshes/*.stl (ribbed bushing, its collision hull and bore lining, suction cup)
  scene.py      builds the MuJoCo model with MjSpec (YAM attached as r/, bore jaws and vacuum tool added on its wrist)
  machine.py    machine + tray: random production, ejector, tray camera, "tray full"
  arm.py        YAM control: damped-least-squares IK, Cartesian min-jerk moves, arc swings, settle
  planner.py    collision-checked hand poses (IK + MuJoCo collision detection on a copy of the model)
  task.py       the packing routine (generator, one yield = one 10 ms control tick)
  sim.py        CellSim: physics stepping, vacuum cup, stack light, grasp assist, packing report
  assets/yam/   I2RT YAM + linear_4310 gripper from MuJoCo Menagerie (MIT, see LICENSE)
  assets/claret_cell.xml   exported cell (open it in MuJoCo's simulate app)
```

To change the demo, edit `layout.py`. For example:

- `GRID`, `BOX_PLAY` and `N_LAYERS` set the packing pattern (the carton follows);
- `PART_R`, `PART_BORE_R` and `PART_H` set the part (the meshes follow);
- `MACHINE_CYCLE`, `CYCLE_SPREAD`, `CHUTE_SLOPE` and `CHUTE_FRICTION` tune the machine and the tray;
- `SET_DOWN_OFFSET` and `PUSH_DIR` set the push.

## Modelling choices and simplifications

- **Machine.** Modelled from the outside. The parts wait, hidden and weightless, inside the
  cabinet. When one is finished, the machine puts it at a random spot on its outlet plate.
  From there on, everything is physical: ejector, curtain strips, tray, end stop, the other
  parts on the tray.
- **Part.** The visual mesh is a ribbed bushing with a 32 mm bore (63 mm over the ribs, 56 mm
  tall, 180 g). For collisions it has two shapes:
  - a solid 24-sided prism, for everything else (tray, carton, sheets, other parts);
  - a lining of the bore made of 8 convex segments, with a lead-in chamfer at the top, which
    only the jaws touch.
- **Grip.** The YAM's stock fingertips are 22 mm wide and do not fit in the bore, so the cell
  has custom jaws, as an integrator would make them. They grip the part from the inside.
  Once both jaws touch the part, it is welded to the hand (grasp assist, for a repeatable
  demo). `--no-assist` grips on friction alone (about 15 N per rubber-faced jaw). It is
  experimental: where the hand leans to clear the walls, the jaws are tilted in the bore
  and touch it at one point each, the part turns in the jaws when lifted, and the packing
  breaks down. A real cell would need jaws shaped for the lean, or a tilting wrist.
- **Push.** After letting go, the jaws stay in the bore and push the part at 3 cm/s towards its
  pocket. The robot stops when the part has stopped moving for 0.12 s: it sits against its
  neighbours. A real cell would detect this from the motor currents or a force sensor.
  Rubber on cardboard is modelled with μ = 0.5, so a push at the jaw height can never tip a
  part.
- **Hand poses near the walls.** The parts touch the walls, but the gripper housing is 10 x
  6.5 cm and the forearm passes over the near wall. At startup the planner tries, for every
  place, which side the vacuum cup faces and how much the hand leans (up to 32°), and keeps
  the first four poses that keep the arm, the jaws and the cup 6 mm clear of the carton and
  of the parts already packed (about 3 s for the 60 places). At the tray, the robot uses
  whichever of these lets it take the part cleanly. Before setting the part down, it
  checks the pose again with the part as it really sits in the jaws and the parts as they
  really lie in the carton. During the transfer, the arm steers towards the planned joint
  configuration.
- **Pedestal.** 82 cm high. With the arm base lower, the forearm comes down over the near
  carton wall when the hand works in the near row.
- **Parts against the tray wall.** If no part on the tray can be taken with the hand poses
  of the next place (the hand would hit the tray wall), the robot first pushes the part
  away from the wall with its jaws, the hand vertical, then takes it.
- **Tray camera.** The robot locates the parts from the simulator state; the camera on its
  pole stands for the vision system that would do this in the real cell (lejunior's YOLO
  detector).
- **Wrist range.** Joint 6 turns ±120° only. The cup therefore always faces roughly towards
  the robot, or sideways.
- **Sheets.** The cup seals when its lip touches a sheet with the vacuum on: a weld at the
  current relative pose stands in for suction. Sheets are rigid boards with 8 mm of play in
  the carton.
- **Arm control.** Menagerie gains, plus a small integral term (only while the command holds
  still, so it does not wind up during fast moves). The hand goes down into the carton
  slowly (15 cm/s on average): the parts go in with 8 mm to spare.
- **Curtain.** The rubber strips are slippery (μ = 0.05). On a 9.5° slope with μ = 0.14 the
  part's pull is only 0.05 N, and grippy strips would hold it at the outlet.
- **Carton liner.** The plastic bag and the folded flaps are visual only.

## Relation to the lejunior pipeline

[lejunior](https://github.com/jogarulfo/lejunior) runs the same task on a real SO-101: it
puts PVC rings into a cardboard box, using YOLO detection and an ACT policy. Its rollout
`store_pvc_pieces_bbox_v2_dagger_v7_ft_v9_ft_v10` shows the strategy used here: take the ring
by its wall, set it down beside its place, and push it in against its neighbours, filling the
box from one corner to the edges. This simulation is the Claret-scale version of the cell:
an industrial machine with random output, a deeper carton, layers and sheets. It can be used to:

- show the customer the cell before building it;
- check reach, cycle time and the packing pattern;
- produce rendered images (all cameras, any lighting) for detection training.
