# Accuwat rack-filling cell: 21700 cells into a rack, + / - / + (MuJoCo)

Demo simulation for the Accuwat use case. One I2RT YAM arm (the same arm as in the Claret cell)
with an electromagnet in place of its gripper takes 21700 cells one by one from two shipping
boxes and fills a cell rack:

- the **"+ up" box** holds cells standing with the positive terminal on top;
- the **"- up" box** holds cells standing the other way up;
- the **rack** (15 x 8 staggered holes) is filled in three blocks along its long side:
  columns 1-5 with + up cells, columns 6-10 with - up cells, columns 11-15 with + up cells again.

A cell never turns over: it goes into the rack the way it stood in its box. Picking from the right
box sets the polarity.

It is built after the two reference photos (`../cell_box_photo.png`, `../rack_photo.png`): blue
21700 cells standing in rows in an open cardboard box, and a light grey holder plate with a
honeycomb hole pattern.

![cell](docs/cell.png)

## The cell

| Element | What it does |
|---|---|
| **Robot** (I2RT YAM) | On a 15 cm steel riser bolted to the table, between the two boxes, the rack in front of it. Mounted lower, joint 4 runs out of range at the far box corners when the magnet points straight down. |
| **Electromagnet** | 20 x 15 mm holding electromagnet (12 V, about 25 N) on a stand-off and an adapter disc on the wrist flange, instead of the linear gripper. It is narrower than the cell (21.1 mm), so it comes down on one cell without touching its neighbours (0.85 mm away in the box). Red LED on the adapter: magnet on. |
| **Cells** | 21700, 21.1 x 70.4 mm, 69 g, blue sleeve. + end: white insulating ring and button cap; - end: flat can bottom. Both ends are nickel-plated steel, so the magnet holds either. |
| **Boxes** | Open cardboard boxes of 10 x 10 cells, standing in rows with cardboard dividers, the cell tops 1.4 mm proud of the rim. A white label with a big red + or blue - on the front. |
| **Rack** | Aluminium base plate the cells stand on, and a light grey holder plate (10 mm, on 6 spacers, top 30 mm above the base) with 15 x 8 staggered 22 mm holes, 24.5 mm apart (0.45 mm of play around a cell), each with a 1.2 mm lead-in chamfer. The cells stand 40 mm proud. Red / black / red tape and + - + marks on the front edge show the three blocks. |

### Cycle

For each hole, block by block (+, -, +), each block from the operator side (far row) to the
robot side:

1. **pick**: the robot swings over the box it needs, above the cell nearest the rack, switches the
   magnet on, and comes down on the top of the cell: quickly to 2 mm above it, waits there until
   the magnet is within 0.15 mm of the cell axis, then the last 3 mm at 1 cm/s. The magnet holds
   the cell as soon as it touches it. The robot lifts it straight out of the box, slowly
   (8 cm/s) while the cell is still among its neighbours.
2. **place**: it swings over the hole with the cell 2.5 cm above all the cell tops, lowers it to
   4 mm above the holder plate, corrects the last tenths of a millimetre from where the cell really
   hangs, and lowers it through the hole at 4 cm/s, until it is 0.5 mm above the base plate.
3. **release**: magnet off, the cell drops 0.5 mm onto the base, the magnet goes up.
4. **Rack full**: the robot goes home.

![pick](docs/pick.png)

*The magnet comes down on a + cap in the "+ up" box.*

![insert](docs/insert.png)

*Block 2: - up cells going in. The discs over the cells are the polarity check (see below).*

![rack](docs/rack_done.png)

*The full rack from above: 40 + / 40 - / 40 +. Red disc: positive terminal up; blue disc:
negative terminal up.*

### Polarity check

The viewer and the videos draw a small disc over every cell in the rack: **red** if its positive
terminal is up, **blue** if its negative terminal is up. The colour comes from the cell's real
orientation in the simulation, not from the plan, so a cell in the wrong block shows up at once.
The final report counts, per block, the cells that are the right way up. `--no-marks` hides the
discs.

### Measured (default settings, the full rack)

- All 120 cells in the rack, every one the right way up: 40/40 + up in block 1, 40/40 - up in
  block 2, 40/40 + up in block 3.
- Every cell standing on the base plate (largest tilt 0.2 deg), 0.26 mm from its hole centre on
  median, 0.50 mm at most (the hole leaves 0.45 mm of play).
- The magnet lands on the cell axis within 0.03 mm on median, 0.16 mm at most: it never touches
  a neighbouring cell (0.85 mm away).
- A cell being lifted out of its box strays at most 0.44 mm sideways while still among its
  neighbours. Along a row they are 0.3 mm apart, so a real cell would brush its neighbour now and
  then (cell-to-cell contacts are off in the simulation, see below).
- The arm never touches the table, the boxes or the rack.
- Robot cycle: 7.6 s per cell on average (8.5 s at most, the far corners of the boxes). Full rack
  (120 cells) in about 15.5 min.
- The simulation alone runs about 26x faster than real time (headless).

## Run

```bash
uv sync
uv run accuwat-sim                       # interactive viewer at real speed
uv run accuwat-sim --speed 4             # 4x faster than real time
uv run accuwat-sim --cells 30            # stop after 30 cells
uv run accuwat-sim --no-marks            # hide the polarity check discs
uv run accuwat-sim --video demo.mp4 --video-speed 4   # headless 4x MP4, main view + captions (needs ffmpeg)
uv run accuwat-sim --video rack.mp4 --camera rack_top --size 1920x1080
uv run accuwat-sim --export-xml cell.xml # standalone MJCF of the cell
uv run accuwat-sim-assets                # regenerate the procedural meshes
```

Headless rendering on a machine without a display: `MUJOCO_GL=egl uv run accuwat-sim --video demo.mp4`.

Cameras: `overview`, `photo` (roughly the viewpoint of the rack photo), `front`, `side`, `rack`,
`rack_top`, `rack_close`, `block1`, `block2`, `block3`, `box_plus`, `box_minus`.

A rendered demo of the full rack (4x) is in `media/accuwat_demo.mp4`. By default (`--camera auto`) the video
stays on the main view (`photo`) and cuts to `rack_top` once the rack is full.

## Code map

```
src/accuwat_sim/
  layout.py     all dimensions and positions (cell, boxes, rack holes and blocks, magnet, riser)
  meshgen.py    procedural meshes -> assets/meshes/*.stl (holder plate with its holes, hole lining segment)
  scene.py      builds the MuJoCo model with MjSpec (YAM attached as r/, gripper removed, magnet tool added)
  arm.py        YAM control: damped-least-squares IK, velocity / inertia / friction feed-forward, min-jerk moves, arc swings
  task.py       the filling routine (generator, one yield = one 10 ms control tick)
  sim.py        CellSim: physics stepping, electromagnet, rack report, polarity overlay
  constraints.py  switching the magnet weld on and off without jumps
  assets/yam/   I2RT YAM + linear_4310 gripper from MuJoCo Menagerie (MIT, see LICENSE); the gripper is stripped at build time
  assets/accuwat_cell.xml   exported cell (open it in MuJoCo's simulate app)
```

To change the demo, edit `layout.py`. For example:

- `BLOCKS` sets which columns get which polarity (e.g. `((0, 3, +1), (3, 6, -1), ...)` for narrower blocks);
- `RACK_COLS`, `RACK_ROWS`, `RACK_PITCH`, `HOLE_R` set the rack (the holder mesh follows);
- `CELL_R`, `CELL_H`, `CELL_MASS` set the cell (18650: 0.009, 0.065, 0.047; the holes must follow);
- `BOX_GRID`, `BOX_PLUS_CENTER`, `BOX_MINUS_CENTER` set the boxes;
- `RISER_H` and `MAGNET_FACE_Z` set the arm height and the tool length.

## Modelling choices and simplifications

- **Magnet.** Switched on, it holds the cell its face touches: a weld at the pose of the moment of
  contact stands in for the magnetic force (a 69 g cell under a 25 N magnet does not slip). The
  coil takes 60 ms to build up and to let go. No residual magnetism, and no pull on the
  neighbouring cells: a real magnet this size reaches about 1 mm, and the neighbours' tops are
  0.85 mm from its edge, so the real cell should check this (or use a magnet a little smaller than
  20 mm).
- **Hand accuracy.** The magnet needs about a tenth of a millimetre: it has 0.85 mm to the
  neighbouring cells, and a lifted cell 0.3 mm. The YAM's joint servos are soft (Menagerie gains),
  and a plain position command leaves the hand 2-9 mm off its path in motion (damping lag that is
  different for each joint, the shoulder's inertia, dry friction). The YAM's motors run in MIT
  mode (tau = kp (q_ref - q) + kd (qd_ref - qd) + tau_ff), so the controller (`arm.py`) gives them
  what a real integration would: a velocity reference, and a torque feed-forward for the inertia
  (M(q) qdd_ref from the model) and the friction. On the way down to a cell, out of the box and
  through a hole, it also corrects the commanded x, y from the measured tool position (joint
  encoders + forward kinematics). Over the hole the last correction uses where the cell really
  hangs: a real cell would use a camera, or trust the magnet's centring.
- **Cell positions.** The robot knows where every cell is from the simulator state. The boxes and
  the rack are at fixed positions, so a real cell needs only a locating stop for each (or a
  camera to check the box).
- **Sleeping.** Cells at rest sleep (MuJoCo's island sleeping): their contacts are not computed
  until the magnet touches them. This is what makes the 200 cells and the 720 hole segments cheap
  (about 26x faster than real time headless). `--no-sleep` keeps them all awake (about 0.5x real time
  with a full rack).
- **Cells do not touch each other.** In the boxes they stand 0.3 mm apart in a row, with dividers
  between the rows; in the rack the holder keeps them apart. Cell-to-cell collisions are therefore
  off, which keeps the contact count low; the report gives the largest sideways drift of a cell
  being lifted out of its box instead. The dividers are visual only.
- **Holes.** The holder plate's visual mesh has round chamfered holes. For collisions, each hole
  is lined with 6 convex segments (a hexagon whose flat faces sit on the 22 mm hole), chamfered at
  the top; their corners sit in the webs between three holes. The plate between the holes
  touches nothing.
- **Riser.** 15 cm. With the arm base on the table, the magnet cannot point straight down over the
  far box corners (joint 4 limit) at the carrying height.
- **No flipping.** The polarity comes from the box. If the cell had a single box, the robot would
  need a regrip station to turn the - cells over.

## Relation to the Claret cell

Same robot, same controller family (IK, min-jerk moves, arc swings, integral term with
anti-windup), same project layout as `../../claret/claret-sim`. New here: the electromagnet tool,
the feed-forward terms and the corrected vertical moves (the magnet needs a tenth of a millimetre),
and MuJoCo's sleeping for the many resting parts. The feed-forward in `arm.py` would make the
Claret cell's slow descents unnecessary too.
