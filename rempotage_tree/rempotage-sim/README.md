# Repotting line: two YAM arms plant young trees into 3 L pots (MuJoCo)

Demo simulation of a repotting line, after the reference video `../rempotage_yam_simu_x2.mp4`
(one YAM on a potting wheel). Here two I2RT YAM arms share the planting work:

- **arm A** pulls a young tree out of its 9 cm square nursery pot (the pots stand packed in plastic
  crates of 4 x 6 on the trolley), holding it by the trunk just above the root ball, sets it almost to
  the bottom of the planting hole of a bigger pot (7.3 of the root ball's 8.5 cm), and leaves at once
  for the next plant;
- **arm B**, whose fingertips carry **tamper feet** (a flat steel sole each), pushes the root ball
  the last 1.3 cm down into the pot with both soles, one on each side of the trunk, on the plant only
  (the hole is 2 mm narrower than the root ball, so the soil resists), and firms it on the floor of
  the hole.

The pots come from a potting wheel that fills them and punches the hole, pass under a wood-chip
mulcher and leave in rows onto a pallet. The wheel's proportions follow a photo of the customer's
machine: 12 nests around a large central soil table, on a drum, under a tall stack of empty pots.

![cell](docs/cell.png)

## The cell

| Element | What it does |
|---|---|
| **Potting wheel** | A ring of 12 nests (pot centres 0.55 m from the axis) on a silver drum, indexing 30 deg counterclockwise around a fixed hub; a U-shaped nest at each station, open to the outside for the ejection. |
| **Soil table** | A large fixed plate over the hub (0.9 m across), its rim just inside the pots and 2 cm above them, heaped with potting soil. |
| **S0 denester** | A tall stack of nested empty 3 L pots (34) in a guide cage on its own column. Two pneumatic fingers open, the bottom pot drops 3 cm into the nest, the fingers close and the stack comes down one step. |
| **S1 filler** | A curved plough over the soil table pushes soil off the table's rim into the pot (3 s, the pot goes from 0.12 to 1.4 kg). |
| **S2 dibber** | A pneumatic cylinder pushes a steel dibber shaped like the root ball (2 mm narrower) down into the soil and back: the planting hole. |
| **S3 planting** | The two robots (below). The wheel waits for their "planting done". |
| **S6 ejector** | A cylinder on the hub, under the soil table, pushes the planted pot off the wheel, over a dead plate onto the outfeed conveyor. |
| **Mulcher** | Beside the outfeed: a big bag of wood chips, a screw conveyor up to a hopper on a column, and a side spout that drops chips on each pot as it rolls by (a sensor starts and stops the feeder). |
| **Outfeed + row former** | Powered rollers (slip clutches) run the pots to a stop, where they queue. When the third pot stands still in front of the row sensor, the conveyor stops and a sweep bar pushes the row onto the pallet. |
| **Pallet** | Two rows of three: the second row pushes the first one ahead of it. Smooth tray (friction 0.25), so a row pushed at the rims slides without tipping. |
| **Plant trolley** | Two levels, two white plastic crates on each. A crate holds 24 young trees, 4 rows of 6, each in a 9 cm square nursery pot, the pots touching. The root balls: 8.5 cm tall, 8 to 6.6 cm diameter, 0.35 kg. Arm A takes from the top crate next to it, front row first; a run plants that row, the rest wait (the lower crates: the next batch, younger trees). |
| **Tray camera** | On a mast over the first crate, looking down: it locates the plants for arm A. |
| **Arm A** (left YAM, linear_4310 gripper) | On a 0.74 m pedestal: high enough for its tilted grasps. The pots touch, so there is no room for fingers beside the root ball: it holds the plant by the trunk, 14 mm above the root ball, with the hand tilted 45 deg down (the trunk passes in front of its gripper housing), and pulls it straight up out of its pot. Its hand stays over the big pot's rim while it sets the root ball deep. |
| **Arm B** (right YAM, linear_4310 gripper with tamper feet) | On a 0.68 m pedestal: high enough for its tilted push. Each fingertip carries a flat steel sole (44 x 22 x 4 mm, clamped on the pad's outer side), level when the hand is tilted 45 deg. |

Neither arm has a wrist camera: with the hand tilted, the wrist D405 (on top of the hand) would sit on
the trunk. The fixed camera over the crates takes its place.

### Planting cycle (station S3)

0. **Arm A is ready beside the pot.** While arm B was pushing the previous plant, it pulled the next
   one out of its pot in the crate (fingers closed on the trunk just above the root ball, straight up
   11 cm), and waits with it 13 cm from the hole (12 cm above the soil) while the wheel indexes.
1. **Arm A sets the plant in, almost to the bottom.** It lowers the root ball to 1 cm above the soil,
   makes a last correction from where the root ball really hangs, then pushes it 7.3 cm into the
   8.5 cm hole against the soil (up to 13 N); its fingertips stay 14 mm over the soil. Meanwhile arm B
   moves up next to the pot.
2. **Arm A lets go and leaves for the next plant** straight away.
3. **Arm B pushes the plant the last 1.3 cm down**, while arm A is already on its way to the trolley. Its hand comes in
   tilted 45 deg below horizontal, so the trunk passes in front of its gripper housing (it needs more
   than 31 deg), fingers closed around the trunk (24 mm gap): the two soles lie on the root ball's top,
   one on each side of the trunk, 12 to 34 mm from it. It goes down until the soles touch, then pushes
   the root ball down to the floor of the hole at 4 cm/s, with a force feed-forward that follows
   the soil's resistance (up to 14 N at full depth), and firms it there (22 N for 0.4 s). It pushes on
   the plant only, not on the soil beside it.
4. **Done.** The soil now holds the plant (a weld in the simulation), arm B goes back to its waiting
   pose and the line indexes at once; arm A comes back with the next plant while the wheel turns.

With `TAMP_SIDES = True` in `layout.py`, arm B also firms the soil beside the root ball after the
push: fingers open to 90 mm, the soles on the soil ring (45 to 67 mm from the trunk), two presses of
30 N turned 90 deg from each other (each press firms four of the eight soil sectors, the two together
all eight). That adds 4.3 s per plant.

Downstream, the ejector pushes the planted pot onto the outfeed, the mulcher covers its soil with
wood chips as it rolls by, and the row former puts it on the pallet.

![insert](docs/insert.png)

*Seen from behind the robots: arm A, holding the trunk with its hand tilted 45 deg, has set the root
ball 7.3 cm deep; arm B waits beside the pot.*

![push](docs/push.png)

*Arm B pushes the root ball the last 1.3 cm down with its soles, one on each side of the trunk,
while arm A is already fetching the next plant.*

![crates](docs/crates.png)

*The main video view, behind the robots: the trolley with its crates of young trees in square pots,
packed against each other; arm A takes the next plant from the front row of the first crate.*

![mulch](docs/mulch.png)

*The mulcher's spout drops wood chips on a planted pot rolling by.*

### Measured (default settings, 6 plants, one pallet)

- All 6 plants repotted, mulched, and all 6 pots on the pallet in two rows of three.
- Every root ball on the hole axis (0.3 mm), 0.4 to 0.6 mm above the hole floor, upright (tilt under
  0.2 deg), with nobody holding the trunk during the push.
- Arm A set each root ball 7.3 to 7.4 cm deep (of 8.5), its fingertips 14 mm over the soil at the lowest.
- Arm B pushed each root ball the last 1.3 cm in 0.4 s, against 13.8 N of soil resistance at the end.
- No contact of either arm with anything but the root ball, the trunk and the soil (the two arms never
  touch each other, nor the pot, the wheel, the crate or the plants waiting in it), and the plants
  waiting in the crate never move.
- Planting station: 8.5 s per plant (pot ready to planting done). With the wheel's index and arm A's
  trip to the trolley, the line makes a plant every 12.9 s; the 6 plants and the full pallet take
  1 min 50 s from the start.
- Earlier versions, for comparison: 28 s per plant at the station and 3 min 35 s in all with the
  plain fingertips, arm A holding the trunk and four soil presses; 15.5 s and 2 min 26 s with the
  soles and two soil presses; 9.4 s and 1 min 50 s with arm A setting the root ball only 1.6 cm deep
  (smaller 8-station wheel); 8.7 s with arm A holding the root ball itself (6.2 cm deep).
- The simulation runs about 5x faster than real time headless (2x while rendering the video).

## Run

```bash
uv sync
uv run rempotage-sim                       # interactive viewer at real speed
uv run rempotage-sim --speed 3             # 3x faster than real time
uv run rempotage-sim --video demo.mp4 --video-speed 2   # headless MP4: view from behind the robots + whole line + planting station
uv run rempotage-sim --video push.mp4 --camera planting_close --no-pip
uv run rempotage-sim --report report.json  # (with --video) the measurements above as JSON
uv run rempotage-sim --export-xml cell.xml # standalone MJCF of the cell
uv run rempotage-sim-assets                # regenerate the procedural meshes
```

Headless rendering on a machine without a display: `MUJOCO_GL=egl uv run rempotage-sim --video demo.mp4`.

Cameras: `behind`, `behind_close`, `overview`, `line`, `planting`, `planting_close`, `pot_top`, `robots`,
`cart`, `wheel_top`, `pallet`, `mulch`, and the camera over the crates (`tray_cam`). By default
(`--camera auto`) the video's main view stays behind the robots: the wide view (`behind`: trolley, both
arms, wheel) while arm A fetches a plant, the close one (`behind_close`) on the planting station while
they plant. The whole line and a close view of the planting station are shown on the right.

A rendered demo (2x) is in `media/rempotage_demo.mp4`.

## Code map

```
src/rempotage_sim/
  layout.py     all dimensions and positions (wheel, soil table, pot, root ball, hole, soil, tray, robots,
                line, pallet)
  meshgen.py    procedural meshes -> assets/meshes/*.stl (pot, root ball, soil layers with and without
                the hole, top soil sectors, dibber, wheel plate, leaf)
  scene.py      builds the MuJoCo model with MjSpec (two YAMs attached as a/ and b/, arm B's tamper feet,
                the soil table, the pot stack, the crates of trees and their camera, the mulcher and its big bag)
  soil.py       filling, the planting hole, the soil's resistance to the root ball, firming
  line.py       wheel cycle and stations (denester, plough filler, dibber, ejector), conveyor, mulcher,
                row former, sweep
  arm.py        YAM control: IK, min-jerk moves, arc swings, velocity / inertia / friction / force feed-forward
  task.py       the bimanual planting routine (generator, one yield = one 10 ms control tick)
  sim.py        CellSim: stepping, grasp assist, contact watch, report
  constraints.py  switching welds on and off without jumps
  assets/rempotage_cell.xml   exported cell (open it in MuJoCo's simulate app)
  assets/yam/   I2RT YAM + linear_4310 gripper from MuJoCo Menagerie (MIT, see LICENSE)
```

To change the demo, edit `layout.py`. For example:

- `N_STATIONS`, `STATION_R`, `TABLE_*` set the wheel and its soil table (the robots, the trolley and the
  outfeed follow the ring's radius);
- `RB_H`, `RB_R_TOP`, `RB_R_BOT` set the root ball, `POT_*` the pot (the meshes follow);
- `HOLE_R_TOP` / `HOLE_R_BOT` set how tight the hole is, `SOIL_F0` and `SOIL_F_PER_M` how hard the soil
  resists the root ball;
- `PRE_INSERT` sets how deep arm A sets the root ball, `GRIP_TILT` and `TRUNK_GRIP_*` how it holds the
  trunk, `PUSH_TILT` arm B's hand angle;
- `FOOT_*` set arm B's tamper soles (size, where they sit under the fingertips);
- `PUSH_SPEED`, `FIRM_FORCE`, `FIRM_TIME` in `task.py` set arm B's push; `TAMP_SIDES` turns on the
  soil presses beside the root ball (`TAMP_*` in both files);
- `CHIP_*` and `N_CHIPS` the mulcher (where, how many chips per pot);
- `CRATE_*`, `CRATES`, `CELL`, `SQPOT_*` the crates and the nursery pots, `N_PLANTS` how many plants a run
  plants (front row of the first crate first),
  `ROW_N`, `ROWS` the pallet pattern.

## Modelling choices and simplifications

- **Soil.** Not particles. The root ball does not collide with the soil; while it is in the hole, forces
  stand in for the soil (`soil.py`): along the hole, dry friction growing with depth (2 N at the top to
  14 N at full depth) as an elastic-plastic element (a stiff spring to an anchor that slips once the
  friction limit is reached), so a root ball left in the hole stays put and pushing harder drives it
  deeper; across the hole, a spring to the axis and a torque keeping it upright. The pot gets the
  opposite forces. The hole floor is a real collision surface. Once the soil is firmed, a weld holds the
  root ball in the pot.
- **Firming** (with `TAMP_SIDES`). The top 14 mm of soil around the root ball is 8 sectors. A sector pressed harder than
  3 N settles (moves down) at a rate proportional to the extra force, up to 10 mm. Its collision part
  is a 4 cm deep box, so a sole cannot go through it. Arm B lines its presses up with the sectors
  (it knows the pot's orientation from the simulator); with real soil, two presses 90 deg apart with
  these soles cover the same ring.
- **Tamper feet.** Plain boxes on the gripper pads (the sole is a collision part, the clamp block only
  visual), no mass of their own. A real version: 3 mm aluminium soles bolted to the linear_4310 pads.
- **Mulch** is visual: the chips appear on the pot as it passes under the spout (on the firmed soil,
  at its height). Their mass is neglected.
- **Filling and dibbling** are shown as soil layers appearing (and swapped for holed ones); the pot's
  mass, centre of mass and inertia change once it is full. The soil table and its heap are scenery
  (fixed; on the real machine the table turns with the ring under a fixed plough).
- **Pot stack.** The run's 6 pots are real bodies at the bottom of the stack; the 28 above them are
  scenery that steps down with them.
- **Plant.** The root ball is one rigid body; the trunk is a chain of 7 capsules with bending springs
  from the stem's stiffness; leaves are visual only. The 42 plants that stay in the top crates are
  fixed copies (their root balls and trunks still collide, so an arm brushing them would be caught).
- **Nursery pots and crates.** The square pots are visual, and the root ball slides out of its pot
  freely (in reality it is pulled out against some friction). The crate walls collide with the arms.
- **Grasp assist.** Once both of arm A's pads touch the trunk, the root ball is welded to its hand
  (`--no-assist` turns it off): the hand holding the root collar firmly. All of arm B's work is pure
  contact.
- **Positions.** The robots know where the pot, the hole and the plants are from the simulator state; a
  real cell would locate the plants in the crate with the camera over it and rely on the wheel's
  nests for the pot.
- **Pallet.** One pallet on a stand, no tractor (the reference video has one take the pallet away).

## Relation to the other cells

Same robot, controller and project layout as `../../tree_stake/my-sim` (bimanual YAM, young trees),
`../../claret/claret-sim` and `../../accuwat/accuwat-sim` (feed-forward controller). New here: the force
feed-forward for pushing, a gripper tool for pushing and firming soil, the soil model, and an indexing
machine (the wheel) the robots work with.
