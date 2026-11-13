"""Cell layout and object dimensions (SI units, metres / radians).

World frame: z up, origin on the floor under the wheel centre. The robots stand on the -y side
of the wheel, the plant cart on their left (-x), the pallet line on the +x side.
"""

from __future__ import annotations

import numpy as np

# --- Potting wheel ---------------------------------------------------------------------------
# Proportions after the customer's machine (photo): a ring of 12 pot nests around a large central
# soil table, on a drum-shaped base, the pot stack standing tall over its station. The ring indexes
# counterclockwise (seen from above). Stations, in the order a pot sees them:
#   S0 denester  : an empty pot drops from the stack into the nest
#   S1 filler    : a plough pushes soil off the soil table into the pot
#   S2 dibber    : a pneumatic dibber punches the planting hole
#   S3 planting  : arm A sets the plant deep in the hole, arm B pushes it down to the floor
#   S4, S5       : (free)
#   S6 ejector   : a cylinder on the hub pushes the pot out onto the outfeed conveyor
#   S7 .. S11    : (free, empty nests coming back)
N_STATIONS = 12
STATION_PITCH = 2 * np.pi / N_STATIONS
S_DENEST, S_FILL, S_DIBBLE = 0, 1, 2
PLANT_STATION = 3
S_EJECT = 6
PLANT_ANGLE = -np.pi / 2               # S3 faces the robots (-y)
WHEEL_Z = 0.72                         # wheel plate top (the pots stand on it)
WHEEL_T = 0.02                         # plate thickness
STATION_R = 0.55                       # pot centre distance from the wheel axis
WHEEL_R = STATION_R + 0.12             # plate outer radius
WHEEL_R_IN = STATION_R - 0.15          # plate inner radius (fixed hub inside)
HUB_R = WHEEL_R_IN - 0.015
INDEX_TIME = 2.2                       # s per 30 deg index
# Everything around the wheel (robots, cart, outfeed) stands this much further out than on the
# first, smaller wheel (pot ring radius 0.40 m).
RING_SHIFT = STATION_R - 0.40
NEST_GUIDE_H = 0.022                   # U-nest guide height above the plate
NEST_CLEAR = 0.004                     # radial play between the pot bottom and the nest guides


def station_angle(k: int) -> float:
    return PLANT_ANGLE + (k - PLANT_STATION) * STATION_PITCH


def station_xy(k: int) -> np.ndarray:
    a = station_angle(k)
    return STATION_R * np.array([np.cos(a), np.sin(a)])


# --- Pot (3 L nursery container) -----------------------------------------------------------
POT_H = 0.16
POT_R_BOT = 0.064                      # outer radius at the bottom
POT_R_TOP = 0.084                      # outer radius at the top (below the lip)
POT_WALL = 0.0025
POT_LIP_R = 0.087
POT_LIP_H = 0.008
POT_MASS = 0.12                        # empty
SOIL_DENSITY = 450.0                   # kg/m3, loose potting soil
SOIL_Z = 0.145                         # soil surface above the pot bottom, once filled
POT_PITCH = 2 * POT_LIP_R              # two pots touching at the lip
NEST_PITCH = 0.03                      # height step of two nested empty pots in the denester
STACK_EXTRA = 28                       # more empty pots in the stack above the run's pots (scenery)
# Central soil table (fixed, on the hub): its rim just inside the pot ring, a little above the pots.
TABLE_R = STATION_R - POT_LIP_R - 0.012
TABLE_Z = WHEEL_Z + 0.16 + 0.02        # table top (pot height + 2 cm)


def pot_outer_radius(z: float) -> float:
    return POT_R_BOT + (POT_R_TOP - POT_R_BOT) * np.clip(z / POT_H, 0.0, 1.0)


def soil_mass() -> float:
    r0, r1 = POT_R_BOT - POT_WALL, pot_outer_radius(SOIL_Z) - POT_WALL
    return float(SOIL_DENSITY * np.pi * SOIL_Z * (r0 * r0 + r0 * r1 + r1 * r1) / 3)


# --- Root ball (the plant comes out of a 9 cm nursery pot) ----------------------------------
RB_H = 0.085
RB_R_TOP = 0.040
RB_R_BOT = 0.033
RB_MASS = 0.35
# Planting hole punched by the dibber: same taper, 2 mm narrower (the root ball is pressed in).
HOLE_R_TOP = RB_R_TOP - 0.002
HOLE_R_BOT = RB_R_BOT - 0.002
HOLE_DEPTH = RB_H                      # root ball top flush with the soil once pushed down
HOLE_FLOOR_Z = SOIL_Z - HOLE_DEPTH     # above the pot bottom


def rb_radius(h: float) -> float:
    """Root ball radius at height h above its bottom."""
    return RB_R_BOT + (RB_R_TOP - RB_R_BOT) * np.clip(h / RB_H, 0.0, 1.0)


# --- Soil (see soil.py) -----------------------------------------------------------------------
# Pushing the root ball into the hole: dry friction that grows with depth (the hole is 2 mm
# narrower), elastic-plastic model. A 0.35 kg root ball stays where it is left.
SOIL_F0 = 2.0                          # N, resistance as the root ball enters the hole
SOIL_F_PER_M = 140.0                   # N per metre of depth (about 14 N at full depth)
SOIL_K = 8000.0                        # N/m, elastic part before the soil gives way
SOIL_C = 40.0                          # N s/m
SOIL_K_LAT = 3000.0                    # N/m, hole wall keeping the root ball on the hole axis
SOIL_K_ROT = 3.0                       # N m/rad, keeping it upright
# Tamping: the soil around the root ball settles under the fingertips.
TAMP_YIELD = 3.0                       # N per sector before the soil settles
TAMP_RATE = 0.004                      # m/s per N above yield
TAMP_MAX = 0.010                       # m, fully firmed
TAMP_SECTORS = 8
TAMP_R = (RB_R_TOP + 0.003, POT_R_TOP - POT_WALL - 0.004)   # soil ring firmed by the fingertips
TAMP_SIDES = False                     # arm B also presses the soil beside the root ball (off: only the plant)

# --- Young tree (flexible trunk standing in the root ball) ----------------------------------
TREE_SEGMENTS = 7
TREE_SEG_LEN = 0.065                   # 45 cm of trunk above the root ball
TREE_R_BASE = 0.0065
TREE_R_TOP = 0.0035

# --- Plant trolley (on the left of arm A) ------------------------------------------------------
# The young trees come in 9 cm square nursery pots, packed against each other in plastic crates
# of CRATE_ROWS x CRATE_COLS. The trolley carries CRATES crates on top (the first one next to arm A)
# and as many below. Arm A pulls the plants out of their pots, front row of the first crate first;
# a run plants N_PLANTS of them, the rest wait in the crates.
CRATE_ROWS = 4
CRATE_COLS = 6
CRATES = 2
CELL = 0.09                            # pot pitch in a crate (the pots touch)
CRATE_WALL = 0.006
CRATE_H = 0.075                        # crate walls, above its floor
CRATE_GAP = 0.03                       # between two crates on the trolley
SQPOT_TOP = 0.0445                     # square nursery pot: half width at the top
SQPOT_BOT = 0.036                      # at the bottom
SQPOT_H = 0.089                        # height (rim 4 mm over the root ball top)
N_PLANTS = 6                           # planted in one run (the front row)
CART_X = -0.80                         # front row root ball centres, along y
CART_Y0 = -0.43 - RING_SHIFT
CART_Z = WHEEL_Z + 0.0                 # root ball bottoms (the crate floor and the pot bottoms below)
CRATE_PITCH_Y = CRATE_COLS * CELL + 2 * CRATE_WALL + CRATE_GAP


def tray_xy(row: int, col: int, crate: int = 0) -> np.ndarray:
    return np.array([CART_X - row * CELL, CART_Y0 - col * CELL - crate * CRATE_PITCH_Y])


def cart_xy(k: int) -> np.ndarray:
    """Place of plant k: the first crate's front row first, then its next rows, then the next crate."""
    per = CRATE_ROWS * CRATE_COLS
    crate, i = divmod(k, per)
    return tray_xy(i // CRATE_COLS, i % CRATE_COLS, crate)


# --- Robots (I2RT YAM, linear_4310 gripper) on pedestals ---------------------------------------
ARM_A_Z = 0.74                         # pedestal tops (base plates). Arm A: high, for its tilted grasps
ARM_B_Z = 0.68                         # arm B: for its tilted push
ARM_A_BASE = np.array([-0.40, -0.74 - RING_SHIFT])   # left: takes the plant from the cart, sets it deep in the hole
ARM_B_BASE = np.array([0.36, -0.74 - RING_SHIFT])   # right: pushes the root ball down to the floor of the hole
ARM_A_YAW = np.deg2rad(130.0)          # base yaw (the YAM's joint 1 zero direction)
ARM_B_YAW = np.deg2rad(137.0)          # towards the planting station
ARM_HOME_Q = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
GRIPPER_OPEN = 0.0475                  # per-finger travel, pad gap = 2 * q
GRIP_TILT = np.deg2rad(45.0)           # arm A's fingers point down this far below horizontal: it holds the
                                       # plant by the trunk just above the root ball (the pots touch: no room
                                       # for fingers beside the root ball), its hand over the pot rim, so it
                                       # can set the root ball deep (the trunk passes in front of its gripper
                                       # housing: needs > 31 deg)
TRUNK_GRIP_ABOVE = 0.014               # arm A's grasp site this far above the root ball top (pads on the trunk)
TRUNK_GRIP_OPEN = 0.016                # arm A's finger travel coming in around the trunk (32 mm gap)
PRE_INSERT = 0.075                     # arm A sets the root ball this deep into the hole (of 8.5 cm); its
                                       # fingertips (3 mm above the root ball top) stay over the soil
PUSH_TILT = np.deg2rad(45.0)           # arm B's fingers point down this far below horizontal (the trunk
                                       # then passes in front of its gripper housing: needs > 31 deg)
PUSH_YAW = np.deg2rad(0.0)             # arm B's push approach turned from its radial direction (away from A)
# Arm B's tamper feet: a flat steel sole under each fingertip, level when the hand is tilted by
# PUSH_TILT. Fingers closed on the trunk, they push on the root ball top; fingers open, they press
# the soil ring around it (one sole each side of the trunk).
FOOT_LEN = 0.044                       # along the approach (horizontal)
FOOT_W = 0.022                         # outwards from the pad's inner face
FOOT_T = 0.004                         # sole thickness
FOOT_FWD = -0.004                      # sole centre from the grasp site along the approach (- : behind)
FOOT_TOP = 0.0115                      # sole top below the grasp site (fingertip's lowest corner: 12 mm)

# --- Pot denester (S0), soil filler (S1), dibber (S2), ejector (S5) ----------------------------
N_POTS = N_PLANTS
DENEST_DROP = 0.03                     # the bottom pot of the stack hangs this far above the plate
FILL_TIME = 3.0
EJECT_STROKE = 0.25
EJECT_REST_R = STATION_R - 0.10        # pusher plate at rest, from the wheel axis
EJECT_Z = 0.05                         # pusher plate centre above the wheel plate

# --- Outfeed conveyor, row former and pallet (+x side) ------------------------------------------
CONV_Y = 0.0                           # lane centre (S5 is at angle 0)
CONV_X = (0.555 + RING_SHIFT, 1.40 + RING_SHIFT)    # roller span
CONV_Z = WHEEL_Z                       # roller tops level with the wheel plate
ROLLER_R = 0.015
ROLLER_PITCH = 0.04
LANE_W = 0.18
CONV_SPEED = 0.12                      # m/s
DRIVE_TORQUE = 0.02                    # N m slip-clutch limit per roller
ROW_STOP_X = CONV_X[1] + 0.012         # stop face at the conveyor end
ROW_N = 3                              # pots per row
ROWS = 2                               # rows per pallet
SWEEP_STROKE = 0.245                   # bar travel: 5.5 cm to the pots, then the row onto the pallet
SWEEP_Z = 0.05
PALLET_X = (ROW_STOP_X - 0.62, ROW_STOP_X + 0.02)
PALLET_Y = (LANE_W / 2 + 0.006, LANE_W / 2 + 0.006 + 0.40)
PALLET_Z = CONV_Z - 0.002              # pallet top, a hair below the roller tops
PALLET_FRICTION = 0.25                 # smooth plastic tray: rows slide without tipping


# Wood-chip mulcher over the outfeed: a side spout drops chips on each pot as it rolls by.
CHIP_X = 0.72 + RING_SHIFT             # spout over the lane at this x
CHIP_SPOUT_Y = 0.055                   # spout outlet beside the trunk, over the soil ring
CHIP_ZONE = 0.07                       # +- x around the spout where a passing pot is mulched
N_CHIPS = 48                           # chips per pot


def row_x(i: int) -> float:
    """Pot centre x of place i in a row (0 against the stop)."""
    return ROW_STOP_X - POT_LIP_R - i * POT_PITCH
