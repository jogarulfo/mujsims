"""Cell layout and object dimensions (SI units, metres / radians).

World frame: z up, floor at z = 0. The robot base is at the origin on a pedestal.
x points from the robot to the carton (front), y points from the robot to the
machine chute (left). The sheet magazine is on the right (-y).

    machine  (+y, far)
       |  outlet tray, sloping down towards the robot
       v
    pick zone ---- robot ---- carton (+x)
                     |
              sheet magazine (-y)
"""

from __future__ import annotations

import numpy as np

# --- Robot (I2RT YAM, linear_4310 gripper) on a pedestal -------------------------------
BASE_Z = 0.82                        # pedestal top = arm base height (high enough to reach over the near carton wall)
ARM_HOME_Q = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
GRIPPER_OPEN = 0.0475                # per-finger travel, pad gap = 2 * q

# --- Part (black rubber-metal bushing, as in the photo) ----------------------------------
PART_R = 0.030                       # body radius
PART_RIB_R = 0.0315                  # radius over the moulded ribs
PART_BORE_R = 0.016
PART_H = 0.056
PART_COL_R = 0.0315                  # collision hull (24-sided prism) corner radius
PART_MASS = 0.18
PART_FRICTION = 0.5                  # rubber on cardboard (low enough that a push never tips a part)
BORE_FLOOR = 0.012                   # the bore's collision lining starts this far above the bottom

# --- Bore jaws (custom fingers for an inside grip) ----------------------------------------
# The YAM's stock fingertips are 22 mm wide and do not fit in the 32 mm bore. The cell uses a
# pair of narrow aluminium jaws bolted under the pads: the gripper goes into the bore closed,
# then opens, and the jaws press on the bore from the inside. Nothing sticks out around the
# part, so it can be set down right against its neighbours.
GRASP_SITE_Z = 0.1347                # YAM grasp_site (stock pad centre) on link_6
JAW_TIP = 0.052                      # jaw tip below the grasp site
JAW_W = 0.010                        # jaw width (across the closing axis)
JAW_T = 0.004                        # jaw thickness (along the closing axis)
BORE_CHAMFER = 0.004                 # lead-in chamfer at the bore mouth
MAX_LEAN = 32.0                      # deg: how far the hand may lean away from the carton walls
JAW_INSERT = 0.026                   # jaw tip depth below the part top when gripping
GRIP_SITE = JAW_TIP - JAW_INSERT / 2  # control point: middle of the inserted jaw length
GZ = PART_H - JAW_INSERT / 2         # grip point height above the part bottom (on its axis)
JAW_CLOSED = 0.0                     # finger command to go into the bore (jaws together: 8 mm thick)
JAW_GRIP = 0.022                     # finger command to grip: ~20 N per jaw (the jaws stop at JAW_HOLD)
JAW_HOLD = 0.0118                    # where the jaws stop on the bore (bore radius - JAW_T)
JAW_LOOSE = 0.003                    # after release, for the push

# --- Machine + outlet tray ----------------------------------------------------------------
# The machine finishes parts at random intervals (MACHINE_CYCLE on average). Each one drops at
# a random spot on its flat outlet plate behind the rubber curtain, and an ejector pushes it
# over the lip. The part slides down the wide stainless tray and ends up wherever it stops:
# against the end stop, or against the parts already lying there. The robot finds it there
# (a camera over the tray, here the simulator's state) and takes it.
MACHINE_CYCLE = 9.0                  # s per part, on average
CYCLE_SPREAD = 0.55                  # interval = cycle * uniform(1 - spread, 1 + spread)
CHUTE_X = 0.0                        # tray centre line
CHUTE_SLOPE = np.deg2rad(9.5)
CHUTE_FRICTION = 0.14                # polished stainless (tan 9.5 deg = 0.167 > 0.14)
STOP_Y = 0.30                        # end stop face, where it meets the tray floor
NEST_Z = 0.75                        # tray surface height at the end stop
CHUTE_LEN = 0.68                     # from the end stop to the machine front (horizontal)
CHUTE_W = 0.34                       # tray width between the side walls
CHUTE_WALL_H = 0.08
MACHINE_FRONT_Y = STOP_Y + CHUTE_LEN
SPAWN_X = CHUTE_W / 2 - PART_RIB_R - 0.03  # parts drop anywhere in +-SPAWN_X on the outlet plate
TRAY_MAX = 6                         # the machine holds while this many parts lie on the tray


def chute_z(y: float) -> float:
    """Tray surface height at y (uniform slope from the end stop up to the machine)."""
    y = min(max(y, STOP_Y), MACHINE_FRONT_Y)
    return NEST_Z + (y - STOP_Y) * np.tan(CHUTE_SLOPE)


OUTLET_Z = chute_z(MACHINE_FRONT_Y)  # flat outlet plate inside the machine
OUTLET_DEPTH = 0.20                  # outlet plate from the machine front into the machine
SPAWN_Y = MACHINE_FRONT_Y + 0.14     # where a finished part lands on the outlet plate
EJECT_STROKE = 0.18                  # ejector travel (pushes the part over the lip)
CURTAIN_Y = MACHINE_FRONT_Y + 0.03
CHUTE_FULL_Y = MACHINE_FRONT_Y - 0.10  # "tray full" eye: the machine waits if a part sits here
MACHINE_W = 0.95
MACHINE_D = 0.85
MACHINE_H = 1.75

# --- Carton (on a roller stand, lined with a plastic bag) -----------------------------------
# Square packing, parts touching each other and the walls. The carton is sized for the grid
# with 8 mm of play per direction; the robot pushes every part into the far +y corner of its
# pocket, so the play ends up along the near and -y walls.
GRID = (4, 5)                        # parts per layer: rows (along x) x columns (along y)
PITCH = 2 * PART_RIB_R               # parts touch: centre distance = diameter over the ribs
BOX_PLAY = 0.008
BOX_CENTER = np.array([0.33, 0.0])
BOX_IN = np.array([GRID[0] * PITCH + BOX_PLAY, GRID[1] * PITCH + BOX_PLAY])  # inner size (x, y)
BOX_WALL = 0.006                     # double-wall cardboard
BOX_H = 0.22                         # wall height above the inner floor
BOX_FLOOR_Z = 0.58                   # inner floor height (carton on a roller stand)
BOX_FRICTION = 0.5                   # carton floor and sheets
N_LAYERS = 3                         # 3 layers + 4 sheets fill the carton to 2 cm below the rim
PUSH_DIR = np.array([1.0, 1.0]) / np.sqrt(2)  # parts are pushed towards the far (+x), left (+y) corner
SET_DOWN_OFFSET = 0.012              # set down this far short of the pocket, then push


def slot_xy(row: int, col: int) -> np.ndarray:
    """Packed position of a part (bottom centre). Row 0 is the far row, column 0 is on the
    +y side; both are filled first, every part touching the far and +y neighbours (or walls)."""
    x = BOX_CENTER[0] + BOX_IN[0] / 2 - PITCH / 2 - row * PITCH
    y = BOX_CENTER[1] + BOX_IN[1] / 2 - PITCH / 2 - col * PITCH
    return np.array([x, y])


def set_down_xy(row: int, col: int, clearance: float = 0.003) -> np.ndarray:
    """Where the part is set down before the push: short of its pocket along the push
    direction, but never closer than `clearance` to the near or -y wall."""
    xy = slot_xy(row, col) - SET_DOWN_OFFSET * PUSH_DIR
    lo = BOX_CENTER - BOX_IN / 2 + PITCH / 2 + clearance
    return np.maximum(xy, lo)


# --- Cardboard layer sheets (slip sheets) + magazine ----------------------------------------
SHEET = BOX_IN - 0.016               # sheet size in the carton (x, y): 8 mm play each side
SHEET_T = 0.003
SHEET_MASS = 0.06
N_SHEETS = 6                         # sheets loaded in the magazine
MAG_CENTER = np.array([-0.06, -0.33])  # sheets lie with their long side along x here
MAG_BASE_Z = 0.66                    # magazine floor (top of the tray plate)

# --- Vacuum cup on the gripper (retractable, for the sheets) --------------------------------
# A pneumatic cylinder on the side of the gripper carries a suction cup. Retracted, the cup is
# well above the jaws; extended, it reaches 1 cm below the jaw tips to pick a sheet.
CUP_OFFSET = 0.062                   # cup axis from the grasp axis (towards grasp-frame +x)
CUP_R = 0.022
CUP_RETRACTED = -0.030               # cup lip along the approach axis, relative to grasp_site
CUP_EXTENDED = JAW_TIP + 0.010
