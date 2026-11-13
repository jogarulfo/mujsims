"""Cell layout and object dimensions (SI units, metres / radians).

World frame: z up, floor at z = 0. The robot base is at the origin, bolted on the table.
x points from the robot to the rack (front), y to the left.

             "+ up" box (+y)
                  |
    robot ----- rack (+x, long side along y)
                  |
             "- up" box (-y)

The rack holds 15 x 8 cells in a staggered (honeycomb) pattern. Its 15 columns run along y and
are split in three blocks of 5 columns: + up, - up, + up (column 0 is on the +y side).
"""

from __future__ import annotations

import numpy as np

# --- Table + robot (I2RT YAM, gripper replaced by an electromagnet) ---------------------
TABLE_Z = 0.75                       # worktop height
TABLE_SIZE = np.array([1.30, 1.40])  # (x, y)
TABLE_CENTER = np.array([0.30, 0.0])
RISER_H = 0.15                       # steel riser block under the arm: with the arm base level with
                                     # the cells, joint 4 runs out of range at the far box corners
BASE_Z = TABLE_Z + RISER_H
ARM_HOME_Q = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])

# --- Electromagnet tool (on the YAM's wrist flange; link_6 frame, z along the approach) ----
FLANGE_Z = 0.035                     # wrist flange face on link_6
ADAPTER_R, ADAPTER_T = 0.031, 0.008  # aluminium adapter disc
STEM_R = 0.008                       # stand-off rod
MAGNET_R = 0.010                     # 20 mm x 15 mm holding electromagnet, 12 V, ~25 N
MAGNET_H = 0.015
MAGNET_FACE_Z = 0.098                # magnet face below the link_6 origin (tool length)
MAGNET_RISE = 0.06                   # s for the coil current to build up / decay

# --- Cell (21700 Li-ion) -----------------------------------------------------------------
CELL_R = 0.01055                     # 21.1 mm
CELL_H = 0.0704                      # 70.4 mm
CELL_MASS = 0.069
CELL_FRICTION = 0.4                  # PVC wrap

# --- Cell boxes (as shipped: cells standing, all the same way up) --------------------------
BOX_GRID = (10, 10)                  # cells per box: rows (along x) x columns (along y)
BOX_PITCH_Y = 0.0214                 # along a row: cells nearly touching (0.3 mm gap)
BOX_PITCH_X = 0.0226                 # across rows: 1 mm cardboard divider + 0.5 mm play
BOX_PLAY = 0.002
BOX_IN = np.array([BOX_GRID[0] * BOX_PITCH_X + BOX_PLAY, BOX_GRID[1] * BOX_PITCH_Y + BOX_PLAY])
BOX_WALL = 0.004
BOX_FLOOR_T = 0.003
BOX_H = 0.069                        # inner height: the cell tops stand 1.4 mm proud of the rim
BOX_FLOOR_Z = TABLE_Z + BOX_FLOOR_T
BOX_PLUS_CENTER = np.array([0.13, 0.31])    # cells + up (positive terminal on top)
BOX_MINUS_CENTER = np.array([0.13, -0.31])  # cells - up
BOX_FRICTION = 0.5


def box_cell_xy(center, row: int, col: int) -> np.ndarray:
    """Centre of a cell in a box. Row 0 is the far (+x) row, column 0 on the +y side."""
    x = center[0] + (BOX_GRID[0] - 1) / 2 * BOX_PITCH_X - row * BOX_PITCH_X
    y = center[1] + (BOX_GRID[1] - 1) / 2 * BOX_PITCH_Y - col * BOX_PITCH_Y
    return np.array([x, y])


# --- Rack (cell holder plate, staggered holes) ----------------------------------------------
RACK_COLS, RACK_ROWS = 15, 8         # holes per row (along y) x rows (along x)
RACK_PITCH = 0.0245                  # hole centre distance along a row (3.4 mm web between cells)
RACK_ROW = RACK_PITCH * np.sqrt(3) / 2  # row spacing: honeycomb
RACK_CENTER = np.array([0.37, 0.0])
HOLE_R = 0.0110                      # 22.0 mm hole: 0.45 mm radial play
HOLE_CHAMFER = 0.0012                # lead-in chamfer at the top of each hole
RACK_MARGIN = 0.019                  # plate edge beyond the outermost hole centres
RACK_BASE_T = 0.006                  # base plate the cells stand on
RACK_BASE_Z = TABLE_Z + RACK_BASE_T  # top of the base plate
HOLDER_T = 0.010                     # holder plate thickness
HOLDER_TOP = RACK_BASE_Z + 0.030     # holder top 30 mm above the base: cells stand 40 mm proud
RACK_FRICTION = 0.4
# Thirds along the long side: column -> polarity facing up (+1: positive terminal up).
BLOCKS = ((0, 5, +1), (5, 10, -1), (10, 15, +1))


def rack_polarity(col: int) -> int:
    for c0, c1, pol in BLOCKS:
        if c0 <= col < c1:
            return pol
    raise ValueError(col)


def rack_hole_xy(row: int, col: int) -> np.ndarray:
    """Centre of a rack hole. Row 0 is the far (+x) row, column 0 on the +y side.
    Odd rows are shifted by half a pitch towards -y."""
    x = RACK_CENTER[0] + (RACK_ROWS - 1) / 2 * RACK_ROW - row * RACK_ROW
    y = RACK_CENTER[1] + (RACK_COLS - 1 + 0.5) / 2 * RACK_PITCH - (col + 0.5 * (row % 2)) * RACK_PITCH
    return np.array([x, y])


RACK_PLATE = np.array([(RACK_ROWS - 1) * RACK_ROW + 2 * RACK_MARGIN,
                       (RACK_COLS - 0.5) * RACK_PITCH + 2 * RACK_MARGIN])  # holder plate size (x, y)

# --- Motion ---------------------------------------------------------------------------------
CLEAR = 0.025                        # held cell bottom above every cell top when travelling
TOPS = max(BOX_FLOOR_Z, RACK_BASE_Z) + CELL_H
TRANSIT_Z = TOPS + CLEAR + CELL_H    # magnet face height when carrying a cell
APPROACH = 0.015                     # slow down this far above contact / the holder
INSERT_GAP = 0.0005                  # release the cell this far above the base plate
