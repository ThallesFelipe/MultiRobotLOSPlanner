"""Line-of-sight utilities over occupancy-grid maps.

The functions in this module evaluate whether two occupancy-grid points are
mutually visible under LOS constraints, accounting for intersections with
obstacle space (`C_obs`).
"""

from collections.abc import Iterator
from typing import Literal

from .map_grid import GridPoint, MapGrid

# Default number of allowed C_obs intersections when validating LOS.
DEFAULT_LOS_TOLERANCE: int = 0

# Corner-cutting policy for diagonal transitions:
# - "both": block only when both diagonal flank cells are in C_obs.
# - "either": block when at least one diagonal flank cell is in C_obs.
DiagonalFlankPolicy = Literal["both", "either"]
DEFAULT_DIAGONAL_FLANK_POLICY: DiagonalFlankPolicy = "either"


def bresenham(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Iterator[GridPoint]:
    """Yields occupancy-grid cells intersected by a segment between two points.

    Args:
        x1: Start row index.
        y1: Start column index.
        x2: End row index.
        y2: End column index.

    Yields:
        Grid points `(row, col)` traversed by the discrete segment.
    """
    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = delta_x - delta_y

    current_row, current_col = x1, y1
    while True:
        yield (current_row, current_col)
        if current_row == x2 and current_col == y2:
            break

        doubled_error = error * 2
        if doubled_error > -delta_y:
            error -= delta_y
            current_row += step_x
        if doubled_error < delta_x:
            error += delta_x
            current_col += step_y


def _is_blocked_diagonal_transition(
    grid: MapGrid,
    previous_point: GridPoint,
    current_point: GridPoint,
    diagonal_flank_policy: DiagonalFlankPolicy,
) -> bool:
    """Checks whether a diagonal Bresenham step is blocked by flank obstacles."""
    row_delta = current_point[0] - previous_point[0]
    col_delta = current_point[1] - previous_point[1]
    if abs(row_delta) != 1 or abs(col_delta) != 1:
        return False

    flank_a = (previous_point[0], current_point[1])
    flank_b = (current_point[0], previous_point[1])
    flank_a_is_obstacle = not grid.is_free(*flank_a)
    flank_b_is_obstacle = not grid.is_free(*flank_b)

    if diagonal_flank_policy == "either":
        return flank_a_is_obstacle or flank_b_is_obstacle

    return flank_a_is_obstacle and flank_b_is_obstacle


def has_line_of_sight(
    grid: MapGrid,
    p1: GridPoint,
    p2: GridPoint,
    tolerance: int = DEFAULT_LOS_TOLERANCE,
    diagonal_flank_policy: DiagonalFlankPolicy = DEFAULT_DIAGONAL_FLANK_POLICY,
) -> bool:
    """Checks whether two points are LOS-connected across `C_free` cells.

    A LOS query remains valid while the number of segment intersections with
    obstacle space (`C_obs`) does not exceed `tolerance`.

    Args:
        grid: Occupancy grid containing `C_free` and `C_obs` cells.
        p1: Start point as `(row, col)`.
        p2: End point as `(row, col)`.
        tolerance: Maximum number of `C_obs` intersections allowed.
        diagonal_flank_policy: Rule applied on diagonal steps to prevent
            corner cutting. Uses "both" to block only when both diagonal
            flank cells are obstacles, or "either" to block when at least
            one flank is an obstacle.

    Returns:
        `True` if the segment from `p1` to `p2` satisfies the LOS constraint,
        otherwise `False`.

    Raises:
        ValueError: If `tolerance` is negative or policy is invalid.
    """
    if tolerance < 0:
        raise ValueError(
            "LOS tolerance must be greater than or equal to 0; "
            f"received tolerance={tolerance}."
        )

    if diagonal_flank_policy not in ("both", "either"):
        raise ValueError(
            "Diagonal flank policy must be either 'both' or 'either'; "
            f"received diagonal_flank_policy={diagonal_flank_policy!r}."
        )

    if not grid.in_bounds(*p1) or not grid.in_bounds(*p2):
        return False

    obstacle_intersections = 0
    previous_point: GridPoint | None = None
    for row_index, col_index in bresenham(*p1, *p2):
        current_point: GridPoint = (row_index, col_index)

        if (
            previous_point is not None
            and _is_blocked_diagonal_transition(
                grid,
                previous_point,
                current_point,
                diagonal_flank_policy,
            )
        ):
            obstacle_intersections += 1
            if obstacle_intersections > tolerance:
                return False

        if not grid.is_free(row_index, col_index):
            obstacle_intersections += 1
            if obstacle_intersections > tolerance:
                return False

        previous_point = current_point

    return True