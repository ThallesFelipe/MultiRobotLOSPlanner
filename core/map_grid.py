"""Occupancy-grid primitives for LOS-constrained motion planning.

This module provides a 2D occupancy grid where each cell belongs to either
free configuration space (`C_free`) or obstacle space (`C_obs`).
"""

import numpy as np
import numpy.typing as npt

# Value used for cells in free configuration space (C_free).
FREE_SPACE_VALUE: int = 0

# Value used for cells in occupied configuration space (C_obs).
OBSTACLE_VALUE: int = 1

GridPoint = tuple[int, int]


class MapGrid:
    """2D occupancy grid with `C_free` and `C_obs` cell states.

    Args:
        rows: Number of rows in the occupancy grid.
        cols: Number of columns in the occupancy grid.

    Raises:
        ValueError: If `rows` or `cols` is not a positive integer.
    """

    def __init__(self, rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(
                "Grid dimensions must be positive integers; "
                f"received rows={rows}, cols={cols}."
            )

        self.rows = rows
        self.cols = cols
        self.grid: npt.NDArray[np.int_] = np.zeros((rows, cols), dtype=int)

    def in_bounds(self, row: int, col: int) -> bool:
        """Checks whether a grid coordinate is within occupancy-grid limits.

        Args:
            row: Row index in the occupancy grid.
            col: Column index in the occupancy grid.

        Returns:
            `True` if the point is inside the grid bounds, otherwise `False`.
        """
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _check_bounds(self, row: int, col: int) -> None:
        """Validates that a grid coordinate is inside the occupancy grid.

        Args:
            row: Row index in the occupancy grid.
            col: Column index in the occupancy grid.

        Raises:
            ValueError: If `(row, col)` is outside the map bounds.
        """
        if not self.in_bounds(row, col):
            raise ValueError(
                "Grid position out of bounds: "
                f"(row={row}, col={col}) is not inside a "
                f"{self.rows}x{self.cols} occupancy grid."
            )

    def add_obstacle(self, row: int, col: int) -> None:
        """Marks one occupancy-grid cell as obstacle space (`C_obs`).

        Args:
            row: Row index of the cell to mark as obstacle.
            col: Column index of the cell to mark as obstacle.

        Raises:
            ValueError: If `(row, col)` is outside the map bounds.
        """
        self._check_bounds(row, col)
        self.grid[row, col] = OBSTACLE_VALUE

    def add_obstacle_rect(
        self,
        r_start: int,
        r_end: int,
        c_start: int,
        c_end: int,
    ) -> None:
        """Marks a rectangular slice of cells as obstacle space (`C_obs`).

        The rectangle follows Python slice semantics `[start, end)` for rows and
        columns.

        Args:
            r_start: Inclusive start row index.
            r_end: Exclusive end row index.
            c_start: Inclusive start column index.
            c_end: Exclusive end column index.

        Raises:
            ValueError: If the rectangle does not satisfy `[start, end)` bounds
                and monotonicity constraints.
        """
        if not (
            0 <= r_start < r_end <= self.rows
            and 0 <= c_start < c_end <= self.cols
        ):
            raise ValueError(
                "Invalid rectangle: expected [start, end) ranges with "
                "start < end and all limits inside the occupancy grid."
            )

        self.grid[r_start:r_end, c_start:c_end] = OBSTACLE_VALUE

    def is_free(self, row: int, col: int) -> bool:
        """Checks whether a cell belongs to free space (`C_free`).

        Args:
            row: Row index of the queried cell.
            col: Column index of the queried cell.

        Returns:
            `True` if the cell is free (`C_free`), otherwise `False`.

        Raises:
            ValueError: If `(row, col)` is outside the map bounds.
        """
        self._check_bounds(row, col)
        return bool(self.grid[row, col] == FREE_SPACE_VALUE)

    def __str__(self) -> str:
        """Builds a human-readable occupancy-grid matrix representation.

        Returns:
            Row-wise string representation of the occupancy grid.
        """
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)

    def __repr__(self) -> str:
        """Builds a concise debug representation for the grid object.

        Returns:
            Constructor-style representation of this `MapGrid`.
        """
        return f"MapGrid(rows={self.rows}, cols={self.cols})"