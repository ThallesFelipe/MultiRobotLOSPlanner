"""Tests for occupancy-grid API and public core exports.

These tests validate that `MapGrid` operations preserve expected `C_free` and
`C_obs` semantics and that the stable core package API remains available.
"""

import pytest

import core
from core import (
    MapGrid,
    MapProcessingConfig,
    MapProcessor,
    compute_blocked_corridor_segment,
    extract_graph_vertices,
    has_line_of_sight,
)

GridPoint = tuple[int, int]
GridRectRange = tuple[int, int, int, int]


def test_core_init_exports_are_available() -> None:
    """Verifies that the package-level public API exports remain stable."""
    assert set(core.__all__) == {
        "MapGrid",
        "MapProcessingConfig",
        "MapProcessor",
        "compute_blocked_corridor_segment",
        "extract_graph_vertices",
        "has_line_of_sight",
    }
    assert core.MapGrid is MapGrid
    assert core.MapProcessingConfig is MapProcessingConfig
    assert core.MapProcessor is MapProcessor
    assert core.compute_blocked_corridor_segment is compute_blocked_corridor_segment
    assert core.extract_graph_vertices is extract_graph_vertices
    assert core.has_line_of_sight is has_line_of_sight

    grid = MapGrid(2, 2)
    assert has_line_of_sight(grid, (0, 0), (1, 1)) is True


def test_map_grid_starts_empty() -> None:
    """Ensures a new occupancy grid starts fully in `C_free`."""
    grid = MapGrid(3, 4)
    assert grid.rows == 3
    assert grid.cols == 4
    assert grid.grid.shape == (3, 4)
    assert grid.grid.sum() == 0


@pytest.mark.parametrize(
    ("grid_point", "expected"),
    [
        ((0, 0), True),
        ((2, 3), True),
        ((-1, 0), False),
        ((3, 0), False),
        ((0, 4), False),
    ],
)
def test_map_grid_in_bounds(grid_point: GridPoint, expected: bool) -> None:
    """Checks `in_bounds` behavior for valid and invalid occupancy-grid points."""
    grid = MapGrid(3, 4)
    assert grid.in_bounds(*grid_point) is expected


def test_add_obstacle_and_is_free() -> None:
    """Confirms single-cell transitions from `C_free` to `C_obs`."""
    grid = MapGrid(4, 4)
    assert grid.is_free(2, 1) is True
    grid.add_obstacle(2, 1)
    assert grid.is_free(2, 1) is False


def test_add_obstacle_rect_marks_expected_area() -> None:
    """Ensures rectangular obstacle insertion marks exactly the target slice."""
    grid = MapGrid(5, 5)
    grid.add_obstacle_rect(1, 4, 2, 5)

    for row_index in range(5):
        for col_index in range(5):
            is_rectangle_cell = 1 <= row_index < 4 and 2 <= col_index < 5
            assert grid.is_free(row_index, col_index) is (not is_rectangle_cell)


@pytest.mark.parametrize(
    "rect_range",
    [
        (2, 2, 1, 3),
        (3, 1, 1, 3),
        (-1, 2, 1, 3),
        (0, 6, 1, 3),
        (0, 2, 3, 3),
    ],
)
def test_add_obstacle_rect_rejects_invalid_ranges(
    rect_range: GridRectRange,
) -> None:
    """Validates that invalid `[start, end)` rectangles raise `ValueError`."""
    grid = MapGrid(5, 5)
    with pytest.raises(ValueError):
        grid.add_obstacle_rect(*rect_range)


@pytest.mark.parametrize(
    "grid_point",
    [(-1, 0), (0, -1), (5, 0), (0, 5)],
)
def test_map_grid_bounds_checks_raise_value_error(grid_point: GridPoint) -> None:
    """Ensures bounds-checked APIs raise `ValueError` for out-of-range points."""
    grid = MapGrid(5, 5)
    with pytest.raises(ValueError):
        grid.is_free(*grid_point)
    with pytest.raises(ValueError):
        grid.add_obstacle(*grid_point)


def test_map_grid_string_and_repr() -> None:
    """Checks stable string and repr representations for debugging and display."""
    grid = MapGrid(2, 2)
    grid.add_obstacle(0, 1)

    assert str(grid) == "0 1\n0 0"
    assert repr(grid) == "MapGrid(rows=2, cols=2)"
