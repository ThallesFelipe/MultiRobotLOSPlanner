"""Tests for Bresenham rasterization and LOS validation on occupancy grids."""

import pytest

from core.map_grid import MapGrid
from core.visibility import bresenham, has_line_of_sight

GridPoint = tuple[int, int]


@pytest.mark.parametrize(
    ("start_point", "end_point", "expected_points"),
    [
        ((0, 0), (0, 3), [(0, 0), (0, 1), (0, 2), (0, 3)]),
        ((0, 0), (3, 0), [(0, 0), (1, 0), (2, 0), (3, 0)]),
        ((0, 0), (3, 3), [(0, 0), (1, 1), (2, 2), (3, 3)]),
        ((3, 3), (0, 0), [(3, 3), (2, 2), (1, 1), (0, 0)]),
    ],
)
def test_bresenham_expected_points(
    start_point: GridPoint,
    end_point: GridPoint,
    expected_points: list[GridPoint],
) -> None:
    """Verifies deterministic rasterization for representative line segments."""
    assert list(bresenham(*start_point, *end_point)) == expected_points


def test_bresenham_same_point() -> None:
    """Checks that a degenerate segment yields a single occupancy-grid point."""
    assert list(bresenham(2, 2, 2, 2)) == [(2, 2)]


def test_line_of_sight_open_space_is_true() -> None:
    """Ensures LOS succeeds when all sampled cells remain in `C_free`."""
    grid = MapGrid(10, 10)
    assert has_line_of_sight(grid, (0, 0), (9, 9)) is True


def test_line_of_sight_blocked_horizontal_is_false() -> None:
    """Ensures LOS fails when the segment intersects obstacle space `C_obs`."""
    grid = MapGrid(10, 10)
    grid.add_obstacle_rect(5, 6, 4, 7)
    assert has_line_of_sight(grid, (5, 0), (5, 9)) is False


def test_line_of_sight_tolerance_controls_block_count() -> None:
    """Validates LOS tolerance as the allowed number of `C_obs` intersections."""
    grid = MapGrid(10, 10)
    grid.add_obstacle(5, 3)
    grid.add_obstacle(5, 7)

    assert has_line_of_sight(grid, (5, 0), (5, 9), tolerance=0) is False
    assert has_line_of_sight(grid, (5, 0), (5, 9), tolerance=1) is False
    assert has_line_of_sight(grid, (5, 0), (5, 9), tolerance=2) is True


def test_line_of_sight_same_point_respects_obstacle_and_tolerance() -> None:
    """Checks LOS behavior for a single-point segment under obstacle tolerance."""
    grid = MapGrid(10, 10)
    grid.add_obstacle(2, 2)

    assert has_line_of_sight(grid, (2, 2), (2, 2), tolerance=0) is False
    assert has_line_of_sight(grid, (2, 2), (2, 2), tolerance=1) is True


def test_line_of_sight_blocked_on_endpoint() -> None:
    """Ensures endpoint occupancy contributes to LOS blocking rules."""
    grid = MapGrid(10, 10)
    grid.add_obstacle(9, 9)

    assert has_line_of_sight(grid, (0, 0), (9, 9), tolerance=0) is False
    assert has_line_of_sight(grid, (0, 0), (9, 9), tolerance=1) is True


def test_line_of_sight_blocks_corner_cut_when_both_diagonal_flanks_are_obstacles() -> None:
    """Prevents diagonal crossing through a closed corner-touching obstacle pair."""
    grid = MapGrid(3, 3)
    grid.add_obstacle(1, 0)
    grid.add_obstacle(0, 1)

    assert has_line_of_sight(grid, (0, 0), (1, 1), tolerance=0) is False
    assert has_line_of_sight(grid, (0, 0), (1, 1), tolerance=1) is True


def test_line_of_sight_diagonal_flank_policy_either_is_stricter_than_both() -> None:
    """Validates configurable strictness for diagonal flank blocking policy."""
    grid = MapGrid(3, 3)
    grid.add_obstacle(1, 0)

    assert has_line_of_sight(grid, (0, 0), (1, 1), diagonal_flank_policy="both") is True
    assert has_line_of_sight(grid, (0, 0), (1, 1), diagonal_flank_policy="either") is False


def test_line_of_sight_is_direction_invariant_for_shallow_diagonal() -> None:
    """Prevents one-way LOS approval caused by Bresenham tie-breaking."""
    grid = MapGrid(3, 3)
    grid.add_obstacle(1, 0)

    forward_either = has_line_of_sight(
        grid,
        (0, 0),
        (1, 2),
        diagonal_flank_policy="either",
    )
    reverse_either = has_line_of_sight(
        grid,
        (1, 2),
        (0, 0),
        diagonal_flank_policy="either",
    )
    assert forward_either is False
    assert reverse_either is False

    forward_both = has_line_of_sight(
        grid,
        (0, 0),
        (1, 2),
        diagonal_flank_policy="both",
    )
    reverse_both = has_line_of_sight(
        grid,
        (1, 2),
        (0, 0),
        diagonal_flank_policy="both",
    )
    assert forward_both is True
    assert reverse_both is True


@pytest.mark.parametrize(
    "point_pair",
    [
        ((-1, 0), (5, 5)),
        ((0, -1), (5, 5)),
        ((5, 5), (10, 10)),
        ((5, 5), (11, 11)),
    ],
)
def test_line_of_sight_out_of_bounds_returns_false(
    point_pair: tuple[GridPoint, GridPoint],
) -> None:
    """Verifies LOS queries fail fast when endpoints fall outside the map."""
    grid = MapGrid(10, 10)
    start_point, end_point = point_pair
    assert has_line_of_sight(grid, start_point, end_point) is False


def test_line_of_sight_negative_tolerance_raises_value_error() -> None:
    """Ensures invalid negative LOS tolerance raises `ValueError`."""
    grid = MapGrid(10, 10)
    with pytest.raises(ValueError):
        has_line_of_sight(grid, (0, 0), (1, 1), tolerance=-1)


def test_line_of_sight_invalid_diagonal_flank_policy_raises_value_error() -> None:
    """Ensures invalid diagonal flank policy raises `ValueError`."""
    grid = MapGrid(10, 10)
    with pytest.raises(ValueError):
        has_line_of_sight(grid, (0, 0), (1, 1), diagonal_flank_policy="invalid")
