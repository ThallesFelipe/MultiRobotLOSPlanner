"""Tests for shared midpoint projection and LOS connectivity helpers."""

from algorithms.connectivity_checks import to_grid_point


def test_to_grid_point_uses_half_away_from_zero_rounding() -> None:
    """Ensures midpoint projection does not use banker rounding on .5 values."""
    assert to_grid_point((0.5, 0.5)) == (1, 1)
    assert to_grid_point((1.5, 2.5)) == (2, 3)
    assert to_grid_point((-0.5, -1.5)) == (-1, -2)
