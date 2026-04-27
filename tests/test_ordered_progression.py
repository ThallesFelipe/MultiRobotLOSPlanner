"""Tests for deterministic ordered progression under LOS constraints."""

import math

import networkx as nx
import pytest

from algorithms.ordered_progression import (
    count_path_traversed_cells,
    deterministic_sequence,
    ordered_progression,
    plan_ordered_progression_on_visibility_graph,
    total_moves_formula,
)
from core.map_grid import MapGrid

GridPoint = tuple[int, int]


@pytest.mark.parametrize(
    ("n_nodes", "expected"),
    [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 3),
        (4, 6),
        (5, 10),
    ],
)
def test_total_moves_formula(n_nodes: int, expected: int) -> None:
    """Validates closed-form movement count for ordered progression."""
    assert total_moves_formula(n_nodes) == expected


@pytest.mark.parametrize("n_nodes", [2, 3, 4, 5, 6])
def test_deterministic_sequence_length_matches_formula(n_nodes: int) -> None:
    """Ensures schedule size matches `M = (n - 1) * n / 2`."""
    sequence = deterministic_sequence(n_nodes)
    assert len(sequence) == total_moves_formula(n_nodes)


@pytest.mark.parametrize(
    ("n_nodes", "expected_sequence"),
    [
        (1, []),
        (2, [0]),
        (3, [0, 1, 0]),
        (4, [0, 1, 0, 2, 1, 0]),
    ],
)
def test_deterministic_sequence_expected_patterns(
    n_nodes: int,
    expected_sequence: list[int],
) -> None:
    """Checks canonical small-size robot scheduling patterns."""
    assert deterministic_sequence(n_nodes) == expected_sequence


@pytest.mark.parametrize(
    "path",
    [[], [(0, 0)]],
)
def test_ordered_progression_short_paths_return_empty(
    path: list[GridPoint],
) -> None:
    """Verifies that invalidly short paths produce no snapshots."""
    assert ordered_progression(path) == []


def test_ordered_progression_snapshot_count_matches_formula() -> None:
    """Confirms total snapshots equal initial state plus scheduled moves."""
    path: list[GridPoint] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    snapshots = ordered_progression(path)

    assert len(snapshots) == total_moves_formula(len(path)) + 1


def test_ordered_progression_reaches_expected_final_positions() -> None:
    """Checks final robot deployment order along the selected path."""
    path: list[GridPoint] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    snapshots = ordered_progression(path)
    final_positions = snapshots[-1]["positions"]

    assert all(snapshot["valid"] for snapshot in snapshots)
    assert final_positions == {
        0: (0, 3),
        1: (0, 2),
        2: (0, 1),
    }


def test_ordered_progression_keeps_extra_available_robots_at_base() -> None:
    """Available robots beyond the initial path requirement remain in the state."""
    path: list[GridPoint] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    snapshots = ordered_progression(path, robot_count=4)

    assert snapshots[0]["positions"] == {
        0: (0, 0),
        1: (0, 0),
        2: (0, 0),
        3: (0, 0),
    }
    assert snapshots[-1]["positions"] == {
        0: (0, 3),
        1: (0, 2),
        2: (0, 1),
        3: (0, 0),
    }


def test_ordered_progression_rejects_insufficient_robot_count() -> None:
    """A path with n nodes requires at least n - 1 robots."""
    path: list[GridPoint] = [(0, 0), (0, 1), (0, 2)]

    with pytest.raises(ValueError, match="insufficient"):
        ordered_progression(path, robot_count=1)


def test_ordered_progression_initial_snapshot_metadata() -> None:
    """Ensures initial snapshot has stable structure and semantics."""
    path: list[GridPoint] = [(1, 1), (1, 2)]
    snapshots = ordered_progression(path)
    initial = snapshots[0]

    assert initial["step"] == 0
    assert initial["robot_id"] is None
    assert initial["from_pos"] is None
    assert initial["to_pos"] is None
    assert initial["valid"] is True
    assert initial["positions"] == {0: (1, 1)}


def test_ordered_progression_blocks_when_midpoint_los_fails() -> None:
    """Validates blocking when midpoint LOS to static chain is broken."""
    grid = MapGrid(4, 4)
    grid.add_obstacle(0, 1)

    path: list[GridPoint] = [(0, 0), (0, 2), (0, 3)]
    snapshots = ordered_progression(path, grid_obj=grid)
    first_move = snapshots[1]

    assert first_move["robot_id"] == 0
    assert first_move["valid"] is False
    assert first_move["from_pos"] == (0, 0)
    assert first_move["to_pos"] == (0, 0)
    assert "midpoint LOS to static chain failed" in first_move["description"]


def test_ordered_progression_blocks_when_visibility_graph_disconnects() -> None:
    """Ensures disconnected visibility graph blocks movement approvals."""
    path: list[GridPoint] = [(0, 0), (0, 1), (0, 2)]

    disconnected_graph: nx.Graph[GridPoint] = nx.Graph()
    disconnected_graph.add_nodes_from(path)

    snapshots = ordered_progression(path, vis_graph=disconnected_graph)
    first_move = snapshots[1]

    assert first_move["robot_id"] == 0
    assert first_move["valid"] is False
    assert first_move["positions"] == {0: (0, 0), 1: (0, 0)}
    assert "visibility-graph BFS connectivity failed" in first_move["description"]
    assert all(position == path[0] for position in snapshots[-1]["positions"].values())


@pytest.mark.parametrize(
    ("path", "expected_cell_count"),
    [
        ([], 0),
        ([(0, 0)], 1),
        ([(0, 0), (0, 3)], 4),
        ([(0, 0), (0, 2), (2, 2)], 5),
    ],
)
def test_count_path_traversed_cells(path: list[GridPoint], expected_cell_count: int) -> None:
    """Counts rasterized occupancy-grid cells traversed by a path."""
    assert count_path_traversed_cells(path) == expected_cell_count


def test_plan_ordered_progression_on_visibility_graph_returns_path_and_snapshots() -> None:
    """Ensures graph planning returns stable metrics and ordered snapshots."""
    vis_graph: nx.Graph[GridPoint] = nx.Graph()
    vis_graph.add_weighted_edges_from(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 0), (0, 2), 5.0),
        ]
    )

    result = plan_ordered_progression_on_visibility_graph(
        vis_graph,
        source=(0, 0),
        target=(0, 2),
        lam=0.0,
    )

    assert result["path"] == [(0, 0), (0, 1), (0, 2)]
    assert math.isclose(result["path_cost"], 2.0, rel_tol=1e-12, abs_tol=1e-12)
    assert result["traversed_cells"] == 3
    assert result["robots_used"] == 2
    assert len(result["movement_snapshots"]) == total_moves_formula(3) + 1
    assert all(snapshot["valid"] for snapshot in result["movement_snapshots"])


def test_plan_ordered_progression_can_prioritize_fewer_relays() -> None:
    """Ensures planner exposes relay-priority mode through ordered progression API."""
    vis_graph: nx.Graph[GridPoint] = nx.Graph()
    vis_graph.add_weighted_edges_from(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 0), (0, 2), 4.0),
        ]
    )

    result = plan_ordered_progression_on_visibility_graph(
        vis_graph,
        source=(0, 0),
        target=(0, 2),
        lam=1.0,
        prefer_fewer_relays=True,
    )

    assert result["path"] == [(0, 0), (0, 2)]
    assert math.isclose(result["path_cost"], 5.0, rel_tol=1e-12, abs_tol=1e-12)
    assert result["traversed_cells"] == 3
    assert result["robots_used"] == 1
    assert len(result["movement_snapshots"]) == total_moves_formula(2) + 1
    assert all(snapshot["valid"] for snapshot in result["movement_snapshots"])


def test_plan_ordered_progression_on_visibility_graph_handles_missing_path() -> None:
    """Returns empty plan payload when no path exists in visibility graph."""
    vis_graph: nx.Graph[GridPoint] = nx.Graph()
    vis_graph.add_nodes_from([(0, 0), (4, 4)])

    result = plan_ordered_progression_on_visibility_graph(
        vis_graph,
        source=(0, 0),
        target=(4, 4),
    )

    assert math.isinf(result["path_cost"])
    assert result["path"] == []
    assert result["movement_snapshots"] == []
    assert result["traversed_cells"] == 0
    assert result["robots_used"] == 0


def test_plan_ordered_progression_respects_available_robot_count() -> None:
    """Initial planning searches for a path feasible with the configured fleet."""
    vis_graph: nx.Graph[GridPoint] = nx.Graph()
    vis_graph.add_weighted_edges_from(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 2), (0, 3), 1.0),
            ((0, 0), (1, 1), 3.0),
            ((1, 1), (0, 3), 3.0),
        ]
    )

    result = plan_ordered_progression_on_visibility_graph(
        vis_graph,
        source=(0, 0),
        target=(0, 3),
        lam=0.0,
        robot_count=2,
    )

    assert result["path"] == [(0, 0), (1, 1), (0, 3)]
    assert len(result["movement_snapshots"][0]["positions"]) == 2
