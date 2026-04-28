"""Tests for reactive replanning under LOS-constrained relay rules."""

import math
from typing import Any

import networkx as nx

import algorithms.reactive_replanning as reactive_module
from algorithms.reactive_replanning import (
    reactive_replan,
    reactive_replanning,
    update_graph_remove_edge,
    update_graph_with_blocked_segment,
)
from core.map_grid import MapGrid

GridPoint = tuple[int, int]
WeightedGridEdge = tuple[GridPoint, GridPoint, float]
RobotPositions = dict[int, GridPoint]


def _build_graph(edges: list[WeightedGridEdge]) -> nx.Graph[GridPoint]:
    """Creates a weighted visibility-graph fixture for reactive tests."""
    graph: nx.Graph[GridPoint] = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph


def test_update_graph_remove_edge_supports_single_and_multiple_inputs() -> None:
    """Ensures blocked-edge removal accepts one edge or an iterable of edges."""
    graph = _build_graph(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 0), (1, 0), 1.0),
        ]
    )

    single_removed = update_graph_remove_edge(graph, ((0, 0), (0, 1)))
    assert single_removed.has_edge((0, 0), (0, 1)) is False
    assert single_removed.has_edge((0, 1), (0, 2)) is True

    multi_removed = update_graph_remove_edge(
        graph,
        [((0, 0), (0, 1)), ((0, 1), (0, 2))],
    )
    assert multi_removed.has_edge((0, 0), (0, 1)) is False
    assert multi_removed.has_edge((0, 1), (0, 2)) is False
    assert multi_removed.has_edge((0, 0), (1, 0)) is True


def test_update_graph_with_blocked_segment_removes_only_intersections() -> None:
    """Checks geometric blocked-corridor pruning by segment intersection."""
    graph = _build_graph(
        [
            ((0, 1), (2, 1), 2.0),
            ((0, 0), (0, 2), 2.0),
        ]
    )

    updated_graph = update_graph_with_blocked_segment(
        graph,
        blocked_segment=((1, 0), (1, 2)),
    )

    assert updated_graph.has_edge((0, 1), (2, 1)) is False
    assert updated_graph.has_edge((0, 0), (0, 2)) is True


def test_update_graph_with_blocked_segment_uses_signed_direction_filter() -> None:
    """Preserves intersecting edges whose orientation points opposite travel."""
    graph: nx.Graph[GridPoint] = nx.Graph()
    graph.add_edge((2, 1), (0, 1), weight=2.0)
    graph.add_edge((0, 2), (2, 2), weight=2.0)

    updated_graph = update_graph_with_blocked_segment(
        graph,
        blocked_segment=((1, 0), (1, 3)),
        path_direction=(1.0, 0.0),
        direction_dot_threshold=0.0,
    )

    assert updated_graph.has_edge((2, 1), (0, 1)) is True
    assert updated_graph.has_edge((0, 2), (2, 2)) is False


def test_reactive_replan_moves_leader_to_goal() -> None:
    """Validates Algorithm 1 progression along a feasible replanned path."""
    graph = _build_graph(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
        ]
    )
    path_new: list[GridPoint] = [(0, 0), (0, 1), (0, 2)]
    initial_positions: RobotPositions = {0: (0, 0), 1: (0, 0)}

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
    )

    assert snapshots[0]["step"] == 0
    assert snapshots[-1]["positions"] == {0: (0, 2), 1: (0, 1)}
    assert all(
        snapshot["valid"]
        for snapshot in snapshots
        if snapshot["robot_id"] is not None
    )


def test_reactive_replan_rejects_partial_connectivity_recovery() -> None:
    """Does not repair a disconnected leader by ignoring full-chain LOS."""
    graph = _build_graph(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 2), (0, 3), 1.0),
        ]
    )
    path_new: list[GridPoint] = [(0, 0), (0, 1), (0, 2), (0, 3)]
    initial_positions: RobotPositions = {
        0: (0, 3),
        1: (0, 0),
        2: (0, 0),
    }

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
        frozen_robot_ids={0},
    )

    assert snapshots
    assert all(snapshot["positions"][0] == (0, 3) for snapshot in snapshots)
    assert snapshots[-1]["valid"] is False
    assert "Deadlock detected" in snapshots[-1]["description"]
    assert not any(snapshot["valid"] for snapshot in snapshots[1:])


def test_reactive_replan_reconnects_extra_robot_after_target_path_is_reached() -> None:
    """All robots must remain connected to the base, even extra support robots."""
    base: GridPoint = (0, 0)
    intermediate: GridPoint = (0, 1)
    leader: GridPoint = (0, 2)
    isolated_start: GridPoint = (1, 1)
    isolated_bridge: GridPoint = (1, 0)

    graph = _build_graph(
        [
            (base, intermediate, 1.0),
            (intermediate, leader, 1.0),
            (isolated_start, isolated_bridge, 1.0),
            (isolated_bridge, intermediate, 1.0),
        ]
    )
    path_new: list[GridPoint] = [base, intermediate, leader]
    initial_positions: RobotPositions = {
        0: leader,
        1: intermediate,
        2: isolated_start,
    }

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
        frozen_robot_ids={0},
        record_blocked_attempts=False,
    )

    assert snapshots[-1]["positions"] == {
        0: leader,
        1: intermediate,
        2: isolated_bridge,
    }
    assert all(snapshot["valid"] for snapshot in snapshots)


def test_reactive_replan_resolves_saturated_future_by_moving_follower_back() -> None:
    """The adaptive scheduler can clear a saturated future vertex."""
    graph = _build_graph(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
        ]
    )
    path_new: list[GridPoint] = [(0, 0), (0, 1), (0, 2)]
    initial_positions: RobotPositions = {0: (0, 1), 1: (0, 2)}

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
    )

    assert snapshots[-1]["positions"] == {0: (0, 2), 1: (0, 1)}
    assert all(snapshot["valid"] for snapshot in snapshots)


def test_reactive_replan_can_skip_blocked_attempt_snapshots() -> None:
    """A successful adaptive schedule contains only valid moves when requested."""
    graph = _build_graph(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
        ]
    )
    path_new: list[GridPoint] = [(0, 0), (0, 1), (0, 2)]
    initial_positions: RobotPositions = {0: (0, 1), 1: (0, 2)}

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
        record_blocked_attempts=False,
    )

    assert snapshots[-1]["positions"] == {0: (0, 2), 1: (0, 1)}
    assert all(snapshot["valid"] for snapshot in snapshots)


def test_reactive_replan_reports_deadlock_in_place() -> None:
    """Deadlock is reported in place without resetting robots to base."""
    graph = _build_graph([((0, 1), (0, 2), 1.0)])
    path_new: list[GridPoint] = [(0, 0), (0, 1), (0, 2)]
    initial_positions: RobotPositions = {0: (0, 2), 1: (0, 0)}

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
        frozen_robot_ids={0},
    )

    descriptions = [snapshot["description"] for snapshot in snapshots]
    assert not any("fallback reset to base" in description for description in descriptions)
    assert snapshots[-1]["positions"] == initial_positions
    assert snapshots[-1]["valid"] is False


def test_reactive_replanning_runs_full_update_path_and_schedule_pipeline() -> None:
    """Validates graph update + path recomputation + reactive scheduling flow."""
    source: GridPoint = (0, 0)
    target: GridPoint = (0, 2)
    graph = _build_graph(
        [
            (source, (0, 1), 1.0),
            ((0, 1), target, 1.0),
            (source, (1, 1), 1.0),
            ((1, 1), target, 1.0),
        ]
    )

    cost, path_new, snapshots, updated_graph = reactive_replanning(
        graph,
        source=source,
        target=target,
        initial_positions={0: source, 1: source},
        blocked_edge=[(source, (0, 1)), ((0, 1), target)],
        lam=0.0,
    )

    assert updated_graph.has_edge(source, (0, 1)) is False
    assert updated_graph.has_edge((0, 1), target) is False
    assert path_new == [source, (1, 1), target]
    assert math.isclose(cost, 2.0, rel_tol=1e-12, abs_tol=1e-12)
    assert snapshots[-1]["positions"][0] == target


def test_reactive_replanning_uses_obstacle_point_to_infer_blocked_segment(
    monkeypatch: Any,
) -> None:
    """Ensures `obstacle_point` path calls EDT corridor inference hook."""
    source: GridPoint = (0, 0)
    target: GridPoint = (0, 2)
    graph = _build_graph(
        [
            (source, (0, 1), 1.0),
            ((0, 1), target, 1.0),
            (source, (1, 1), 1.0),
            ((1, 1), target, 1.0),
        ]
    )
    grid_obj = MapGrid(4, 4)

    def _fake_segment_inference(_: MapGrid, __: GridPoint) -> tuple[GridPoint, GridPoint]:
        return ((-1, 1), (0, 1))

    monkeypatch.setattr(
        reactive_module,
        "compute_blocked_corridor_segment",
        _fake_segment_inference,
    )

    cost, path_new, snapshots, updated_graph = reactive_replanning(
        graph,
        source=source,
        target=target,
        initial_positions={0: source, 1: source},
        obstacle_point=(0, 1),
        grid_obj=grid_obj,
        lam=0.0,
    )

    assert updated_graph.has_edge(source, (0, 1)) is False
    assert updated_graph.has_edge((0, 1), target) is False
    assert path_new == [source, (1, 1), target]
    assert math.isclose(cost, 2.0, rel_tol=1e-12, abs_tol=1e-12)
    assert snapshots[-1]["positions"][0] == target


def test_reactive_replanning_obstacle_point_requires_grid_obj() -> None:
    """Validates required map input when requesting obstacle-based inference."""
    source: GridPoint = (0, 0)
    target: GridPoint = (0, 1)
    graph = _build_graph([(source, target, 1.0)])

    try:
        reactive_replanning(
            graph,
            source=source,
            target=target,
            initial_positions={0: source},
            obstacle_point=(0, 0),
        )
    except ValueError as exc:
        assert "grid_obj is required" in str(exc)
    else:
        raise AssertionError("reactive_replanning() did not raise ValueError")


def test_reactive_replanning_respects_max_relay_robots_constraint() -> None:
    """Rejects paths that require more relays than currently available."""
    source: GridPoint = (0, 0)
    target: GridPoint = (0, 4)
    graph = _build_graph(
        [
            (source, (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 2), (0, 3), 1.0),
            ((0, 3), target, 1.0),
        ]
    )

    cost, path_new, _, _ = reactive_replanning(
        graph,
        source=source,
        target=target,
        initial_positions={0: source, 1: source, 2: source},
        lam=0.0,
        max_relay_robots=2,
    )

    assert cost == reactive_module.INFINITE_PATH_COST
    assert path_new == []


def test_reactive_replanning_never_adds_additional_relays() -> None:
    """Insufficient available robots makes replanning infeasible."""
    source: GridPoint = (0, 0)
    target: GridPoint = (0, 4)
    graph = _build_graph(
        [
            (source, (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 2), (0, 3), 1.0),
            ((0, 3), target, 1.0),
        ]
    )

    cost, path_new, snapshots, _ = reactive_replanning(
        graph,
        source=source,
        target=target,
        initial_positions={0: source, 1: source},
        lam=0.0,
    )

    assert cost == reactive_module.INFINITE_PATH_COST
    assert path_new == []
    assert snapshots
    assert len(snapshots[0]["positions"]) == 2
    assert snapshots[0]["valid"] is False
    assert "available robot count" in snapshots[0]["description"]


def test_reactive_replan_reports_deadlock_when_extra_robot_cannot_reconnect() -> None:
    """A finished target path is not enough when an extra robot is disconnected."""
    graph = _build_graph(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 0), (1, 0), 1.0),
            ((0, 1), (1, 0), 1.0),
        ]
    )
    path_new: list[GridPoint] = [(0, 0), (2, 2)]
    initial_positions: RobotPositions = {0: (2, 2), 1: (0, 0), 2: (1, 0)}

    snapshots = reactive_replan(
        graph,
        path_new,
        initial_positions,
        frozen_robot_ids={0},
        max_steps=40,
    )

    assert snapshots[-1]["positions"] == initial_positions
    assert snapshots[-1]["valid"] is False
    assert "Deadlock detected" in snapshots[-1]["description"]
