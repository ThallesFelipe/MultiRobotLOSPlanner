"""Regression tests for interactive replanner helper behavior."""

import networkx as nx

import tools.interactive_replanner as interactive_module
from core.map_grid import MapGrid
from presets.map_catalog import create_map_from_catalog
from tools.interactive_replanner import InteractiveReplannerApp


def _make_headless_app() -> InteractiveReplannerApp:
    """Creates an app shell with UI side effects replaced by no-ops."""
    app = InteractiveReplannerApp.__new__(InteractiveReplannerApp)
    app.pending_obstacle_events = []
    app.robot_count_var = type("_RobotCount", (), {"get": lambda self: 4})()
    app._log = lambda *_, **__: None  # type: ignore[method-assign]
    app._set_status = lambda *_, **__: None  # type: ignore[method-assign]
    app._stop_playback = lambda *_, **__: None  # type: ignore[method-assign]
    app._update_step_info = lambda *_, **__: None  # type: ignore[method-assign]
    app._render = lambda *_, **__: None  # type: ignore[method-assign]
    return app


def test_spread_overlapping_positions_preserves_valid_non_overlapping_chain() -> None:
    """Non-overlapping connected seeds must remain unchanged."""
    app = InteractiveReplannerApp.__new__(InteractiveReplannerApp)
    app.map_grid = MapGrid(5, 5)
    app.current_path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    graph: nx.Graph[tuple[int, int]] = nx.Graph()
    graph.add_nodes_from(app.current_path)
    for source_node, target_node in zip(app.current_path, app.current_path[1:]):
        graph.add_edge(source_node, target_node, weight=1.0)

    positions = {0: (0, 2), 1: (0, 3), 2: (0, 4)}

    spread = app._spread_overlapping_positions(positions, graph, (0, 0))

    assert spread == positions


def test_spread_overlapping_positions_does_not_reposition_overlaps() -> None:
    """Compatibility helper must not teleport robots before replanning."""
    app = InteractiveReplannerApp.__new__(InteractiveReplannerApp)
    app.map_grid = MapGrid(5, 5)
    app.current_path = [(0, 0), (0, 1), (0, 2)]

    graph: nx.Graph[tuple[int, int]] = nx.Graph()
    graph.add_nodes_from(app.current_path)

    positions = {0: (0, 1), 1: (0, 1)}

    spread = app._spread_overlapping_positions(positions, graph, (0, 0))

    assert spread == positions


def test_build_planning_graph_uses_current_map_state_with_dynamic_obstacle() -> None:
    """Planning graph must not include LOS edges crossing fresh obstacles."""
    app = InteractiveReplannerApp.__new__(InteractiveReplannerApp)
    app.map_grid = MapGrid(3, 3)
    app.source_point = (0, 0)
    app.target_point = (0, 2)

    app.map_grid.add_obstacle(0, 1)

    planning_graph = app._build_planning_graph_for_current_map()

    assert planning_graph.has_node((0, 0))
    assert planning_graph.has_node((0, 2))
    assert not planning_graph.has_edge((0, 0), (0, 2))


def test_build_planning_graph_connects_bidas_corridors() -> None:
    """The interactive graph must stay connected on corridor-heavy catalog maps."""
    app = InteractiveReplannerApp.__new__(InteractiveReplannerApp)
    app.map_grid = create_map_from_catalog("bidas")
    app.source_point = (52, 7)
    app.target_point = (8, 98)

    planning_graph = app._build_planning_graph_for_current_map()

    assert planning_graph.has_node(app.source_point)
    assert planning_graph.has_node(app.target_point)
    assert nx.has_path(planning_graph, app.source_point, app.target_point)


def test_plan_candidate_uses_configured_robot_count() -> None:
    """The GUI does not size the fleet only from the first path length."""
    app = _make_headless_app()
    app.map_grid = MapGrid(3, 4)
    app.source_point = (0, 0)
    app.target_point = (0, 3)
    app.robot_count_var = type("_RobotCount", (), {"get": lambda self: 2})()

    graph: nx.Graph[tuple[int, int]] = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ((0, 0), (0, 1), 1.0),
            ((0, 1), (0, 2), 1.0),
            ((0, 2), (0, 3), 1.0),
            ((0, 0), (1, 1), 3.0),
            ((1, 1), (0, 3), 3.0),
        ]
    )

    candidate = app._build_plan_candidate(graph, prefer_fewer_relays=False)

    assert candidate is not None
    _, path, snapshots, _ = candidate
    assert path == [(0, 0), (1, 1), (0, 3)]
    assert len(snapshots[0]["positions"]) == 2


def test_replanning_waits_until_leader_reaches_obstacle_edge() -> None:
    """An obstacle farther ahead is stored until it blocks the leader's next edge."""
    app = _make_headless_app()
    app.map_grid = MapGrid(1, 5)
    app.map_grid.add_obstacle(0, 3)
    app.current_path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    app.current_positions = {0: (0, 0), 1: (0, 0)}
    triggered: list[tuple[int, int]] = []
    app._trigger_reactive_replanning = triggered.append  # type: ignore[method-assign]

    triggered_now = app._maybe_trigger_replanning({(0, 3)}, (0, 3))

    assert triggered_now is False
    assert triggered == []
    assert app.pending_obstacle_events == [({(0, 3)}, (0, 3))]


def test_obstacle_on_leader_next_edge_is_deferred_until_attempt() -> None:
    """Manual obstacle insertion arms detection; the leader attempt triggers it."""
    app = _make_headless_app()
    app.map_grid = MapGrid(1, 5)
    app.map_grid.add_obstacle(0, 1)
    app.current_path = [(0, 0), (0, 1), (0, 2)]
    app.current_positions = {0: (0, 0), 1: (0, 0)}
    triggered: list[tuple[int, int]] = []
    app._trigger_reactive_replanning = triggered.append  # type: ignore[method-assign]

    triggered_now = app._maybe_trigger_replanning({(0, 1)}, (0, 1))

    assert triggered_now is False
    assert triggered == []
    assert app.pending_obstacle_events == [({(0, 1)}, (0, 1))]


def test_pending_obstacle_does_not_replan_until_leader_attempt() -> None:
    """Deferred obstacle detection is handled by runtime leader move validation."""
    app = _make_headless_app()
    app.map_grid = MapGrid(1, 5)
    app.map_grid.add_obstacle(0, 2)
    app.current_path = [(0, 0), (0, 1), (0, 2), (0, 3)]
    app.current_positions = {0: (0, 0), 1: (0, 0)}
    app.pending_obstacle_events = [({(0, 2)}, (0, 2))]
    triggered: list[tuple[int, int]] = []
    app._trigger_reactive_replanning = triggered.append  # type: ignore[method-assign]

    app.current_positions = {0: (0, 1), 1: (0, 0)}
    app._check_pending_obstacles_for_leader()

    assert triggered == []
    assert app.pending_obstacle_events == [({(0, 2)}, (0, 2))]


def test_leader_runtime_block_triggers_replanning_before_snapshot_applies() -> None:
    """A stale leader snapshot crossing a new obstacle is never applied."""
    app = _make_headless_app()
    app.map_grid = MapGrid(1, 5)
    app.map_grid.add_obstacle(0, 2)
    app.source_point = (0, 0)
    app.current_path = [(0, 0), (0, 1), (0, 4)]
    app.current_positions = {0: (0, 1), 1: (0, 0)}
    app.current_snapshot_index = 0
    app.snapshots = [
        {
            "step": 0,
            "robot_id": None,
            "from_pos": None,
            "to_pos": None,
            "positions": {0: (0, 1), 1: (0, 0)},
            "valid": True,
            "description": "before leader attempt",
        },
        {
            "step": 1,
            "robot_id": 0,
            "from_pos": (0, 1),
            "to_pos": (0, 4),
            "positions": {0: (0, 4), 1: (0, 0)},
            "valid": True,
            "description": "stale leader jump",
        },
    ]
    triggered: list[tuple[int, int]] = []
    app._trigger_reactive_replanning = triggered.append  # type: ignore[method-assign]

    app._on_next()

    assert triggered == [(0, 2)]
    assert app.current_snapshot_index == 0
    assert app.current_positions == {0: (0, 1), 1: (0, 0)}


def test_failed_replanning_replaces_future_original_snapshots(
    monkeypatch,
) -> None:
    """When replanning is infeasible, old future moves must not remain executable."""
    app = _make_headless_app()
    app.map_grid = MapGrid(3, 4)
    app.source_point = (0, 0)
    app.target_point = (0, 3)
    app.current_path = [(0, 0), (0, 1), (0, 2), (0, 3)]
    app.current_positions = {0: (0, 1), 1: (0, 0)}
    app.current_snapshot_index = 1
    app.snapshots = [
        {
            "step": 0,
            "robot_id": None,
            "from_pos": None,
            "to_pos": None,
            "positions": {0: (0, 0), 1: (0, 0)},
            "valid": True,
            "description": "initial",
        },
        {
            "step": 1,
            "robot_id": 0,
            "from_pos": (0, 0),
            "to_pos": (0, 1),
            "positions": {0: (0, 1), 1: (0, 0)},
            "valid": True,
            "description": "leader moved before obstacle",
        },
        {
            "step": 2,
            "robot_id": 0,
            "from_pos": (0, 1),
            "to_pos": (0, 3),
            "positions": {0: (0, 3), 1: (0, 0)},
            "valid": True,
            "description": "stale future move through blocked corridor",
        },
    ]

    def _fake_reactive_replanning(*_, **__):
        return (
            interactive_module.INFINITE_PATH_COST,
            [],
            [],
            nx.Graph(),
        )

    monkeypatch.setattr(
        interactive_module,
        "reactive_replanning",
        _fake_reactive_replanning,
    )

    app._trigger_reactive_replanning((0, 2))

    assert app.current_snapshot_index == len(app.snapshots) - 1
    assert app.snapshots[-1]["valid"] is False
    assert app.snapshots[-1]["positions"] == {0: (0, 1), 1: (0, 0)}
    assert app._has_terminal_failure() is True
    assert all(
        snapshot["description"] != "stale future move through blocked corridor"
        for snapshot in app.snapshots
    )

    app._on_next()

    assert app.current_positions == {0: (0, 1), 1: (0, 0)}
    assert app.current_snapshot_index == len(app.snapshots) - 1
