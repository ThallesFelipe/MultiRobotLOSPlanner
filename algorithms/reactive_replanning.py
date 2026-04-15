"""Reactive replanning for LOS-constrained multi-robot relay deployment.

This module implements the paper's adaptive strategy (Algorithm 1 and
Algorithm 2, Section III-D): when an obstacle blocks the current corridor, the
planner updates the visibility graph, computes a new relay path, and generates
movement snapshots that preserve LOS connectivity constraints.
"""

from collections.abc import Iterable, Sequence
from typing import TypeGuard

import networkx as nx
import numpy as np

from algorithms.connectivity_checks import (
    bfs_connected,
    midpoint,
    midpoint_has_los_to_chain,
    temporary_los_connectivity_check,
)
from algorithms.ordered_progression import (
    MovementSnapshot,
    ordered_progression,
)
from algorithms.relay_dijkstra import (
    DEFAULT_RELAY_PENALTY_LAMBDA,
    INFINITE_PATH_COST,
    relay_dijkstra,
)
from core.map_grid import GridPoint, MapGrid
from core.map_processor import compute_blocked_corridor_segment

Edge = tuple[GridPoint, GridPoint]
RobotPositions = dict[int, GridPoint]
PathDirection = tuple[float, float]

LEADER_ROBOT_ID: int = 0
DEFAULT_MAX_REACTIVE_STEPS: int = 500
DEFAULT_DIRECTION_DOT_THRESHOLD: float = 0.0


def _is_grid_point(value: object) -> TypeGuard[GridPoint]:
    """Checks whether an object has the `(row, col)` integer-grid shape."""
    match value:
        case (int(), int()):
            return True
        case _:
            return False


def _is_edge(value: object) -> TypeGuard[Edge]:
    """Checks whether an object has the `((r1, c1), (r2, c2))` edge shape."""
    match value:
        case (object() as source_vertex, object() as target_vertex):
            return _is_grid_point(source_vertex) and _is_grid_point(target_vertex)
        case _:
            return False


def _normalize_blocked_edges(
    blocked_edge: Edge | Iterable[Edge],
) -> list[Edge]:
    """Normalizes a blocked-edge input into a concrete edge list."""
    if _is_edge(blocked_edge):
        return [blocked_edge]

    edges: list[Edge] = []
    for blocked_edge_candidate in blocked_edge:
        if not _is_edge(blocked_edge_candidate):
            raise ValueError(
                "Each blocked edge must be a pair of grid points "
                "`((row1, col1), (row2, col2))`."
            )
        edges.append(blocked_edge_candidate)
    return edges


def _euclidean_distance(p1: GridPoint, p2: GridPoint) -> float:
    """Computes Euclidean distance between two occupancy-grid points."""
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _segment_orientation(p: GridPoint, q: GridPoint, r: GridPoint) -> int:
    """Returns orientation sign for the ordered triplet `(p, q, r)`.

    Returns:
        `0` if collinear, `1` if clockwise, `2` if counter-clockwise.
    """
    cross_value = ((q[1] - p[1]) * (r[0] - q[0])) - (
        (q[0] - p[0]) * (r[1] - q[1])
    )
    if cross_value == 0:
        return 0
    return 1 if cross_value > 0 else 2


def _point_on_segment(p: GridPoint, q: GridPoint, r: GridPoint) -> bool:
    """Checks whether point `q` lies on segment `[p, r]` (inclusive)."""
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )


def _segments_intersect(
    p1: GridPoint,
    q1: GridPoint,
    p2: GridPoint,
    q2: GridPoint,
) -> bool:
    """Checks whether two 2D segments intersect (including endpoints)."""
    o1 = _segment_orientation(p1, q1, p2)
    o2 = _segment_orientation(p1, q1, q2)
    o3 = _segment_orientation(p2, q2, p1)
    o4 = _segment_orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _point_on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _point_on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _point_on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _point_on_segment(p2, q1, q2):
        return True

    return False


def _normalize_direction_vector(
    direction: PathDirection,
) -> PathDirection | None:
    """Normalizes a direction vector and handles zero-length vectors safely."""
    norm = float(np.hypot(direction[0], direction[1]))
    if norm == 0.0:
        return None
    return (direction[0] / norm, direction[1] / norm)


def update_graph_remove_edge(
    graph: nx.Graph[GridPoint],
    blocked_edge: Edge | Iterable[Edge],
) -> nx.Graph[GridPoint]:
    """Returns a copy of `graph` with blocked edges removed.

    Args:
        graph: Current visibility graph.
        blocked_edge: A single blocked edge or an iterable of blocked edges.

    Returns:
        Updated graph copy without blocked edges that were present.
    """
    updated_graph = graph.copy()
    for source_vertex, target_vertex in _normalize_blocked_edges(blocked_edge):
        if updated_graph.has_edge(source_vertex, target_vertex):
            updated_graph.remove_edge(source_vertex, target_vertex)
    return updated_graph


def update_graph_with_blocked_segment(
    graph: nx.Graph[GridPoint],
    blocked_segment: Edge,
    path_direction: PathDirection | None = None,
    direction_dot_threshold: float = DEFAULT_DIRECTION_DOT_THRESHOLD,
) -> nx.Graph[GridPoint]:
    """Removes edges that intersect a blocked corridor segment.

    When `path_direction` is provided, edge removals are filtered by directional
    alignment with the travel direction using `|dot| > threshold` over normalized
    vectors. The absolute value is used because visibility edges are undirected.

    Args:
        graph: Current visibility graph.
        blocked_segment: Corridor segment inferred as blocked by obstacle update.
        path_direction: Path-travel direction vector `(dr, dc)`.
        direction_dot_threshold: Minimum directional alignment for removal.

    Returns:
        Updated graph copy with intersecting blocked-corridor edges removed.

    Raises:
        ValueError: If `direction_dot_threshold` is outside `[-1, 1]`.
    """
    if direction_dot_threshold < -1.0 or direction_dot_threshold > 1.0:
        raise ValueError(
            "direction_dot_threshold must be in [-1, 1]; "
            f"received {direction_dot_threshold}."
        )

    normalized_path_direction = (
        _normalize_direction_vector(path_direction)
        if path_direction is not None
        else None
    )

    updated_graph = graph.copy()
    blocked_start, blocked_end = blocked_segment

    edges_to_remove: list[Edge] = []
    for source_vertex, target_vertex in updated_graph.edges:
        if not _segments_intersect(
            source_vertex,
            target_vertex,
            blocked_start,
            blocked_end,
        ):
            continue

        if normalized_path_direction is not None:
            edge_direction = (
                float(target_vertex[0] - source_vertex[0]),
                float(target_vertex[1] - source_vertex[1]),
            )
            normalized_edge_direction = _normalize_direction_vector(edge_direction)
            if normalized_edge_direction is None:
                continue

            direction_alignment = abs(
                (normalized_path_direction[0] * normalized_edge_direction[0])
                + (normalized_path_direction[1] * normalized_edge_direction[1])
            )
            if direction_alignment <= direction_dot_threshold:
                continue

        edges_to_remove.append((source_vertex, target_vertex))

    updated_graph.remove_edges_from(edges_to_remove)
    return updated_graph


def _next_path_vertex(
    current_position: GridPoint,
    path_new: Sequence[GridPoint],
) -> GridPoint | None:
    """Returns the immediate successor of `current_position` in `path_new`."""
    if current_position not in path_new:
        return None

    current_index = path_new.index(current_position)
    if current_index + 1 >= len(path_new):
        return None
    return path_new[current_index + 1]


def _max_path_progress_reachable(
    graph: nx.Graph[GridPoint],
    candidate_position: GridPoint,
    path_new: Sequence[GridPoint],
) -> int:
    """Returns the farthest index in `path_new` reachable from candidate node."""
    if candidate_position not in graph:
        return -1

    for index in range(len(path_new) - 1, -1, -1):
        path_vertex = path_new[index]
        if path_vertex not in graph:
            continue
        if nx.has_path(graph, candidate_position, path_vertex):
            return index
    return -1


def _deadlock_avoidance_violation(
    current_position: GridPoint,
    candidate_position: GridPoint,
    positions: RobotPositions,
    path_new: Sequence[GridPoint],
) -> bool:
    """Checks the deadlock-prevention rule from Algorithm 2."""
    if current_position not in path_new or candidate_position not in path_new:
        return False

    current_index = path_new.index(current_position)
    if current_index + 1 >= len(path_new):
        return False
    if candidate_position != path_new[current_index + 1]:
        return False

    forward_vertices = path_new[current_index + 1 :]
    occupied_vertices = set(positions.values())
    return all(vertex in occupied_vertices for vertex in forward_vertices)


def _validate_move(
    robot_id: int,
    current_position: GridPoint,
    candidate_position: GridPoint,
    positions: RobotPositions,
    path_new: Sequence[GridPoint],
    leader_id: int,
    grid_obj: MapGrid | None,
    connectivity_graph: nx.Graph[GridPoint],
    base: GridPoint,
) -> tuple[bool, str]:
    """Validates a movement proposal under Algorithm 2 constraints."""
    if candidate_position == current_position:
        return False, "candidate is identical to current position"

    if (
        current_position not in connectivity_graph
        or candidate_position not in connectivity_graph
        or not connectivity_graph.has_edge(current_position, candidate_position)
    ):
        return False, "candidate is not an adjacent visibility-graph neighbor"

    if candidate_position in path_new:
        occupied_vertices = set(positions.values())
        candidate_index = path_new.index(candidate_position)
        for path_index in range(1, candidate_index):
            if path_new[path_index] not in occupied_vertices:
                return False, "sequential formation violated"

        if current_position in path_new:
            current_index = path_new.index(current_position)
            if candidate_index > current_index + 1:
                return False, "sequential formation violated"

    if robot_id != leader_id and positions[leader_id] in path_new:
        leader_current_index = path_new.index(positions[leader_id])
        if leader_current_index + 1 < len(path_new):
            leader_next_vertex = path_new[leader_current_index + 1]
            if candidate_position == leader_next_vertex:
                return False, "leader-priority rule violated"

    if _deadlock_avoidance_violation(
        current_position,
        candidate_position,
        positions,
        path_new,
    ):
        return False, "deadlock-avoidance rule violated"

    if grid_obj is not None:
        static_positions = [
            positions[other_robot_id]
            for other_robot_id in positions
            if other_robot_id != robot_id
        ]
        if not midpoint_has_los_to_chain(
            grid_obj,
            current_position,
            candidate_position,
            static_positions,
        ):
            return False, "midpoint LOS to static chain failed"

        movement_midpoint = midpoint(current_position, candidate_position)
        simulated_midpoint_positions = [
            movement_midpoint
            if other_robot_id == robot_id
            else positions[other_robot_id]
            for other_robot_id in sorted(positions)
        ]
        if not temporary_los_connectivity_check(
            grid_obj,
            simulated_midpoint_positions,
            base,
        ):
            return False, "temporary midpoint connectivity to base failed"

    simulated_positions = dict(positions)
    simulated_positions[robot_id] = candidate_position
    if not bfs_connected(
        list(simulated_positions.values()),
        base,
        connectivity_graph,
    ):
        return False, "visibility-graph BFS connectivity failed"

    return True, "ok"


def _acquisition_heuristic(
    robot_id: int,
    current_position: GridPoint,
    path_new: Sequence[GridPoint],
    positions: RobotPositions,
    leader_id: int,
    grid_obj: MapGrid | None,
    connectivity_graph: nx.Graph[GridPoint],
    base: GridPoint,
) -> GridPoint | None:
    """Selects the best acquisition move for robots outside `path_new`.

    Candidate neighbors are ranked by:
    1. Maximum reachable progress along `path_new`.
    2. Minimum single-step travel distance.
    3. Minimum distance to the goal vertex.
    """
    if current_position not in connectivity_graph:
        return None

    goal = path_new[-1]
    best_candidate: GridPoint | None = None
    best_score = (-1, float("-inf"), float("-inf"))

    for candidate_position in connectivity_graph.neighbors(current_position):
        is_valid, _ = _validate_move(
            robot_id,
            current_position,
            candidate_position,
            positions,
            path_new,
            leader_id,
            grid_obj,
            connectivity_graph,
            base,
        )
        if not is_valid:
            continue

        path_progress = _max_path_progress_reachable(
            connectivity_graph,
            candidate_position,
            path_new,
        )
        step_distance = _euclidean_distance(current_position, candidate_position)
        goal_distance = _euclidean_distance(candidate_position, goal)
        score = (path_progress, -step_distance, -goal_distance)

        if (
            best_candidate is None
            or score > best_score
            or (score == best_score and candidate_position < best_candidate)
        ):
            best_score = score
            best_candidate = candidate_position

    return best_candidate


def _append_snapshot(
    snapshots: list[MovementSnapshot],
    step: int,
    robot_id: int | None,
    from_pos: GridPoint | None,
    to_pos: GridPoint | None,
    positions: RobotPositions,
    valid: bool,
    description: str,
) -> None:
    """Appends one movement snapshot in the project-standard schema."""
    snapshots.append(
        {
            "step": step,
            "robot_id": robot_id,
            "from_pos": from_pos,
            "to_pos": to_pos,
            "positions": dict(positions),
            "valid": valid,
            "description": description,
        }
    )


def _append_deadlock_fallback(
    snapshots: list[MovementSnapshot],
    positions: RobotPositions,
    path_new: Sequence[GridPoint],
    grid_obj: MapGrid | None,
    connectivity_graph: nx.Graph[GridPoint],
    current_step: int,
) -> int:
    """Applies the paper's fallback strategy after deadlock detection."""
    base = path_new[0]
    reset_positions = {robot_id: base for robot_id in positions}

    next_step = current_step + 1
    _append_snapshot(
        snapshots,
        next_step,
        None,
        None,
        None,
        reset_positions,
        True,
        "Deadlock detected: fallback reset to base and deterministic redeployment.",
    )

    deterministic_snapshots = ordered_progression(
        path_new,
        grid_obj=grid_obj,
        vis_graph=connectivity_graph,
    )

    for deterministic_snapshot in deterministic_snapshots[1:]:
        next_step += 1
        snapshots.append(
            {
                "step": next_step,
                "robot_id": deterministic_snapshot["robot_id"],
                "from_pos": deterministic_snapshot["from_pos"],
                "to_pos": deterministic_snapshot["to_pos"],
                "positions": deterministic_snapshot["positions"],
                "valid": deterministic_snapshot["valid"],
                "description": (
                    "Fallback deterministic: "
                    f"{deterministic_snapshot['description']}"
                ),
            }
        )

    positions.clear()
    positions.update(reset_positions)
    return next_step


def reactive_replan(
    updated_graph: nx.Graph[GridPoint],
    path_new: Sequence[GridPoint],
    initial_positions: RobotPositions,
    grid_obj: MapGrid | None = None,
    max_steps: int = DEFAULT_MAX_REACTIVE_STEPS,
    fallback_on_deadlock: bool = True,
) -> list[MovementSnapshot]:
    """Runs Algorithm 1 and returns reactive movement snapshots.

    Args:
        updated_graph: Visibility graph after obstacle-induced updates.
        path_new: Replanned path `[base, ..., goal]`.
        initial_positions: Current robot positions by robot id.
        grid_obj: Occupancy grid used by midpoint LOS checks.
        max_steps: Safety bound for attempted movement snapshots.
        fallback_on_deadlock: Enables deterministic fallback after deadlock.

    Returns:
        Sequence of movement snapshots in the same schema used by
        `ordered_progression`.

    Raises:
        ValueError: If input arguments violate planner assumptions.
    """
    if len(path_new) < 2:
        return []

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be greater than 0; "
            f"received max_steps={max_steps}."
        )

    positions: RobotPositions = dict(initial_positions)
    if not positions:
        return []

    n_robots = len(positions)
    expected_ids = set(range(n_robots))
    if set(positions) != expected_ids:
        raise ValueError(
            "initial_positions must use contiguous robot ids in [0, n_robots)."
        )
    if LEADER_ROBOT_ID not in positions:
        raise ValueError("Leader robot id 0 must be present in initial_positions.")

    base = path_new[0]
    goal = path_new[-1]

    snapshots: list[MovementSnapshot] = []
    _append_snapshot(
        snapshots,
        0,
        None,
        None,
        None,
        positions,
        True,
        "Reactive replanning initial state after obstacle update.",
    )

    step = 0
    while positions[LEADER_ROBOT_ID] != goal and step < max_steps:
        moved_this_round = False

        for robot_id in range(n_robots):
            current_position = positions[robot_id]

            if current_position in path_new:
                candidate_position = _next_path_vertex(current_position, path_new)
            else:
                candidate_position = _acquisition_heuristic(
                    robot_id,
                    current_position,
                    path_new,
                    positions,
                    LEADER_ROBOT_ID,
                    grid_obj,
                    updated_graph,
                    base,
                )

            if candidate_position is None:
                continue

            is_valid, reason = _validate_move(
                robot_id,
                current_position,
                candidate_position,
                positions,
                path_new,
                LEADER_ROBOT_ID,
                grid_obj,
                updated_graph,
                base,
            )

            step += 1
            if is_valid:
                positions[robot_id] = candidate_position
                moved_this_round = True
                description = (
                    f"Step {step}: r{robot_id + 1} moves "
                    f"{current_position}->{candidate_position}"
                )
                snapshot_to_position = candidate_position
            else:
                description = (
                    f"Step {step}: r{robot_id + 1} blocked "
                    f"({reason}) at {current_position}"
                )
                snapshot_to_position = current_position

            _append_snapshot(
                snapshots,
                step,
                robot_id,
                current_position,
                snapshot_to_position,
                positions,
                is_valid,
                description,
            )

            if positions[LEADER_ROBOT_ID] == goal or step >= max_steps:
                break

        if positions[LEADER_ROBOT_ID] == goal or step >= max_steps:
            break

        if not moved_this_round:
            if fallback_on_deadlock:
                step = _append_deadlock_fallback(
                    snapshots,
                    positions,
                    path_new,
                    grid_obj,
                    updated_graph,
                    step,
                )
            else:
                step += 1
                _append_snapshot(
                    snapshots,
                    step,
                    None,
                    None,
                    None,
                    positions,
                    False,
                    "Deadlock detected: no robot could perform a valid move.",
                )
            break

    return snapshots


def reactive_replanning(
    visibility_graph: nx.Graph[GridPoint],
    source: GridPoint,
    target: GridPoint,
    initial_positions: RobotPositions,
    blocked_edge: Edge | Iterable[Edge] | None = None,
    blocked_segment: Edge | None = None,
    obstacle_point: GridPoint | None = None,
    grid_obj: MapGrid | None = None,
    lam: float = DEFAULT_RELAY_PENALTY_LAMBDA,
    path_direction: PathDirection | None = None,
    direction_dot_threshold: float = DEFAULT_DIRECTION_DOT_THRESHOLD,
    max_steps: int = DEFAULT_MAX_REACTIVE_STEPS,
    fallback_on_deadlock: bool = True,
) -> tuple[float, list[GridPoint], list[MovementSnapshot], nx.Graph[GridPoint]]:
    """Runs full reactive replanning: graph update, path search, move generation.

    Args:
        visibility_graph: Baseline visibility graph before obstacle update.
        source: Start path vertex (base).
        target: Goal path vertex.
        initial_positions: Current robot positions by robot id.
        blocked_edge: Optional blocked edge(s) to remove from the graph.
        blocked_segment: Optional blocked corridor segment for geometric pruning.
        obstacle_point: Dynamic obstacle coordinate `(row, col)` used to infer
            a blocked corridor via EDT. Ignored when `blocked_segment` is
            provided explicitly.
        grid_obj: Occupancy grid used for midpoint LOS checks.
        lam: Relay-penalty objective factor used by `relay_dijkstra`.
        path_direction: Path-travel direction `(dr, dc)` for directional pruning.
        direction_dot_threshold: Dot threshold for directional blocked-corridor
            pruning.
        max_steps: Maximum number of attempted reactive movement snapshots.
        fallback_on_deadlock: Enables deterministic fallback after deadlock.

    Returns:
        `(cost, path_new, snapshots, updated_graph)` where `cost` and `path_new`
        come from the updated-graph shortest path and `snapshots` contains the
        resulting movement schedule under Algorithm 1.

    Raises:
        ValueError: If `obstacle_point` is provided without `grid_obj`.
    """
    updated_graph = visibility_graph.copy()

    if blocked_edge is not None:
        updated_graph = update_graph_remove_edge(updated_graph, blocked_edge)

    if blocked_segment is None and obstacle_point is not None:
        if grid_obj is None:
            raise ValueError(
                "grid_obj is required when obstacle_point is provided."
            )
        blocked_segment = compute_blocked_corridor_segment(
            grid_obj,
            obstacle_point,
        )

    if blocked_segment is not None:
        direction = path_direction
        if direction is None:
            direction = (
                float(target[0] - source[0]),
                float(target[1] - source[1]),
            )
        updated_graph = update_graph_with_blocked_segment(
            updated_graph,
            blocked_segment,
            path_direction=direction,
            direction_dot_threshold=direction_dot_threshold,
        )

    path_cost, path_new = relay_dijkstra(updated_graph, source, target, lam=lam)
    if path_cost == INFINITE_PATH_COST:
        positions = dict(initial_positions)
        unreachable_snapshots: list[MovementSnapshot] = [
            {
                "step": 0,
                "robot_id": None,
                "from_pos": None,
                "to_pos": None,
                "positions": positions,
                "valid": False,
                "description": (
                    "Reactive replanning failed: no feasible path after graph "
                    "update."
                ),
            }
        ]
        return path_cost, [], unreachable_snapshots, updated_graph

    snapshots = reactive_replan(
        updated_graph,
        path_new,
        initial_positions,
        grid_obj=grid_obj,
        max_steps=max_steps,
        fallback_on_deadlock=fallback_on_deadlock,
    )
    return path_cost, path_new, snapshots, updated_graph
