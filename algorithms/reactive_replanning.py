"""Reactive replanning for LOS-constrained multi-robot relay deployment.

This module implements the paper's adaptive strategy (Algorithm 1 and
Algorithm 2, Section III-D): when an obstacle blocks the current corridor, the
planner updates the visibility graph, computes a new relay path, and generates
movement snapshots that preserve LOS connectivity constraints.
"""

from collections import deque
from collections.abc import Iterable, Sequence
from typing import TypeGuard

import networkx as nx
import numpy as np

from algorithms.connectivity_checks import (
    midpoint,
    midpoint_has_los_to_chain,
    temporary_los_connectivity_check,
)
from algorithms.ordered_progression import (
    MovementSnapshot,
)
from algorithms.relay_dijkstra import (
    DEFAULT_RELAY_PENALTY_LAMBDA,
    INFINITE_PATH_COST,
    relay_dijkstra_with_edge_cap,
)
from core.map_grid import GridPoint, MapGrid
from core.map_processor import compute_blocked_corridor_segment
from core.visibility import has_line_of_sight

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
    alignment with the travel direction using `dot > threshold` over normalized
    vectors. This intentionally avoids removing edges whose stored orientation
    points opposite the leader's travel direction, preserving retreat edges when
    the caller provides directionally meaningful edge orientation.

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

            direction_alignment = (
                (normalized_path_direction[0] * normalized_edge_direction[0])
                + (normalized_path_direction[1] * normalized_edge_direction[1])
            )
            if direction_alignment <= direction_dot_threshold:
                continue

        edges_to_remove.append((source_vertex, target_vertex))

    updated_graph.remove_edges_from(edges_to_remove)
    return updated_graph


def _add_required_vertices_with_current_los(
    graph: nx.Graph[GridPoint],
    vertices: Iterable[GridPoint],
    grid_obj: MapGrid | None,
) -> nx.Graph[GridPoint]:
    """Adds required vertices and current-map LOS edges when a map is available."""
    updated_graph = graph.copy()
    for vertex in vertices:
        if vertex not in updated_graph:
            updated_graph.add_node(vertex)

    if grid_obj is None:
        return updated_graph

    for vertex in vertices:
        if not grid_obj.in_bounds(*vertex):
            continue
        for candidate in list(updated_graph.nodes):
            if candidate == vertex:
                continue
            if not grid_obj.in_bounds(*candidate):
                continue
            if not has_line_of_sight(grid_obj, vertex, candidate):
                continue
            if not updated_graph.has_edge(vertex, candidate):
                updated_graph.add_edge(
                    vertex,
                    candidate,
                    weight=_euclidean_distance(vertex, candidate),
                )

    return updated_graph


def _remove_edges_without_current_los(
    graph: nx.Graph[GridPoint],
    grid_obj: MapGrid | None,
) -> nx.Graph[GridPoint]:
    """Drops visibility edges that no longer have LOS on the current map."""
    if grid_obj is None:
        return graph.copy()

    updated_graph = graph.copy()
    invalid_edges: list[Edge] = []
    for source_vertex, target_vertex in updated_graph.edges:
        if not has_line_of_sight(grid_obj, source_vertex, target_vertex):
            invalid_edges.append((source_vertex, target_vertex))

    updated_graph.remove_edges_from(invalid_edges)
    return updated_graph


def _infeasible_snapshot(
    positions: RobotPositions,
    description: str,
) -> list[MovementSnapshot]:
    """Builds a standard one-snapshot infeasibility result."""
    return [
        {
            "step": 0,
            "robot_id": None,
            "from_pos": None,
            "to_pos": None,
            "positions": dict(positions),
            "valid": False,
            "description": description,
        }
    ]


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


def _first_unoccupied_path_vertex(
    path_new: Sequence[GridPoint],
    occupied_vertices: set[GridPoint],
) -> GridPoint:
    """Returns the next path vertex that still needs a robot assignment."""
    for path_vertex in path_new[1:]:
        if path_vertex not in occupied_vertices:
            return path_vertex
    return path_new[-1]


def _shortest_hops(
    graph: nx.Graph[GridPoint],
    source: GridPoint,
    target: GridPoint,
) -> int:
    """Returns unweighted hop distance, or a large value when unreachable."""
    if source == target:
        return 0

    if source not in graph or target not in graph:
        return 10**9

    visited: set[GridPoint] = {source}
    queue: deque[tuple[GridPoint, int]] = deque([(source, 0)])

    while queue:
        current_node, depth = queue.popleft()
        for neighbor in graph.neighbors(current_node):
            if neighbor in visited:
                continue
            if neighbor == target:
                return depth + 1
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

    return 10**9


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


def _robots_connected_to_base(
    positions: RobotPositions,
    required_robot_ids: set[int],
    base: GridPoint,
    connectivity_graph: nx.Graph[GridPoint],
) -> bool:
    """Checks whether a robot subset is connected to base over occupied nodes."""
    if not required_robot_ids:
        return True
    if base not in connectivity_graph:
        return False

    allowed_nodes = set(positions.values()) | {base}
    reachable_nodes: set[GridPoint] = {base}
    queue: deque[GridPoint] = deque([base])

    while queue:
        node = queue.popleft()
        for neighbor in connectivity_graph.neighbors(node):
            if neighbor not in allowed_nodes or neighbor in reachable_nodes:
                continue
            reachable_nodes.add(neighbor)
            queue.append(neighbor)

    return all(
        positions[robot_id] in reachable_nodes
        for robot_id in required_robot_ids
    )


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
        midpoint_connectivity_ids = sorted(positions)
        static_positions = [
            positions[other_robot_id]
            for other_robot_id in midpoint_connectivity_ids
            if other_robot_id != robot_id
        ]
        static_positions.append(base)

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
            for other_robot_id in midpoint_connectivity_ids
        ]
        if not temporary_los_connectivity_check(
            grid_obj,
            simulated_midpoint_positions,
            base,
        ):
            return False, "temporary midpoint connectivity to base failed"

    simulated_positions = dict(positions)
    simulated_positions[robot_id] = candidate_position
    required_robot_ids = set(positions)
    if not _robots_connected_to_base(
        simulated_positions,
        required_robot_ids,
        base,
        connectivity_graph,
    ):
        return False, "required connectivity to base failed"

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
    1. Preference for entering the replanned path.
    2. Minimum hop distance to the next missing path vertex.
    3. Maximum reachable progress along `path_new`.
    4. Minimum single-step travel distance.
    5. Minimum distance to the goal vertex.
    """
    if current_position not in connectivity_graph:
        return None

    goal = path_new[-1]
    best_candidate: GridPoint | None = None
    occupied_vertices = set(positions.values())
    frontier_vertex = _first_unoccupied_path_vertex(path_new, occupied_vertices)
    best_score = (
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
        float("-inf"),
    )

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
        candidate_on_path = 1.0 if candidate_position in path_new else 0.0
        frontier_hops = float(
            _shortest_hops(
                connectivity_graph,
                candidate_position,
                frontier_vertex,
            )
        )
        step_distance = _euclidean_distance(current_position, candidate_position)
        goal_distance = _euclidean_distance(candidate_position, goal)
        score = (
            candidate_on_path,
            -frontier_hops,
            float(path_progress),
            -step_distance,
            -goal_distance,
        )

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


def reactive_replan(
    updated_graph: nx.Graph[GridPoint],
    path_new: Sequence[GridPoint],
    initial_positions: RobotPositions,
    grid_obj: MapGrid | None = None,
    max_steps: int = DEFAULT_MAX_REACTIVE_STEPS,
    frozen_robot_ids: set[int] | None = None,
    record_blocked_attempts: bool = True,
) -> list[MovementSnapshot]:
    """Runs Algorithm 1 and returns reactive movement snapshots.

    Args:
        updated_graph: Visibility graph after obstacle-induced updates.
        path_new: Replanned path `[base, ..., goal]`.
        initial_positions: Current robot positions by robot id.
        grid_obj: Occupancy grid used by midpoint LOS checks.
        max_steps: Safety bound for attempted movement snapshots.
        frozen_robot_ids: Robot ids that must remain stationary during
            replanning.
        record_blocked_attempts: When `True`, includes blocked move attempts
            in the returned snapshot list.

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

    frozen_ids = set(frozen_robot_ids or set())
    if not frozen_ids.issubset(expected_ids):
        raise ValueError("frozen_robot_ids must be valid robot ids.")

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

    seen_states: set[tuple[GridPoint, ...]] = set()

    def _termination_reached() -> bool:
        return (
            positions[LEADER_ROBOT_ID] == goal
            and _robots_connected_to_base(
                positions,
                set(positions),
                base,
                updated_graph,
            )
        )

    step = 0
    while not _termination_reached() and step < max_steps:
        state_key = tuple(positions[robot_id] for robot_id in range(n_robots))
        if state_key in seen_states:
            step += 1
            _append_snapshot(
                snapshots,
                step,
                None,
                None,
                None,
                positions,
                False,
                "Reactive replanning cycle detected: no convergent progress.",
            )
            break
        seen_states.add(state_key)

        moved_this_round = False

        for robot_id in range(n_robots):
            if robot_id in frozen_ids:
                continue

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

            if is_valid:
                step += 1
                positions[robot_id] = candidate_position
                moved_this_round = True
                description = (
                    f"Step {step}: r{robot_id + 1} moves "
                    f"{current_position}->{candidate_position}"
                )
                snapshot_to_position = candidate_position
            else:
                if not record_blocked_attempts:
                    continue
                step += 1
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

            if _termination_reached() or step >= max_steps:
                break

        if _termination_reached() or step >= max_steps:
            break

        if not moved_this_round:
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
    frozen_robot_ids: set[int] | None = None,
    max_relay_robots: int | None = None,
    record_blocked_attempts: bool = True,
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
        frozen_robot_ids: Robot ids that must remain stationary during
            replanning.
        max_relay_robots: Optional upper bound on support relay robots in the
            replanned path. The available robot count is always enforced.
        record_blocked_attempts: When `True`, includes blocked move attempts
            in the returned snapshot list.

    Returns:
        `(cost, path_new, snapshots, updated_graph)` where `cost` and `path_new`
        come from the updated-graph shortest path and `snapshots` contains the
        resulting movement schedule under Algorithm 1.

    Raises:
        ValueError: If `obstacle_point` is provided without `grid_obj`.
    """
    updated_graph = visibility_graph.copy()
    required_vertices = [source, target]
    if obstacle_point is not None:
        required_vertices.append(obstacle_point)
    updated_graph = _add_required_vertices_with_current_los(
        updated_graph,
        required_vertices,
        grid_obj,
    )

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

    updated_graph = _remove_edges_without_current_los(updated_graph, grid_obj)

    available_robot_count = len(initial_positions)
    if available_robot_count <= 0:
        return (
            INFINITE_PATH_COST,
            [],
            _infeasible_snapshot(
                dict(initial_positions),
                "Reactive replanning failed: no robots are available.",
            ),
            updated_graph,
        )

    max_edges_allowed = available_robot_count
    if max_relay_robots is not None:
        if max_relay_robots < 0:
            raise ValueError(
                "max_relay_robots must be greater than or equal to 0; "
                f"received max_relay_robots={max_relay_robots}."
            )
        max_edges_allowed = min(max_edges_allowed, max_relay_robots + 1)

    path_cost, path_new = relay_dijkstra_with_edge_cap(
        updated_graph,
        source,
        target,
        lam,
        max_edges=max_edges_allowed,
    )

    if path_cost == INFINITE_PATH_COST:
        unreachable_snapshots = _infeasible_snapshot(
            dict(initial_positions),
            (
                "Reactive replanning failed: no feasible path after graph "
                "update within available robot count."
            ),
        )
        return path_cost, [], unreachable_snapshots, updated_graph

    effective_initial_positions = dict(initial_positions)
    required_robot_count = len(path_new) - 1
    if len(effective_initial_positions) < required_robot_count:
        infeasible_snapshots = _infeasible_snapshot(
            effective_initial_positions,
            (
                "Reactive replanning failed: path requires "
                f"{required_robot_count} robot(s), but only "
                f"{len(effective_initial_positions)} are available."
            ),
        )
        return INFINITE_PATH_COST, [], infeasible_snapshots, updated_graph

    snapshots = reactive_replan(
        updated_graph,
        path_new,
        effective_initial_positions,
        grid_obj=grid_obj,
        max_steps=max_steps,
        frozen_robot_ids=frozen_robot_ids,
        record_blocked_attempts=record_blocked_attempts,
    )
    return path_cost, path_new, snapshots, updated_graph
