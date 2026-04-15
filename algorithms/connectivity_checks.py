"""Shared LOS and connectivity checks for relay-movement validation.

These utilities are used by both deterministic ordered progression and reactive
replanning to keep movement validation logic consistent across planners.
"""

from collections import deque
from collections.abc import Sequence

import networkx as nx

from core.map_grid import GridPoint, MapGrid
from core.visibility import bresenham

FloatPoint = tuple[float, float]
ConnectivityPoint = GridPoint | FloatPoint


def to_grid_point(point: ConnectivityPoint) -> GridPoint:
    """Rounds a geometric point to the nearest occupancy-grid cell."""
    return (round(point[0]), round(point[1]))


def midpoint(p1: GridPoint, p2: GridPoint) -> FloatPoint:
    """Computes the geometric midpoint between two occupancy-grid points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def has_los_between_points(
    grid_obj: MapGrid,
    start_point: ConnectivityPoint,
    end_point: ConnectivityPoint,
) -> bool:
    """Checks LOS between two points after projecting them to grid cells."""
    start_grid_point = to_grid_point(start_point)
    end_grid_point = to_grid_point(end_point)

    for row, col in bresenham(
        start_grid_point[0],
        start_grid_point[1],
        end_grid_point[0],
        end_grid_point[1],
    ):
        if not grid_obj.in_bounds(row, col):
            return False
        if not grid_obj.is_free(row, col):
            return False
    return True


def midpoint_has_los_to_chain(
    grid_obj: MapGrid,
    p_from: GridPoint,
    p_to: GridPoint,
    static_positions: Sequence[GridPoint],
) -> bool:
    """Checks LOS from the movement midpoint to at least one static robot."""
    if not static_positions:
        return True

    movement_midpoint = midpoint(p_from, p_to)
    return any(
        has_los_between_points(grid_obj, movement_midpoint, static_position)
        for static_position in static_positions
    )


def temporary_los_connectivity_check(
    grid_obj: MapGrid,
    positions: Sequence[ConnectivityPoint],
    base: GridPoint,
) -> bool:
    """Validates whether all robots remain LOS-connected to the base."""
    if not positions:
        return True

    temporary_graph: nx.Graph[ConnectivityPoint] = nx.Graph()
    all_positions: list[ConnectivityPoint] = [base, *positions]
    temporary_graph.add_nodes_from(all_positions)

    for source_index in range(len(all_positions)):
        for target_index in range(source_index + 1, len(all_positions)):
            source_position = all_positions[source_index]
            target_position = all_positions[target_index]
            if has_los_between_points(grid_obj, source_position, target_position):
                temporary_graph.add_edge(source_position, target_position)

    reachable: set[ConnectivityPoint] = {base}
    queue: deque[ConnectivityPoint] = deque([base])

    while queue:
        node = queue.popleft()
        for neighbor in temporary_graph.neighbors(node):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    return all(position in reachable for position in positions)


def bfs_connected(
    positions: Sequence[GridPoint],
    base: GridPoint,
    vis_graph: nx.Graph[GridPoint],
) -> bool:
    """Checks connectivity to the base over a precomputed visibility graph."""
    if not positions:
        return True
    if base not in vis_graph:
        return False

    reachable: set[GridPoint] = {base}
    queue: deque[GridPoint] = deque([base])
    occupied_nodes = set(positions)
    allowed_nodes = occupied_nodes | {base}

    while queue:
        node = queue.popleft()
        for neighbor in vis_graph.neighbors(node):
            if neighbor not in reachable and neighbor in allowed_nodes:
                reachable.add(neighbor)
                queue.append(neighbor)

    return occupied_nodes.issubset(reachable)
