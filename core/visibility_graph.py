"""Visibility-graph construction from occupancy-grid maps.

This module builds a geometric visibility graph `G = (V, E)` where each vertex
is a point in the occupancy grid and each edge represents a LOS-valid segment
through free configuration space (`C_free`).
"""

from collections.abc import Sequence

import networkx as nx
import numpy as np

from .map_grid import GridPoint, MapGrid
from .visibility import (
    DEFAULT_DIAGONAL_FLANK_POLICY,
    DiagonalFlankPolicy,
    has_line_of_sight,
)

# Node attribute name that stores plotting coordinates (x, y).
NODE_POSITION_ATTRIBUTE: str = "pos"


def build_visibility_graph(
    grid: MapGrid,
    vertices: Sequence[GridPoint] | None,
    diagonal_flank_policy: DiagonalFlankPolicy = DEFAULT_DIAGONAL_FLANK_POLICY,
) -> nx.Graph[GridPoint]:
    """Builds a LOS-constrained visibility graph from occupancy-grid vertices.

    Args:
        grid: Occupancy grid with `C_free` and `C_obs` cells.
        vertices: Candidate visibility-graph vertices as `(row, col)` tuples.
        diagonal_flank_policy: Corner-cutting policy used during LOS checks.

    Returns:
        An undirected weighted visibility graph. Every included edge has a
        Euclidean weight and connects two vertices that satisfy LOS.

    Raises:
        ValueError: If `vertices` is `None`.
    """
    if vertices is None:
        raise ValueError("vertices must be a sequence of grid points, not None.")

    visibility_graph: nx.Graph[GridPoint] = nx.Graph()
    for vertex in vertices:
        visibility_graph.add_node(
            vertex,
            **{NODE_POSITION_ATTRIBUTE: (vertex[1], -vertex[0])},
        )

    for source_index in range(len(vertices)):
        for target_index in range(source_index + 1, len(vertices)):
            source_vertex, target_vertex = (
                vertices[source_index],
                vertices[target_index],
            )
            if has_line_of_sight(
                grid,
                source_vertex,
                target_vertex,
                diagonal_flank_policy=diagonal_flank_policy,
            ):
                euclidean_distance = float(
                    np.hypot(
                        target_vertex[0] - source_vertex[0],
                        target_vertex[1] - source_vertex[1],
                    )
                )
                visibility_graph.add_edge(
                    source_vertex,
                    target_vertex,
                    weight=euclidean_distance,
                )

    return visibility_graph
