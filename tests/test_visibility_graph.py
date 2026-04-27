"""Tests for visibility-graph construction over occupancy-grid maps."""

import math

import networkx as nx

from core.map_grid import MapGrid
from core.visibility_graph import build_visibility_graph

GridPoint = tuple[int, int]


def test_visibility_graph_adds_nodes_and_positions() -> None:
    """Ensures all provided vertices become nodes with plotting coordinates."""
    grid = MapGrid(6, 6)
    vertices: list[GridPoint] = [(1, 1), (2, 4), (4, 2)]

    graph = build_visibility_graph(grid, vertices)

    assert isinstance(graph, nx.Graph)
    assert set(graph.nodes) == set(vertices)
    assert graph.nodes[(1, 1)]["pos"] == (1, -1)
    assert graph.nodes[(2, 4)]["pos"] == (4, -2)
    assert graph.nodes[(4, 2)]["pos"] == (2, -4)
    assert graph.number_of_edges() == 3


def test_visibility_graph_blocks_edges_without_line_of_sight() -> None:
    """Checks that LOS-blocked vertex pairs are omitted from graph edges."""
    grid = MapGrid(6, 6)
    grid.add_obstacle_rect(0, 6, 2, 3)
    vertices: list[GridPoint] = [(1, 1), (4, 1), (1, 4), (4, 4)]

    graph = build_visibility_graph(grid, vertices)

    expected_edges = {
        frozenset(((1, 1), (4, 1))),
        frozenset(((1, 4), (4, 4))),
    }
    actual_edges = {frozenset(edge) for edge in graph.edges}
    assert actual_edges == expected_edges


def test_visibility_graph_edge_weight_is_euclidean_distance() -> None:
    """Validates Euclidean edge weights assigned in the visibility graph."""
    grid = MapGrid(6, 6)
    vertices: list[GridPoint] = [(0, 0), (3, 4)]

    graph = build_visibility_graph(grid, vertices)

    assert graph.has_edge((0, 0), (3, 4))
    edge_weight = float(graph[(0, 0)][(3, 4)]["weight"])
    assert math.isclose(edge_weight, 5.0, rel_tol=1e-12, abs_tol=1e-12)


def test_visibility_graph_out_of_bounds_vertex_has_no_edges() -> None:
    """Ensures out-of-bounds endpoints prevent LOS edges from being created."""
    grid = MapGrid(5, 5)
    vertices: list[GridPoint] = [(1, 1), (6, 6)]

    graph = build_visibility_graph(grid, vertices)

    assert set(graph.nodes) == set(vertices)
    assert graph.number_of_edges() == 0



def test_visibility_graph_accepts_custom_diagonal_flank_policy() -> None:
    """Checks custom diagonal policy wiring for visibility-graph LOS checks."""
    grid = MapGrid(3, 3)
    grid.add_obstacle(1, 0)
    vertices: list[GridPoint] = [(0, 0), (1, 1)]

    strict_graph = build_visibility_graph(
        grid,
        vertices,
        diagonal_flank_policy="either",
    )
    permissive_graph = build_visibility_graph(
        grid,
        vertices,
        diagonal_flank_policy="both",
    )

    assert strict_graph.number_of_edges() == 0
    assert permissive_graph.number_of_edges() == 1

def test_visibility_graph_empty_vertices_returns_empty_graph() -> None:
    """Checks that an empty vertex set yields an empty visibility graph."""
    grid = MapGrid(5, 5)
    graph = build_visibility_graph(grid, [])

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_visibility_graph_single_vertex_has_no_edges() -> None:
    """Confirms a one-vertex visibility graph contains no edges."""
    grid = MapGrid(5, 5)
    graph = build_visibility_graph(grid, [(2, 2)])

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0
