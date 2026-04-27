"""Tests for relay-penalty shortest-path search and relay counting helpers."""

import math

import networkx as nx
import pytest

from algorithms.relay_dijkstra import (
    count_relay_robots,
    relay_dijkstra,
    relay_dijkstra_with_edge_cap,
)

NodeName = str
WeightedEdge = tuple[NodeName, NodeName, float]
PathNodes = list[NodeName]


def _build_graph(edges: list[WeightedEdge]) -> nx.Graph[NodeName]:
    """Creates an undirected weighted graph for relay path tests."""
    graph: nx.Graph[NodeName] = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return graph


def test_relay_dijkstra_finds_minimum_penalized_path() -> None:
    """Validates shortest-path selection using Euclidean sum plus relay penalty."""
    graph = _build_graph(
        [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("A", "C", 4.0),
        ]
    )

    cost, path = relay_dijkstra(graph, "A", "C", lam=1.0)

    assert path == ["A", "B", "C"]
    assert math.isclose(cost, 4.0, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("lam", "expected_cost", "expected_path"),
    [
        (0.0, 2.0, ["A", "B", "C"]),
        (1.0, 3.5, ["A", "C"]),
    ],
)
def test_relay_dijkstra_penalty_changes_selected_path(
    lam: float,
    expected_cost: float,
    expected_path: PathNodes,
) -> None:
    """Checks that larger relay penalty biases the solution toward fewer hops."""
    graph = _build_graph(
        [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("A", "C", 2.5),
        ]
    )

    cost, path = relay_dijkstra(graph, "A", "C", lam=lam)

    assert path == expected_path
    assert math.isclose(cost, expected_cost, rel_tol=1e-12, abs_tol=1e-12)


def test_relay_dijkstra_uses_default_lambda() -> None:
    """Ensures omitted lambda defaults to one relay-penalty unit per edge."""
    graph = _build_graph(
        [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("A", "C", 2.5),
        ]
    )

    cost, path = relay_dijkstra(graph, "A", "C")

    assert path == ["A", "C"]
    assert math.isclose(cost, 3.5, rel_tol=1e-12, abs_tol=1e-12)


def test_relay_dijkstra_can_prioritize_fewer_relays() -> None:
    """Checks optional lexicographic mode that prefers fewer relay hops."""
    graph = _build_graph(
        [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("A", "C", 4.0),
        ]
    )

    cost_default, path_default = relay_dijkstra(graph, "A", "C", lam=1.0)
    cost_fewer_relays, path_fewer_relays = relay_dijkstra(
        graph,
        "A",
        "C",
        lam=1.0,
        prefer_fewer_relays=True,
    )

    assert path_default == ["A", "B", "C"]
    assert math.isclose(cost_default, 4.0, rel_tol=1e-12, abs_tol=1e-12)

    assert path_fewer_relays == ["A", "C"]
    assert math.isclose(cost_fewer_relays, 5.0, rel_tol=1e-12, abs_tol=1e-12)


def test_relay_dijkstra_with_edge_cap_chooses_feasible_alternative() -> None:
    """Caps path segments by available robot count without dropping lambda cost."""
    graph = _build_graph(
        [
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "D", 1.0),
            ("A", "X", 2.0),
            ("X", "D", 2.0),
        ]
    )

    cost, path = relay_dijkstra_with_edge_cap(
        graph,
        "A",
        "D",
        lam=1.0,
        max_edges=2,
    )

    assert path == ["A", "X", "D"]
    assert math.isclose(cost, 6.0, rel_tol=1e-12, abs_tol=1e-12)


def test_relay_dijkstra_source_equals_target() -> None:
    """Confirms zero-cost trivial path when source and target are identical."""
    graph: nx.Graph[NodeName] = nx.Graph()
    graph.add_node("A")

    cost, path = relay_dijkstra(graph, "A", "A", lam=5.0)

    assert path == ["A"]
    assert math.isclose(cost, 0.0, rel_tol=1e-12, abs_tol=1e-12)


def test_relay_dijkstra_missing_endpoint_returns_infinite_cost() -> None:
    """Ensures missing source or target returns no path and infinite objective."""
    graph = _build_graph([("A", "B", 1.0)])

    cost, path = relay_dijkstra(graph, "A", "Z")

    assert math.isinf(cost)
    assert path == []


def test_relay_dijkstra_disconnected_target_returns_infinite_cost() -> None:
    """Checks disconnected components correctly report no feasible path."""
    graph: nx.Graph[NodeName] = nx.Graph()
    graph.add_edge("A", "B", weight=1.0)
    graph.add_node("C")

    cost, path = relay_dijkstra(graph, "A", "C", lam=1.0)

    assert math.isinf(cost)
    assert path == []


def test_relay_dijkstra_negative_lambda_raises_value_error() -> None:
    """Validates argument checking for invalid negative relay penalties."""
    graph = _build_graph([("A", "B", 1.0)])

    with pytest.raises(ValueError):
        relay_dijkstra(graph, "A", "B", lam=-0.1)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ([], 0),
        (["A"], 0),
        (["A", "B"], 0),
        (["A", "B", "C"], 1),
        (["A", "B", "C", "D"], 2),
    ],
)
def test_count_relay_robots(path: PathNodes, expected: int) -> None:
    """Ensures relay counting excludes source and target terminal nodes."""
    assert count_relay_robots(path) == expected
