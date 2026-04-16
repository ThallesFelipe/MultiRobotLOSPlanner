"""Path search with relay-penalty objective for LOS communication chains.

The module provides a Dijkstra search that minimizes path cost
`C(P) = Σ w(e) + (k - 1) * λ`, where `w(e)` is Euclidean edge weight,
`k` is the number of path nodes (relays), and `λ` is the relay penalty.
"""

import heapq
from collections.abc import Hashable, Sequence
from typing import TypeVar

import networkx as nx

NodeT = TypeVar("NodeT", bound=Hashable)

# Default relay penalty λ in C(P) when no custom value is provided.
DEFAULT_RELAY_PENALTY_LAMBDA: float = 1.0

# Sentinel value returned when no feasible path is found.
INFINITE_PATH_COST: float = float("inf")

# Number of terminal nodes in a path (source and target), not relay nodes.
TERMINAL_PATH_NODES: int = 2


def relay_dijkstra(
    graph: nx.Graph[NodeT],
    source: NodeT,
    target: NodeT,
    lam: float = DEFAULT_RELAY_PENALTY_LAMBDA,
    prefer_fewer_relays: bool = True,
) -> tuple[float, list[NodeT]]:
    """Computes a minimum-cost path with relay penalty over a weighted graph.

    Args:
        graph: Visibility graph whose edges include a numeric `weight`.
        source: Start node for the path search.
        target: Goal node for the path search.
        lam: Relay penalty `λ` added per traversed edge in the path cost `C(P)`.
        prefer_fewer_relays: When `True`, the search prioritizes paths with
            fewer traversed edges (fewer relay robots) before minimizing
            penalized cost and geometric distance.

    Returns:
        A tuple `(cost, path)` where `cost` is the objective value `C(P)` and
        `path` is the corresponding node sequence. Returns `(inf, [])` when
        `source` or `target` is missing, or when no path exists.

    Raises:
        ValueError: If `lam` is negative.
    """
    if lam < 0:
        raise ValueError(
            "Relay penalty lambda must be greater than or equal to 0; "
            f"received lam={lam}."
        )

    if source not in graph or target not in graph:
        return INFINITE_PATH_COST, []

    tie_breaker_counter = 0

    if prefer_fewer_relays:
        def push_fewer_relay_candidate(
            priority_queue: list[tuple[int, float, float, int, NodeT, list[NodeT]]],
            traversed_edge_count: int,
            total_cost: float,
            accumulated_euclidean_distance: float,
            node: NodeT,
            path: list[NodeT],
        ) -> None:
            """Pushes one lexicographic candidate favoring fewer traversed edges."""
            nonlocal tie_breaker_counter
            tie_breaker_counter += 1
            heapq.heappush(
                priority_queue,
                (
                    traversed_edge_count,
                    total_cost,
                    accumulated_euclidean_distance,
                    tie_breaker_counter,
                    node,
                    path,
                ),
            )

        priority_queue_fewer_relays: list[
            tuple[int, float, float, int, NodeT, list[NodeT]]
        ] = []
        best_label_by_node: dict[NodeT, tuple[int, float, float]] = {
            source: (0, 0.0, 0.0)
        }

        push_fewer_relay_candidate(
            priority_queue_fewer_relays,
            0,
            0.0,
            0.0,
            source,
            [source],
        )

        while priority_queue_fewer_relays:
            (
                traversed_edge_count,
                total_cost,
                accumulated_euclidean_distance,
                _,
                current_node,
                current_path,
            ) = heapq.heappop(priority_queue_fewer_relays)

            current_label = (
                traversed_edge_count,
                total_cost,
                accumulated_euclidean_distance,
            )
            if best_label_by_node.get(current_node) != current_label:
                continue

            if current_node == target:
                return total_cost, current_path

            for neighbor_node in graph.neighbors(current_node):
                edge_distance = graph[current_node][neighbor_node]["weight"]
                new_accumulated_euclidean_distance = (
                    accumulated_euclidean_distance + edge_distance
                )
                new_traversed_edge_count = traversed_edge_count + 1
                new_total_cost = new_accumulated_euclidean_distance + (
                    new_traversed_edge_count * lam
                )

                new_label = (
                    new_traversed_edge_count,
                    new_total_cost,
                    new_accumulated_euclidean_distance,
                )

                best_known_label = best_label_by_node.get(neighbor_node)
                if best_known_label is not None and new_label >= best_known_label:
                    continue

                best_label_by_node[neighbor_node] = new_label
                push_fewer_relay_candidate(
                    priority_queue_fewer_relays,
                    new_traversed_edge_count,
                    new_total_cost,
                    new_accumulated_euclidean_distance,
                    neighbor_node,
                    current_path + [neighbor_node],
                )

        return INFINITE_PATH_COST, []

    def push_candidate(
        priority_queue: list[tuple[float, float, int, int, NodeT, list[NodeT]]],
        total_cost: float,
        accumulated_euclidean_distance: float,
        traversed_edge_count: int,
        node: NodeT,
        path: list[NodeT],
    ) -> None:
        """Pushes a candidate state into the priority queue."""
        nonlocal tie_breaker_counter
        tie_breaker_counter += 1
        heapq.heappush(
            priority_queue,
            (
                total_cost,
                accumulated_euclidean_distance,
                traversed_edge_count,
                tie_breaker_counter,
                node,
                path,
            ),
        )

    priority_queue: list[tuple[float, float, int, int, NodeT, list[NodeT]]] = []
    push_candidate(priority_queue, 0.0, 0.0, 0, source, [source])
    settled_nodes: set[NodeT] = set()

    while priority_queue:
        (
            _,
            accumulated_euclidean_distance,
            traversed_edge_count,
            _,
            current_node,
            current_path,
        ) = heapq.heappop(priority_queue)

        if current_node in settled_nodes:
            continue
        settled_nodes.add(current_node)

        if current_node == target:
            total_path_cost = accumulated_euclidean_distance + (
                (len(current_path) - 1) * lam
            )
            return total_path_cost, current_path

        for neighbor_node in graph.neighbors(current_node):
            if neighbor_node in settled_nodes:
                continue

            edge_distance = graph[current_node][neighbor_node]["weight"]
            new_accumulated_euclidean_distance = (
                accumulated_euclidean_distance + edge_distance
            )
            new_traversed_edge_count = traversed_edge_count + 1
            new_total_cost = new_accumulated_euclidean_distance + (
                new_traversed_edge_count * lam
            )

            push_candidate(
                priority_queue,
                new_total_cost,
                new_accumulated_euclidean_distance,
                new_traversed_edge_count,
                neighbor_node,
                current_path + [neighbor_node],
            )

    return INFINITE_PATH_COST, []


def count_relay_robots(path: Sequence[Hashable]) -> int:
    """Counts relay nodes required by a source-to-target path.

    Args:
        path: Path as an ordered sequence of nodes.

    Returns:
        Number of relay nodes, excluding source and target endpoints.
    """
    return max(0, len(path) - TERMINAL_PATH_NODES)