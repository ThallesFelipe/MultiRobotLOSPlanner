"""Deterministic ordered progression for LOS-constrained relay deployment.

The implementation follows the paper's deterministic policy where robots move
sequentially from the base toward the goal while preserving line-of-sight (LOS)
connectivity to the communication chain.
"""

from collections.abc import Sequence
from typing import TypedDict

import networkx as nx

from algorithms.connectivity_checks import (
    ConnectivityPoint,
    bfs_connected as _bfs_connected,
    midpoint as _midpoint,
    midpoint_has_los_to_chain as _midpoint_has_los_to_chain,
    temporary_los_connectivity_check as _temporary_los_connectivity_check,
)
from algorithms.relay_dijkstra import (
    DEFAULT_RELAY_PENALTY_LAMBDA,
    INFINITE_PATH_COST,
    relay_dijkstra,
    relay_dijkstra_with_edge_cap,
)
from core.map_grid import GridPoint, MapGrid
from core.visibility import bresenham

RobotPositions = dict[int, GridPoint]


class MovementSnapshot(TypedDict):
    """State snapshot captured after each scheduled movement attempt."""

    step: int
    robot_id: int | None
    from_pos: GridPoint | None
    to_pos: GridPoint | None
    positions: RobotPositions
    valid: bool
    description: str


class OrderedPathPlanningResult(TypedDict):
    """Result payload for ordered planning over a visibility graph."""

    source: GridPoint
    target: GridPoint
    path_cost: float
    path: list[GridPoint]
    movement_snapshots: list[MovementSnapshot]
    traversed_cells: int
    robots_used: int


def _preserves_leader_front_sequential_formation(
    progress: dict[int, int],
    moving_robot_id: int,
    candidate_progress: int,
) -> bool:
    """Algoritmo 2, linhas 1-6: valida formacao e prioridade do lider.

    Formation invariants:
    - r0 is never behind any follower.
    - Adjacent robot progress differs by at most one path vertex.
    """
    simulated_progress = dict(progress)
    simulated_progress[moving_robot_id] = candidate_progress

    n_robots = len(simulated_progress)
    ordered_progression = [
        simulated_progress[robot_id] for robot_id in range(n_robots)
    ]

    for leader_index in range(n_robots - 1):
        leader_progress = ordered_progression[leader_index]
        follower_progress = ordered_progression[leader_index + 1]

        if leader_progress < follower_progress:
            return False
        if leader_progress - follower_progress > 1:
            return False

    return True


def deterministic_sequence(n_nodes: int) -> list[int]:
    """Builds the deterministic ordered-progression robot schedule."""
    if n_nodes < 2:
        return []

    n_robots = n_nodes - 1
    sequence: list[int] = []
    for block_index in range(n_robots):
        for robot_index in range(block_index, -1, -1):
            sequence.append(robot_index)
    return sequence


def count_path_traversed_cells(path: Sequence[GridPoint]) -> int:
    """Counts traversed occupancy-grid cells along a piecewise-linear path."""
    if not path:
        return 0
    if len(path) == 1:
        return 1

    traversed_cell_count = 0
    for segment_index in range(len(path) - 1):
        start_point = path[segment_index]
        end_point = path[segment_index + 1]

        segment_cells = list(
            bresenham(
                start_point[0],
                start_point[1],
                end_point[0],
                end_point[1],
            )
        )
        if segment_index > 0:
            segment_cells = segment_cells[1:]

        traversed_cell_count += len(segment_cells)

    return traversed_cell_count


def ordered_progression(
    path: Sequence[GridPoint],
    grid_obj: MapGrid | None = None,
    vis_graph: nx.Graph[GridPoint] | None = None,
    robot_count: int | None = None,
) -> list[MovementSnapshot]:
    """Generates deterministic movement snapshots for the relay chain.

    Mapeamento do pseudo-codigo do artigo:
    - Algoritmo 1: o `for step, robot_id in enumerate(sequence, ...)` executa
      a movimentacao coordenada ate todos os vertices do caminho serem ocupados.
    - Algoritmo 2: os blocos `valid` abaixo fazem a validacao de formacao,
      conectividade no ponto medio e conectividade final antes de mover o robo.

    Args:
        path: Ordered path `[base, n1, ..., nk, goal]` selected for deployment.
        grid_obj: Occupancy grid used for midpoint LOS validation.
        vis_graph: Visibility graph used for graph-based connectivity checks.
        robot_count: Available fleet size. When omitted, uses the minimum
            `len(path) - 1` robots required by the path.

    Returns:
        A list of per-step movement snapshots.
    """
    if len(path) < 2:
        return []

    n_nodes = len(path)
    required_robot_count = n_nodes - 1
    n_robots = required_robot_count if robot_count is None else robot_count
    if n_robots < required_robot_count:
        raise ValueError(
            "robot_count is insufficient for path deployment: "
            f"path requires {required_robot_count}, received {n_robots}."
        )

    # Algoritmo 1, entrada P: todos os robos comecam na base do caminho-alvo.
    positions: RobotPositions = {robot_id: path[0] for robot_id in range(n_robots)}
    progress: dict[int, int] = {robot_id: 0 for robot_id in range(n_robots)}

    expected_move_count = total_moves_formula(n_nodes)
    # Algoritmo 1, linhas 1-2: a sequencia deterministica substitui o while +
    # for each robot do artigo para o caso de planejamento ordenado.
    sequence = deterministic_sequence(n_nodes)
    if len(sequence) != expected_move_count:
        raise RuntimeError(
            "Deterministic sequence length does not match the expected count: "
            f"got {len(sequence)}, expected {expected_move_count}."
        )

    snapshots: list[MovementSnapshot] = [
        {
            "step": 0,
            "robot_id": None,
            "from_pos": None,
            "to_pos": None,
            "positions": dict(positions),
            "valid": True,
            "description": (
                f"Initial state: {n_robots} robot(s) at base {path[0]}"
            ),
        }
    ]

    # Algoritmo 1, linha 1: repete ate a sequencia preencher o caminho-alvo.
    for step, robot_id in enumerate(sequence, start=1):
        current_pos = positions[robot_id]
        current_progress = progress[robot_id]
        next_progress = current_progress + 1

        if next_progress >= n_nodes:
            snapshots.append(
                {
                    "step": step,
                    "robot_id": robot_id,
                    "from_pos": current_pos,
                    "to_pos": current_pos,
                    "positions": dict(positions),
                    "valid": True,
                    "description": (
                        f"Step {step}: r{robot_id + 1} already at destination "
                        f"{current_pos}"
                    ),
                }
            )
            continue

        # Algoritmo 1, linha 3: pcand e o proximo vertice do Ptarget para r.
        next_pos = path[next_progress]
        valid = True
        reason = ""

        # Algoritmo 1, linha 4 + Algoritmo 2, linhas 1-6:
        # VALIDATEMOVE com regras de formacao sequencial e prioridade do lider.
        if not _preserves_leader_front_sequential_formation(
            progress,
            robot_id,
            next_progress,
        ):
            valid = False
            reason = "leader-front sequential formation violated"

        if valid and grid_obj is not None:
            # Algoritmo 2, linhas 7-10: valida LOS do ponto medio do movimento
            # com a cadeia estatica antes de aprovar a transicao.
            static_positions = [
                positions[index]
                for index in range(n_robots)
                if index != robot_id
            ]
            if not _midpoint_has_los_to_chain(
                grid_obj,
                current_pos,
                next_pos,
                static_positions,
            ):
                valid = False
                reason = "midpoint LOS to static chain failed"

        if valid and grid_obj is not None:
            # Algoritmo 2, linhas 7-13: pmid, Psim, Gtemp e teste de conexao
            # ate a base durante o movimento.
            midpoint = _midpoint(current_pos, next_pos)
            simulated_midpoint_positions: list[ConnectivityPoint] = [
                midpoint if index == robot_id else positions[index]
                for index in range(n_robots)
            ]
            if not _temporary_los_connectivity_check(
                grid_obj,
                simulated_midpoint_positions,
                path[0],
            ):
                valid = False
                reason = "temporary LOS connectivity to base failed"

        if valid and vis_graph is not None:
            # Algoritmo 2, linhas 9-13: checagem em grafo de visibilidade para
            # garantir que o estado final tambem conecta todos os robos a base.
            simulated_final_positions = dict(positions)
            simulated_final_positions[robot_id] = next_pos
            if not _bfs_connected(
                list(simulated_final_positions.values()),
                path[0],
                vis_graph,
            ):
                valid = False
                reason = "visibility-graph BFS connectivity failed"

        if valid:
            # Algoritmo 1, linhas 5-7 e Algoritmo 2, linha 14: candidato seguro;
            # executa o movimento e atualiza P/progress.
            positions[robot_id] = next_pos
            progress[robot_id] = next_progress
            description = (
                f"Step {step}: r{robot_id + 1} moves {current_pos}->{next_pos}"
            )
        else:
            # Algoritmo 1, linhas 8-9: movimento rejeitado; robo permanece onde
            # estava e o snapshot registra o motivo.
            description = (
                f"Step {step}: r{robot_id + 1} blocked ({reason}) at "
                f"{current_pos}"
            )

        snapshots.append(
            {
                "step": step,
                "robot_id": robot_id,
                "from_pos": current_pos,
                "to_pos": next_pos if valid else current_pos,
                "positions": dict(positions),
                "valid": valid,
                "description": description,
            }
        )

    return snapshots


def plan_ordered_progression_on_visibility_graph(
    vis_graph: nx.Graph[GridPoint],
    source: GridPoint,
    target: GridPoint,
    lam: float = DEFAULT_RELAY_PENALTY_LAMBDA,
    grid_obj: MapGrid | None = None,
    prefer_fewer_relays: bool = False,
    robot_count: int | None = None,
) -> OrderedPathPlanningResult:
    """Plans a visibility-graph path and expands it with ordered progression.

    Args:
        vis_graph: Visibility graph used for path search.
        source: Start occupancy-grid point.
        target: Goal occupancy-grid point.
        lam: Relay penalty used by relay-penalty Dijkstra.
        grid_obj: Optional occupancy grid for midpoint LOS checks.
        prefer_fewer_relays: When `True`, prioritizes paths with fewer relays
            before penalized-cost minimization.
        robot_count: Available fleet size. Paths requiring more robots are
            treated as infeasible.

    Returns:
        Planning result with the path, movement snapshots, and summary metrics.
        When no feasible path exists, the path and snapshots are empty.
    """
    if robot_count is None:
        path_cost, planned_path = relay_dijkstra(
            vis_graph,
            source,
            target,
            lam=lam,
            prefer_fewer_relays=prefer_fewer_relays,
        )
    else:
        path_cost, planned_path = relay_dijkstra_with_edge_cap(
            vis_graph,
            source,
            target,
            lam=lam,
            max_edges=robot_count,
            prefer_fewer_relays=prefer_fewer_relays,
        )

    if path_cost == INFINITE_PATH_COST or not planned_path:
        return {
            "source": source,
            "target": target,
            "path_cost": INFINITE_PATH_COST,
            "path": [],
            "movement_snapshots": [],
            "traversed_cells": 0,
            "robots_used": 0,
        }

    movement_snapshots = ordered_progression(
        planned_path,
        grid_obj=grid_obj,
        vis_graph=vis_graph,
        robot_count=robot_count,
    )
    return {
        "source": source,
        "target": target,
        "path_cost": path_cost,
        "path": planned_path,
        "movement_snapshots": movement_snapshots,
        "traversed_cells": count_path_traversed_cells(planned_path),
        "robots_used": max(0, len(planned_path) - 1),
    }


def total_moves_formula(n_nodes: int) -> int:
    """Returns the deterministic ordered-progression movement count."""
    if n_nodes < 2:
        return 0

    n_robots = n_nodes - 1
    return (n_robots * (n_robots + 1)) // 2
