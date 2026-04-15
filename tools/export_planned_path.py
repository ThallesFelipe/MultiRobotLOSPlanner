"""Interactive ordered-progression planning over visibility graphs.

Map source options:
- JSON catalog entry (recommended): `presets/maps_catalog.json`
- Python preset factory (legacy): `presets.<module>:<factory>`

Examples:
    python tools/export_planned_path.py
    python tools/export_planned_path.py --interactive
    python tools/export_planned_path.py --map-name map1 --source-point 1,1 --target-point 17,16
"""

from __future__ import annotations

import argparse
from importlib import import_module
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Protocol, cast

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.ordered_progression import (
    OrderedPathPlanningResult,
    plan_ordered_progression_on_visibility_graph,
)
from algorithms.relay_dijkstra import (
    DEFAULT_RELAY_PENALTY_LAMBDA,
    INFINITE_PATH_COST,
)
from core.map_grid import GridPoint, MapGrid
from core.map_processor import MapProcessor
from core.visibility import has_line_of_sight
from core.visibility_graph import build_visibility_graph
from presets.map_catalog import (
    DEFAULT_CATALOG_PATH,
    create_map_from_catalog,
    list_catalog_maps,
)


class MapFactory(Protocol):
    """Callable signature for Python preset map factory functions."""

    def __call__(self, scale_factor: int = 1) -> MapGrid:
        ...


DEFAULT_MAP_NAME = "map1"
DEFAULT_FACTORY_NAME = "create_custom_map"
DEFAULT_SCALE_FACTOR = 1
DEFAULT_DPI = 220
DEFAULT_VERTEX_SOURCE = "boundary"
DEFAULT_BOUNDARY_STRIDE = 1
DEFAULT_MAX_VERTICES = 600

FREE_SPACE_COLOR = "#F4F1E8"
OBSTACLE_COLOR = "#1F2321"
VISIBILITY_EDGE_COLOR = "#C0681B"
VERTEX_COLOR = "#1F4D3A"
PATH_COLOR = "#E2572C"
START_COLOR = "#1F4D3A"
GOAL_COLOR = "#C7472D"


def _load_python_factory(module_name: str, factory_name: str) -> MapFactory:
    """Loads map factory from a Python preset module."""
    try:
        module: ModuleType = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Nao foi possivel importar modulo: {module_name!r}.") from exc

    factory = getattr(module, factory_name, None)
    if factory is None:
        raise ValueError(
            f"Funcao {factory_name!r} nao encontrada no modulo {module_name!r}."
        )
    if not callable(factory):
        raise ValueError(
            f"Atributo {factory_name!r} em {module_name!r} nao e chamavel."
        )

    return cast(MapFactory, factory)


def _default_output_path(source_name: str) -> Path:
    """Builds default output PNG path from map source name."""
    safe_name = source_name.replace(":", "_").replace(".", "_")
    return Path("exports") / f"{safe_name}_planned_path.png"


def _load_map(
    map_name: str | None,
    catalog_path: Path,
    preset_module: str | None,
    factory_name: str,
    scale_factor: int,
) -> tuple[MapGrid, str]:
    """Loads map from JSON catalog (preferred) or Python preset module."""
    if map_name is not None:
        map_grid = create_map_from_catalog(
            map_name=map_name,
            scale_factor=scale_factor,
            catalog_path=catalog_path,
        )
        return (map_grid, f"catalog_{map_name}")

    if preset_module is None:
        raise ValueError("Informe --map-name ou --preset-module.")

    factory = _load_python_factory(preset_module, factory_name)
    map_grid = factory(scale_factor=scale_factor)
    return (map_grid, f"{preset_module}_{factory_name}")


def _extract_boundary_vertices(
    map_grid: MapGrid,
    boundary_stride: int,
    max_vertices: int | None,
) -> list[GridPoint]:
    """Extracts free-space boundary vertices with optional stride and cap."""
    if boundary_stride <= 0:
        raise ValueError("boundary_stride must be greater than 0.")

    rows = map_grid.rows
    cols = map_grid.cols
    vertices: list[GridPoint] = []

    for row in range(rows):
        for col in range(cols):
            if not map_grid.is_free(row, col):
                continue
            if ((row + col) % boundary_stride) != 0:
                continue

            has_obstacle_neighbor = False
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue

                    neighbor_row = row + delta_row
                    neighbor_col = col + delta_col
                    if not map_grid.in_bounds(neighbor_row, neighbor_col):
                        continue
                    if not map_grid.is_free(neighbor_row, neighbor_col):
                        has_obstacle_neighbor = True
                        break

                if has_obstacle_neighbor:
                    break

            if has_obstacle_neighbor:
                vertices.append((row, col))

    if max_vertices is not None and len(vertices) > max_vertices:
        sampling_step = int(math.ceil(len(vertices) / max_vertices))
        vertices = vertices[::sampling_step]

    return vertices


def _build_visibility_graph_with_source(
    map_grid: MapGrid,
    vertex_source: str,
    boundary_stride: int,
    max_vertices: int | None,
) -> nx.Graph[GridPoint]:
    """Builds visibility graph from selected vertex source strategy."""
    if vertex_source == "processor":
        processor = MapProcessor(map_grid)
        return processor.build_initial_visibility_graph()

    if vertex_source == "boundary":
        vertices = _extract_boundary_vertices(
            map_grid=map_grid,
            boundary_stride=boundary_stride,
            max_vertices=max_vertices,
        )
        return build_visibility_graph(map_grid, vertices)

    raise ValueError(
        f"Unknown vertex_source={vertex_source!r}. Use 'processor' or 'boundary'."
    )


def _format_grid_point(point: GridPoint) -> str:
    """Formats a grid point for terminal and plot labels."""
    return f"({point[0]}, {point[1]})"


def _parse_grid_point(raw_value: str) -> GridPoint | None:
    """Parses `(row,col)` or `row col` text into a grid point."""
    normalized = raw_value.strip().replace(";", ",")
    if not normalized:
        return None

    parts: list[str]
    if "," in normalized:
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
    else:
        parts = [part.strip() for part in normalized.split() if part.strip()]

    if len(parts) != 2:
        return None

    try:
        row_value = int(parts[0])
        col_value = int(parts[1])
    except ValueError:
        return None

    return (row_value, col_value)


def _euclidean_distance(point_a: GridPoint, point_b: GridPoint) -> float:
    """Computes Euclidean distance between two occupancy-grid points."""
    return float(
        math.hypot(
            point_b[0] - point_a[0],
            point_b[1] - point_a[1],
        )
    )


def _validate_free_grid_cell(
    map_grid: MapGrid,
    point: GridPoint,
    point_label: str,
) -> None:
    """Validates that a point is in bounds and belongs to free space."""
    row_index, col_index = point
    if not map_grid.in_bounds(row_index, col_index):
        raise ValueError(
            f"{point_label} {_format_grid_point(point)} esta fora dos limites do mapa."
        )
    if not map_grid.is_free(row_index, col_index):
        raise ValueError(
            f"{point_label} {_format_grid_point(point)} nao e celula livre no grid."
        )


def _connect_endpoint_to_visible_nodes(
    graph: nx.Graph[GridPoint],
    map_grid: MapGrid,
    endpoint: GridPoint,
    point_label: str,
) -> int:
    """Connects a free-grid endpoint to all LOS-visible graph nodes.

    Returns:
        Number of LOS-visible neighbors connected to the endpoint node.
    """
    _validate_free_grid_cell(map_grid, endpoint, point_label)

    if endpoint not in graph:
        graph.add_node(endpoint)

    visible_neighbor_count = 0
    for candidate_node in list(graph.nodes):
        if candidate_node == endpoint:
            continue
        if not has_line_of_sight(map_grid, endpoint, candidate_node):
            continue

        visible_neighbor_count += 1
        if not graph.has_edge(endpoint, candidate_node):
            graph.add_edge(
                endpoint,
                candidate_node,
                weight=_euclidean_distance(endpoint, candidate_node),
            )

    return visible_neighbor_count


def _build_planning_graph_with_endpoints(
    map_grid: MapGrid,
    vis_graph: nx.Graph[GridPoint],
    source_point: GridPoint,
    target_point: GridPoint,
) -> tuple[nx.Graph[GridPoint], int, int]:
    """Builds planning graph by attaching free-grid endpoints via LOS edges."""
    planning_graph = vis_graph.copy()
    source_visible_neighbors = _connect_endpoint_to_visible_nodes(
        planning_graph,
        map_grid,
        source_point,
        "Origem",
    )
    target_visible_neighbors = _connect_endpoint_to_visible_nodes(
        planning_graph,
        map_grid,
        target_point,
        "Destino",
    )
    return planning_graph, source_visible_neighbors, target_visible_neighbors


def _prompt_positive_int(label: str, default: int, *, allow_zero: bool = False) -> int:
    """Reads a numeric option from terminal input."""
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default

        try:
            parsed_value = int(raw_value)
        except ValueError:
            print("Valor invalido. Digite um numero inteiro.")
            continue

        if allow_zero and parsed_value == 0:
            return parsed_value
        if parsed_value <= 0:
            print("Valor invalido. Digite um inteiro positivo.")
            continue
        return parsed_value


def _prompt_non_negative_float(label: str, default: float) -> float:
    """Reads a non-negative floating-point option from terminal input."""
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default

        try:
            parsed_value = float(raw_value)
        except ValueError:
            print("Valor invalido. Digite um numero real.")
            continue

        if parsed_value < 0.0:
            print("Valor invalido. Digite um numero maior ou igual a zero.")
            continue
        return parsed_value


def _prompt_yes_no(label: str, default: bool) -> bool:
    """Prompts a yes/no question and returns the resulting boolean."""
    suffix = "S/n" if default else "s/N"
    while True:
        raw_value = input(f"{label} [{suffix}]: ").strip().lower()
        if not raw_value:
            return default
        if raw_value in {"s", "sim", "y", "yes"}:
            return True
        if raw_value in {"n", "nao", "no"}:
            return False
        print("Resposta invalida. Digite s ou n.")


def _prompt_map_name(catalog_path: Path, default_map_name: str | None) -> str:
    """Displays catalog maps and returns the chosen map name."""
    available_maps = list_catalog_maps(catalog_path)
    if not available_maps:
        raise ValueError(
            "Catalogo vazio. Adicione mapas em presets/maps_catalog.json "
            "ou use --preset-module."
        )

    if default_map_name in available_maps:
        selected_default_map = default_map_name
    else:
        selected_default_map = available_maps[0]
    default_index = available_maps.index(selected_default_map) + 1

    print("\nMapas disponiveis no catalogo:")
    for index, map_name in enumerate(available_maps, start=1):
        print(f"  [{index}] {map_name}")

    while True:
        raw_value = input(
            f"Escolha o mapa (numero ou nome) [padrao: {default_index}]: "
        ).strip()
        if not raw_value:
            return selected_default_map

        if raw_value.isdigit():
            map_index = int(raw_value)
            if 1 <= map_index <= len(available_maps):
                return available_maps[map_index - 1]

        if raw_value in available_maps:
            return raw_value

        print("Opcao invalida. Digite um numero da lista ou o nome do mapa.")


def _print_vertex_preview(nodes: list[GridPoint]) -> None:
    """Prints a concise preview of available graph vertices."""
    print(f"\nVertices disponiveis no grafo: {len(nodes)}")
    if not nodes:
        return

    if len(nodes) <= 24:
        for index, node in enumerate(nodes, start=1):
            print(f"  [{index:03d}] {_format_grid_point(node)}")
        return

    print("Primeiros vertices:")
    for index, node in enumerate(nodes[:12], start=1):
        print(f"  [{index:03d}] {_format_grid_point(node)}")

    print("  ...")
    print("Ultimos vertices:")
    offset = len(nodes) - 8
    for index, node in enumerate(nodes[offset:], start=offset + 1):
        print(f"  [{index:03d}] {_format_grid_point(node)}")


def _prompt_endpoint(
    label: str,
    map_grid: MapGrid,
    nodes: list[GridPoint],
    default_point: GridPoint | None = None,
    blocked_grid_point: GridPoint | None = None,
) -> GridPoint:
    """Prompts for a free-grid endpoint used directly in planning."""
    node_set = set(nodes)

    while True:
        if default_point is None:
            default_suffix = ""
        else:
            default_suffix = f" [padrao: {default_point[0]},{default_point[1]}]"

        raw_value = input(f"{label} (row,col){default_suffix}: ").strip()
        if not raw_value:
            if default_point is None:
                print("Valor obrigatorio.")
                continue
            candidate = default_point
        else:
            parsed_point = _parse_grid_point(raw_value)
            if parsed_point is None:
                print("Formato invalido. Use row,col (ex: 5,10).")
                continue
            candidate = parsed_point

        if blocked_grid_point is not None and candidate == blocked_grid_point:
            print("Origem e destino devem ser diferentes no grid.")
            continue

        try:
            _validate_free_grid_cell(map_grid, candidate, label)
        except ValueError as exc:
            print(exc)
            continue

        if candidate not in node_set:
            print(
                f"Ponto {_format_grid_point(candidate)} nao e vertice do grafo. "
                "Ele sera conectado aos vertices com linha de visada (LOS)."
            )

        return candidate


def _resolve_cli_endpoint(
    raw_value: str | None,
    map_grid: MapGrid,
    argument_name: str,
) -> GridPoint:
    """Resolves a CLI endpoint from free grid cell coordinates."""
    if raw_value is None:
        raise ValueError(f"Informe --{argument_name} no formato row,col.")

    parsed_point = _parse_grid_point(raw_value)
    if parsed_point is None:
        raise ValueError(
            f"Valor invalido para --{argument_name}: {raw_value!r}. "
            "Use row,col (ex: 5,10)."
        )

    _validate_free_grid_cell(map_grid, parsed_point, argument_name)
    return parsed_point


def _collect_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Collects interactive options to avoid long command lines."""
    print("\n=== Planejador Interativo de Caminho ===")

    available_maps = list_catalog_maps(args.catalog_path)
    if available_maps:
        args.map_name = _prompt_map_name(args.catalog_path, args.map_name)
        args.preset_module = None
    else:
        print("Catalogo sem mapas. Usando modo legado por modulo Python.")
        args.map_name = None
        default_module = args.preset_module or "presets.map1"
        preset_module = input(
            f"Modulo preset Python [{default_module}]: "
        ).strip()
        args.preset_module = preset_module if preset_module else default_module

    while True:
        vertex_source = input(
            f"Fonte de vertices [boundary/processor] [{args.vertex_source}]: "
        ).strip().lower()
        if not vertex_source:
            break
        if vertex_source in {"boundary", "processor"}:
            args.vertex_source = vertex_source
            break
        print("Opcao invalida. Use boundary ou processor.")

    if args.vertex_source == "boundary":
        args.boundary_stride = _prompt_positive_int(
            "Boundary stride",
            args.boundary_stride,
        )
        args.max_vertices = _prompt_positive_int(
            "Max vertices (0 para sem limite)",
            args.max_vertices,
            allow_zero=True,
        )

    args.scale_factor = _prompt_positive_int("Scale factor", args.scale_factor)
    args.lam = _prompt_non_negative_float("Relay penalty lambda", args.lam)
    args.dpi = _prompt_positive_int("DPI", args.dpi)
    args.show_grid = _prompt_yes_no("Mostrar grade", args.show_grid)
    args.show_vertices = _prompt_yes_no("Mostrar vertices", args.show_vertices)
    args.show_visibility_edges = _prompt_yes_no(
        "Mostrar arestas do grafo de visibilidade",
        args.show_visibility_edges,
    )

    output_value = input(
        "Arquivo PNG de saida (Enter para caminho automatico): "
    ).strip()
    args.output = Path(output_value) if output_value else None

    return args


def _render_planned_path(
    map_grid: MapGrid,
    vis_graph: nx.Graph[GridPoint],
    planning_result: OrderedPathPlanningResult,
    source_name: str,
    source_grid_point: GridPoint,
    target_grid_point: GridPoint,
    output_path: Path,
    dpi: int,
    show_grid: bool,
    show_vertices: bool,
    show_visibility_edges: bool,
) -> None:
    """Renders occupancy grid + visibility graph + highlighted planned path."""
    try:
        plt = cast(Any, import_module("matplotlib.pyplot"))
        colors_module = cast(Any, import_module("matplotlib.colors"))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib nao encontrado. Instale com: pip install matplotlib"
        ) from exc

    listed_colormap = colors_module.ListedColormap

    figure_width = min(20.0, max(7.0, map_grid.cols / 7.0))
    figure_height = min(14.0, max(5.0, map_grid.rows / 7.0))

    fig, ax = cast(
        tuple[Any, Any],
        plt.subplots(figsize=(figure_width, figure_height), dpi=dpi),
    )

    ax.imshow(
        map_grid.grid,
        cmap=listed_colormap([FREE_SPACE_COLOR, OBSTACLE_COLOR]),
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
        zorder=0,
    )

    if show_visibility_edges:
        for source_vertex, target_vertex in vis_graph.edges:
            source_row, source_col = source_vertex
            target_row, target_col = target_vertex
            ax.plot(
                [source_col, target_col],
                [source_row, target_row],
                color=VISIBILITY_EDGE_COLOR,
                linewidth=0.7,
                alpha=0.22,
                zorder=1,
            )

    path = planning_result["path"]
    if len(path) >= 2:
        for path_index in range(len(path) - 1):
            source_point = path[path_index]
            target_point = path[path_index + 1]
            ax.plot(
                [source_point[1], target_point[1]],
                [source_point[0], target_point[0]],
                color=PATH_COLOR,
                linewidth=2.6,
                alpha=0.95,
                zorder=4,
            )

        path_rows = [point[0] for point in path]
        path_cols = [point[1] for point in path]
        ax.scatter(
            path_cols,
            path_rows,
            s=28,
            color=PATH_COLOR,
            edgecolors="#FFFFFF",
            linewidths=0.5,
            zorder=5,
        )

    source_point = source_grid_point
    target_point = target_grid_point

    ax.scatter(
        [source_point[1]],
        [source_point[0]],
        s=90,
        color=START_COLOR,
        edgecolors="#FFFFFF",
        linewidths=1.0,
        label="source",
        zorder=6,
    )
    ax.scatter(
        [target_point[1]],
        [target_point[0]],
        s=90,
        color=GOAL_COLOR,
        edgecolors="#FFFFFF",
        linewidths=1.0,
        label="target",
        zorder=6,
    )

    if show_vertices and vis_graph.number_of_nodes() > 0:
        node_rows = [node[0] for node in vis_graph.nodes]
        node_cols = [node[1] for node in vis_graph.nodes]
        ax.scatter(
            node_cols,
            node_rows,
            s=16,
            color=VERTEX_COLOR,
            edgecolors="#FFFFFF",
            linewidths=0.5,
            alpha=0.9,
            zorder=3,
        )

    if show_grid and map_grid.rows <= 150 and map_grid.cols <= 220:
        ax.set_xticks([value - 0.5 for value in range(map_grid.cols + 1)], minor=True)
        ax.set_yticks([value - 0.5 for value in range(map_grid.rows + 1)], minor=True)
        ax.grid(which="minor", color="#D6CFBF", linewidth=0.3)

    feasible = planning_result["path_cost"] != INFINITE_PATH_COST
    status = "feasible" if feasible else "infeasible"

    ax.set_title(
        f"Planned Path - {source_name} | status={status} "
        f"| cells={planning_result['traversed_cells']} "
        f"| robots={planning_result['robots_used']}",
        fontsize=12,
        color="#1F2321",
    )
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    if map_grid.cols > 90:
        ax.set_xticks([])
    if map_grid.rows > 90:
        ax.set_yticks([])

    cost_text = (
        f"{planning_result['path_cost']:.3f}"
        if feasible
        else "inf"
    )
    metrics_lines = [
        f"source (grid): {_format_grid_point(source_point)}",
        f"target (grid): {_format_grid_point(target_point)}",
    ]
    metrics_lines.extend(
        [
            f"path nodes: {len(path)}",
            f"cells traversed: {planning_result['traversed_cells']}",
            f"robots used: {planning_result['robots_used']}",
            f"cost: {cost_text}",
        ]
    )
    metrics_text = "\n".join(metrics_lines)

    fig.subplots_adjust(left=0.08, right=0.74, top=0.92, bottom=0.10)

    fig.text(
        0.76,
        0.90,
        metrics_text,
        ha="left",
        va="top",
        fontsize=9,
        color="#1F2321",
        bbox={"facecolor": "#F8F5ED", "alpha": 0.9, "edgecolor": "#D6CFBF"},
    )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.76, 0.36),
        framealpha=0.9,
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parses CLI options."""
    parser = argparse.ArgumentParser(
        description=(
            "Planeja caminho no grafo de visibilidade e exporta o mapa com "
            "caminho destacado e metricas."
        )
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Abre um assistente no terminal para escolher mapa e parametros.",
    )
    parser.add_argument(
        "--map-name",
        default=DEFAULT_MAP_NAME,
        help=(
            "Nome do mapa no catalogo JSON. "
            "Defina como vazio para usar --preset-module. "
            f"Padrao: {DEFAULT_MAP_NAME}"
        ),
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help=(
            "Caminho para o catalogo JSON de mapas. "
            f"Padrao: {DEFAULT_CATALOG_PATH}"
        ),
    )
    parser.add_argument(
        "--preset-module",
        default=None,
        help="Modulo do preset Python (modo legado), ex: presets.map1",
    )
    parser.add_argument(
        "--factory",
        default=DEFAULT_FACTORY_NAME,
        help=f"Factory no modulo Python legado. Padrao: {DEFAULT_FACTORY_NAME}",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=DEFAULT_SCALE_FACTOR,
        help=f"Scale factor para construcao do mapa. Padrao: {DEFAULT_SCALE_FACTOR}",
    )
    parser.add_argument(
        "--vertex-source",
        choices=["boundary", "processor"],
        default=DEFAULT_VERTEX_SOURCE,
        help=(
            "Estrategia de vertices do grafo: "
            "boundary (mais linhas) ou processor (pipeline do artigo). "
            f"Padrao: {DEFAULT_VERTEX_SOURCE}"
        ),
    )
    parser.add_argument(
        "--boundary-stride",
        type=int,
        default=DEFAULT_BOUNDARY_STRIDE,
        help=(
            "Subamostragem de vertices de fronteira (somente vertex-source=boundary). "
            f"Padrao: {DEFAULT_BOUNDARY_STRIDE}"
        ),
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=DEFAULT_MAX_VERTICES,
        help=(
            "Limite maximo de vertices (somente vertex-source=boundary). "
            "Use 0 para sem limite. "
            f"Padrao: {DEFAULT_MAX_VERTICES}"
        ),
    )
    parser.add_argument(
        "--source-point",
        default=None,
        help="Ponto de partida no formato row,col (qualquer celula livre).",
    )
    parser.add_argument(
        "--target-point",
        default=None,
        help="Ponto de destino no formato row,col (qualquer celula livre).",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=DEFAULT_RELAY_PENALTY_LAMBDA,
        help=(
            "Penalidade de rele lambda usada no caminho (Dijkstra com penalidade). "
            f"Padrao: {DEFAULT_RELAY_PENALTY_LAMBDA}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG de saida. Se omitido, usa ./exports/<source>_planned_path.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Resolucao da imagem exportada. Padrao: {DEFAULT_DPI}",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Exibe grade fina entre celulas para mapas pequenos/medios.",
    )
    parser.add_argument(
        "--show-vertices",
        action="store_true",
        help="Desenha os vertices extraidos no grafo.",
    )
    parser.add_argument(
        "--hide-visibility-edges",
        action="store_false",
        dest="show_visibility_edges",
        help="Oculta as arestas do grafo de visibilidade no fundo da imagem.",
    )
    parser.set_defaults(show_visibility_edges=True)

    return parser.parse_args()


def main() -> None:
    """Entrypoint for interactive path planning and export."""
    args = parse_args()

    if args.interactive or len(sys.argv) == 1:
        args = _collect_interactive_inputs(args)

    if args.scale_factor <= 0:
        raise ValueError("--scale-factor deve ser inteiro positivo.")
    if args.dpi <= 0:
        raise ValueError("--dpi deve ser inteiro positivo.")
    if args.boundary_stride <= 0:
        raise ValueError("--boundary-stride deve ser inteiro positivo.")
    if args.lam < 0.0:
        raise ValueError("--lam deve ser maior ou igual a zero.")

    max_vertices: int | None
    if args.max_vertices <= 0:
        max_vertices = None
    else:
        max_vertices = args.max_vertices

    map_name: str | None = args.map_name.strip() if args.map_name is not None else None
    if map_name == "":
        map_name = None

    map_grid, source_name = _load_map(
        map_name=map_name,
        catalog_path=args.catalog_path,
        preset_module=args.preset_module,
        factory_name=args.factory,
        scale_factor=args.scale_factor,
    )

    vis_graph = _build_visibility_graph_with_source(
        map_grid=map_grid,
        vertex_source=args.vertex_source,
        boundary_stride=args.boundary_stride,
        max_vertices=max_vertices,
    )

    if vis_graph.number_of_nodes() == 0:
        raise ValueError(
            "Nao foi possivel construir vertices no grafo de visibilidade para este mapa."
        )
    if vis_graph.number_of_nodes() == 1:
        raise ValueError(
            "O grafo de visibilidade possui apenas um vertice; "
            "nao ha caminho entre origem e destino distintos."
        )

    sorted_nodes = sorted(vis_graph.nodes)

    if args.interactive or args.source_point is None or args.target_point is None:
        _print_vertex_preview(sorted_nodes)
        default_source = sorted_nodes[0]
        default_target = sorted_nodes[-1]
        source_grid_point = _prompt_endpoint(
            "Origem",
            map_grid,
            sorted_nodes,
            default_point=default_source,
        )
        target_grid_point = _prompt_endpoint(
            "Destino",
            map_grid,
            sorted_nodes,
            default_point=default_target,
            blocked_grid_point=source_grid_point,
        )
    else:
        source_grid_point = _resolve_cli_endpoint(
            args.source_point,
            map_grid,
            "source-point",
        )
        target_grid_point = _resolve_cli_endpoint(
            args.target_point,
            map_grid,
            "target-point",
        )

    if source_grid_point == target_grid_point:
        raise ValueError("Origem e destino devem ser diferentes no grid.")

    if source_grid_point not in vis_graph:
        print(
            f"Origem {_format_grid_point(source_grid_point)} nao e vertice do grafo; "
            "sera conectada por LOS durante o planejamento."
        )
    if target_grid_point not in vis_graph:
        print(
            f"Destino {_format_grid_point(target_grid_point)} nao e vertice do grafo; "
            "sera conectado por LOS durante o planejamento."
        )

    planning_graph, source_neighbor_count, target_neighbor_count = (
        _build_planning_graph_with_endpoints(
            map_grid=map_grid,
            vis_graph=vis_graph,
            source_point=source_grid_point,
            target_point=target_grid_point,
        )
    )

    print(
        "Conexoes LOS da origem ao grafo: "
        f"{source_neighbor_count}"
    )
    print(
        "Conexoes LOS do destino ao grafo: "
        f"{target_neighbor_count}"
    )

    planning_result = plan_ordered_progression_on_visibility_graph(
        vis_graph=planning_graph,
        source=source_grid_point,
        target=target_grid_point,
        lam=args.lam,
        grid_obj=map_grid,
    )

    output_path = args.output if args.output is not None else _default_output_path(source_name)

    _render_planned_path(
        map_grid=map_grid,
        vis_graph=vis_graph,
        planning_result=planning_result,
        source_name=source_name,
        source_grid_point=source_grid_point,
        target_grid_point=target_grid_point,
        output_path=output_path,
        dpi=args.dpi,
        show_grid=args.show_grid,
        show_vertices=args.show_vertices,
        show_visibility_edges=args.show_visibility_edges,
    )

    print(f"Planejamento exportado em: {output_path}")
    print(f"Mapa: {map_grid.rows}x{map_grid.cols}")
    print(f"Estrategia de vertices: {args.vertex_source}")
    print(f"Vertices: {vis_graph.number_of_nodes()}")
    print(f"Arestas LOS: {vis_graph.number_of_edges()}")
    print(f"Origem (grid): {_format_grid_point(source_grid_point)}")
    print(f"Destino (grid): {_format_grid_point(target_grid_point)}")

    if planning_result["path_cost"] == INFINITE_PATH_COST:
        print("Nao foi encontrado caminho viavel entre origem e destino.")
    else:
        print(f"Custo do caminho: {planning_result['path_cost']:.3f}")
        print(f"Nos do caminho: {len(planning_result['path'])}")
        print(f"Celulas percorridas: {planning_result['traversed_cells']}")
        print(f"Robos utilizados: {planning_result['robots_used']}")
        print(f"Movimentos (snapshots): {len(planning_result['movement_snapshots'])}")


if __name__ == "__main__":
    main()
