"""Builds and exports visibility graph visualization for occupancy-grid maps.

Map source options:
- JSON catalog entry (recommended): `presets/maps_catalog.json`
- Python preset factory (legacy): `presets.<module>:<factory>`

Examples:
    python tools/export_visibility_graph.py
    python tools/export_visibility_graph.py --interactive
    python tools/export_visibility_graph.py --map-name map1 --show-grid --show-vertices
    python tools/export_visibility_graph.py --preset-module presets.map1 --factory create_custom_map
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

from core.map_grid import MapGrid
from core.map_processor import MapProcessor
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
EDGE_COLOR = "#C0681B"
VERTEX_COLOR = "#1F4D3A"


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
    return Path("exports") / f"{safe_name}_visibility_graph.png"


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
) -> list[tuple[int, int]]:
    """Extracts free-space boundary vertices with optional stride and cap.

    Boundary cells are free cells with at least one 8-neighbor obstacle cell.
    """
    if boundary_stride <= 0:
        raise ValueError("boundary_stride must be greater than 0.")

    rows = map_grid.rows
    cols = map_grid.cols
    vertices: list[tuple[int, int]] = []

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
) -> nx.Graph[tuple[int, int]]:
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


def _render_visibility_graph(
    map_grid: MapGrid,
    vis_graph: nx.Graph[tuple[int, int]],
    output_path: Path,
    title: str,
    dpi: int,
    show_grid: bool,
    show_vertices: bool,
) -> None:
    """Renders occupancy grid + all visibility graph edges to PNG."""
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

    for source_vertex, target_vertex in vis_graph.edges:
        source_row, source_col = source_vertex
        target_row, target_col = target_vertex
        ax.plot(
            [source_col, target_col],
            [source_row, target_row],
            color=EDGE_COLOR,
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
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
            zorder=3,
        )

    if show_grid and map_grid.rows <= 150 and map_grid.cols <= 220:
        ax.set_xticks([value - 0.5 for value in range(map_grid.cols + 1)], minor=True)
        ax.set_yticks([value - 0.5 for value in range(map_grid.rows + 1)], minor=True)
        ax.grid(which="minor", color="#D6CFBF", linewidth=0.3)

    ax.set_title(title, fontsize=12, color="#1F2321")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    if map_grid.cols > 90:
        ax.set_xticks([])
    if map_grid.rows > 90:
        ax.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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


def _collect_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Collects interactive options to avoid long command lines."""
    print("\n=== Exportador de Grafo de Visibilidade ===")

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
    args.dpi = _prompt_positive_int("DPI", args.dpi)
    args.show_grid = _prompt_yes_no("Mostrar grade", args.show_grid)
    args.show_vertices = _prompt_yes_no("Mostrar vertices", args.show_vertices)

    output_value = input(
        "Arquivo PNG de saida (Enter para caminho automatico): "
    ).strip()
    args.output = Path(output_value) if output_value else None

    return args


def parse_args() -> argparse.Namespace:
    """Parses CLI options."""
    parser = argparse.ArgumentParser(
        description="Gera e exporta visualizacao do grafo de visibilidade."
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
        "--output",
        type=Path,
        default=None,
        help="PNG de saida. Se omitido, usa ./exports/<source>_visibility_graph.png",
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
    return parser.parse_args()


def main() -> None:
    """Entrypoint for visibility graph generation and export."""
    args = parse_args()

    if args.interactive or len(sys.argv) == 1:
        args = _collect_interactive_inputs(args)

    if args.scale_factor <= 0:
        raise ValueError("--scale-factor deve ser inteiro positivo.")
    if args.dpi <= 0:
        raise ValueError("--dpi deve ser inteiro positivo.")
    if args.boundary_stride <= 0:
        raise ValueError("--boundary-stride deve ser inteiro positivo.")

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

    output_path = args.output if args.output is not None else _default_output_path(source_name)
    title = (
        f"Visibility Graph - {source_name} "
        f"| source={args.vertex_source} "
        f"| nodes={vis_graph.number_of_nodes()} edges={vis_graph.number_of_edges()}"
    )
    _render_visibility_graph(
        map_grid=map_grid,
        vis_graph=vis_graph,
        output_path=output_path,
        title=title,
        dpi=args.dpi,
        show_grid=args.show_grid,
        show_vertices=args.show_vertices,
    )

    print(f"Visibilidade exportada em: {output_path}")
    print(f"Mapa: {map_grid.rows}x{map_grid.cols}")
    print(f"Estrategia de vertices: {args.vertex_source}")
    print(f"Vertices: {vis_graph.number_of_nodes()}")
    print(f"Arestas LOS: {vis_graph.number_of_edges()}")


if __name__ == "__main__":
    main()
