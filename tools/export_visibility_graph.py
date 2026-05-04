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
from pathlib import Path
import sys
from typing import Any, cast

import networkx as nx

try:
    from tools._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from core.map_grid import MapGrid
from presets.map_catalog import DEFAULT_CATALOG_PATH
from tools.common import (
    DEFAULT_FACTORY_NAME,
    build_visibility_graph_with_source,
    coerce_max_vertices,
    default_output_path,
    load_matplotlib_modules,
    load_map,
    normalize_optional_map_name,
    prompt_positive_int,
    prompt_yes_no,
    select_catalog_or_preset,
)


DEFAULT_MAP_NAME = "map1"
DEFAULT_SCALE_FACTOR = 1
DEFAULT_DPI = 220
DEFAULT_VERTEX_SOURCE = "boundary"
DEFAULT_BOUNDARY_STRIDE = 1
DEFAULT_MAX_VERTICES = 600

FREE_SPACE_COLOR = "#F4F1E8"
OBSTACLE_COLOR = "#1F2321"
EDGE_COLOR = "#C0681B"
VERTEX_COLOR = "#1F4D3A"


def _default_output_path(source_name: str) -> Path:
    """Builds default output PNG path from map source name."""
    return default_output_path(source_name, "visibility_graph")


def _build_visibility_graph_with_source(
    map_grid: MapGrid,
    vertex_source: str,
    boundary_stride: int,
    max_vertices: int | None,
) -> nx.Graph[tuple[int, int]]:
    """Builds visibility graph from selected vertex source strategy."""
    return build_visibility_graph_with_source(
        map_grid=map_grid,
        vertex_source=vertex_source,
        boundary_stride=boundary_stride,
        max_vertices=max_vertices,
        sampling_mode="ceil",
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
    plt, colors_module = load_matplotlib_modules()

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


def _collect_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Collects interactive options to avoid long command lines."""
    print("\n=== Exportador de Grafo de Visibilidade ===")

    select_catalog_or_preset(args)

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
        args.boundary_stride = prompt_positive_int(
            "Boundary stride",
            args.boundary_stride,
        )
        args.max_vertices = prompt_positive_int(
            "Max vertices (0 para sem limite)",
            args.max_vertices,
            allow_zero=True,
        )

    args.scale_factor = prompt_positive_int("Scale factor", args.scale_factor)
    args.dpi = prompt_positive_int("DPI", args.dpi)
    args.show_grid = prompt_yes_no("Mostrar grade", args.show_grid)
    args.show_vertices = prompt_yes_no("Mostrar vertices", args.show_vertices)

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

    max_vertices = coerce_max_vertices(args.max_vertices)

    map_name = normalize_optional_map_name(args.map_name)

    map_grid, source_name = load_map(
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
