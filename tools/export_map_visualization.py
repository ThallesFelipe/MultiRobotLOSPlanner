"""Exports a rendered image for occupancy-grid maps.

Map source options:
- JSON catalog entry (recommended): `presets/maps_catalog.json`
- Python preset factory (legacy): `presets.<module>:<factory>`

Usage examples:
    python tools/export_map_visualization.py
    python tools/export_map_visualization.py --interactive
    python tools/export_map_visualization.py --map-name map_custom --show-grid
    python tools/export_map_visualization.py --preset-module presets.map1 --factory create_custom_map
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, cast

try:
    from tools._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from core.map_grid import MapGrid
from presets.map_catalog import DEFAULT_CATALOG_PATH
from tools.common import (
    DEFAULT_FACTORY_NAME,
    DEFAULT_LEGACY_PRESET_MODULE,
    default_output_path,
    load_matplotlib_modules,
    load_map,
    normalize_optional_map_name,
    prompt_positive_int,
    prompt_yes_no,
    select_catalog_or_preset,
)


DEFAULT_MAP_NAME = "map1"
DEFAULT_PRESET_MODULE = DEFAULT_LEGACY_PRESET_MODULE
DEFAULT_SCALE_FACTOR = 1
DEFAULT_DPI = 180

FREE_SPACE_COLOR = "#FFFFFF"
OBSTACLE_COLOR = "#000000"


def _render_png(
    map_grid: MapGrid,
    output_path: Path,
    source_name: str,
    scale_factor: int,
    dpi: int,
    show_grid: bool,
) -> None:
    """Renders an occupancy grid and writes it as a PNG image."""
    plt, colors_module = load_matplotlib_modules()

    listed_colormap = colors_module.ListedColormap

    figure_width = min(18.0, max(6.0, map_grid.cols / 8.0))
    figure_height = min(12.0, max(4.0, map_grid.rows / 8.0))

    fig, ax = cast(
        tuple[Any, Any],
        plt.subplots(figsize=(figure_width, figure_height), dpi=dpi),
    )
    colormap = listed_colormap([FREE_SPACE_COLOR, OBSTACLE_COLOR])

    ax.imshow(
        map_grid.grid,
        cmap=colormap,
        interpolation="nearest",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    if show_grid and map_grid.rows <= 150 and map_grid.cols <= 200:
        ax.set_xticks([value - 0.5 for value in range(map_grid.cols + 1)], minor=True)
        ax.set_yticks([value - 0.5 for value in range(map_grid.rows + 1)], minor=True)
        ax.grid(which="minor", color="#D6CFBF", linewidth=0.35)

    ax.set_title(
        f"Occupancy Grid - {source_name} (scale_factor={scale_factor})",
        fontsize=12,
        color="#1F2321",
    )
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    if map_grid.cols > 80:
        ax.set_xticks([])
    if map_grid.rows > 80:
        ax.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _collect_interactive_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Collects interactive options to avoid long command lines."""
    print("\n=== Exportador de Occupancy Grid ===")

    select_catalog_or_preset(args, default_preset_module=DEFAULT_PRESET_MODULE)

    args.scale_factor = prompt_positive_int("Scale factor", args.scale_factor)
    args.dpi = prompt_positive_int("DPI", args.dpi)
    args.show_grid = prompt_yes_no("Mostrar grade", args.show_grid)

    output_value = input(
        "Arquivo PNG de saida (Enter para caminho automatico): "
    ).strip()
    args.output = Path(output_value) if output_value else None

    return args


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Exporta uma visualizacao PNG da occupancy grid."
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
        help=(
            "Modulo do preset Python (modo legado), "
            "ex: presets.map1, presets.paper_map"
        ),
    )
    parser.add_argument(
        "--factory",
        default=DEFAULT_FACTORY_NAME,
        help=(
            "Nome da funcao factory no modulo preset. "
            f"Padrao: {DEFAULT_FACTORY_NAME}"
        ),
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=DEFAULT_SCALE_FACTOR,
        help=f"Scale factor passado para factory. Padrao: {DEFAULT_SCALE_FACTOR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho de saida PNG. Se omitido, usa ./exports/<source>_preview.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Resolucao final da imagem. Padrao: {DEFAULT_DPI}",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Desenha grade fina entre celulas para mapas pequenos/medios.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for map visualization export."""
    args = parse_args()

    if args.interactive or len(sys.argv) == 1:
        args = _collect_interactive_inputs(args)

    if args.scale_factor <= 0:
        raise ValueError("--scale-factor deve ser inteiro positivo.")
    if args.dpi <= 0:
        raise ValueError("--dpi deve ser inteiro positivo.")

    map_name = normalize_optional_map_name(args.map_name)

    map_grid, source_name = load_map(
        map_name=map_name,
        catalog_path=args.catalog_path,
        preset_module=args.preset_module,
        factory_name=args.factory,
        scale_factor=args.scale_factor,
    )

    output_path = (
        args.output
        if args.output is not None
        else default_output_path(source_name, "preview")
    )

    _render_png(
        map_grid=map_grid,
        output_path=output_path,
        source_name=source_name,
        scale_factor=args.scale_factor,
        dpi=args.dpi,
        show_grid=args.show_grid,
    )

    print(f"Visualizacao exportada em: {output_path}")
    print(f"Fonte do mapa: {source_name}")
    print(f"Dimensao do mapa: {map_grid.rows}x{map_grid.cols}")


if __name__ == "__main__":
    main()
