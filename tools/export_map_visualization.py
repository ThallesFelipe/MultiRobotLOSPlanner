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
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Protocol, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.map_grid import MapGrid
from presets.map_catalog import (
    DEFAULT_CATALOG_PATH,
    create_map_from_catalog,
    list_catalog_maps,
)


class MapFactory(Protocol):
    """Callable signature for preset map factory functions."""

    def __call__(self, scale_factor: int = 1) -> MapGrid:
        ...


DEFAULT_MAP_NAME = "map1"
DEFAULT_PRESET_MODULE = "presets.map1"
DEFAULT_FACTORY_NAME = "create_custom_map"
DEFAULT_SCALE_FACTOR = 1
DEFAULT_DPI = 180

FREE_SPACE_COLOR = "#FFFFFF"
OBSTACLE_COLOR = "#000000"


def _load_factory(module_name: str, factory_name: str) -> MapFactory:
    """Loads a map factory function from a preset module."""
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
    """Builds a default output file path from map source name."""
    safe_name = source_name.replace(":", "_").replace(".", "_")
    return Path("exports") / f"{safe_name}_preview.png"


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

    factory = _load_factory(preset_module, factory_name)
    map_grid = factory(scale_factor=scale_factor)
    return (map_grid, f"{preset_module}_{factory_name}")


def _render_png(
    map_grid: MapGrid,
    output_path: Path,
    source_name: str,
    scale_factor: int,
    dpi: int,
    show_grid: bool,
) -> None:
    """Renders an occupancy grid and writes it as a PNG image."""
    try:
        plt = cast(Any, import_module("matplotlib.pyplot"))
        colors_module = cast(Any, import_module("matplotlib.colors"))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib nao encontrado. Instale com: pip install matplotlib"
        ) from exc

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


def _prompt_positive_int(label: str, default: int) -> int:
    """Reads a positive integer option from terminal input."""
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default

        try:
            parsed_value = int(raw_value)
        except ValueError:
            print("Valor invalido. Digite um numero inteiro.")
            continue

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
    print("\n=== Exportador de Occupancy Grid ===")

    available_maps = list_catalog_maps(args.catalog_path)
    if available_maps:
        args.map_name = _prompt_map_name(args.catalog_path, args.map_name)
        args.preset_module = None
    else:
        print("Catalogo sem mapas. Usando modo legado por modulo Python.")
        args.map_name = None
        default_module = args.preset_module or DEFAULT_PRESET_MODULE
        preset_module = input(
            f"Modulo preset Python [{default_module}]: "
        ).strip()
        args.preset_module = preset_module if preset_module else default_module

    args.scale_factor = _prompt_positive_int("Scale factor", args.scale_factor)
    args.dpi = _prompt_positive_int("DPI", args.dpi)
    args.show_grid = _prompt_yes_no("Mostrar grade", args.show_grid)

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

    output_path = args.output if args.output is not None else _default_output_path(source_name)

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
