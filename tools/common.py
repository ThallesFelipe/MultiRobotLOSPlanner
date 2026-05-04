"""Shared helpers for command-line map tools."""

from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Protocol, cast

import networkx as nx

try:
    from tools._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from core.map_grid import GridPoint, MapGrid
from core.map_processor import MapProcessor
from core.visibility_graph import build_visibility_graph
from presets.map_catalog import create_map_from_catalog, list_catalog_maps

DEFAULT_LEGACY_PRESET_MODULE = "presets.map1"
DEFAULT_FACTORY_NAME = "create_custom_map"

BoundarySamplingMode = Literal["ceil", "floor"]


class MapFactory(Protocol):
    """Callable signature for Python preset map factory functions."""

    def __call__(self, scale_factor: int = 1) -> MapGrid:
        ...


def load_python_factory(module_name: str, factory_name: str) -> MapFactory:
    """Loads a map factory from a Python preset module."""
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


def load_matplotlib_modules() -> tuple[Any, Any]:
    """Loads Matplotlib modules with the non-interactive PNG backend."""
    try:
        matplotlib_module = import_module("matplotlib")
        matplotlib_module.use("Agg", force=True)
        pyplot_module = import_module("matplotlib.pyplot")
        colors_module = import_module("matplotlib.colors")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib nao encontrado. Instale com: pip install matplotlib"
        ) from exc

    return pyplot_module, colors_module


def default_output_path(source_name: str, suffix: str) -> Path:
    """Builds a default export path from a map source name and artifact suffix."""
    safe_name = source_name.replace(":", "_").replace(".", "_")
    return Path("exports") / f"{safe_name}_{suffix}.png"


def normalize_optional_map_name(map_name: str | None) -> str | None:
    """Normalizes optional CLI map-name text into either a name or `None`."""
    if map_name is None:
        return None

    normalized = map_name.strip()
    return normalized or None


def load_map(
    map_name: str | None,
    catalog_path: Path,
    preset_module: str | None,
    factory_name: str,
    scale_factor: int,
) -> tuple[MapGrid, str]:
    """Loads a map from JSON catalog or from a legacy Python preset module."""
    if map_name is not None:
        map_grid = create_map_from_catalog(
            map_name=map_name,
            scale_factor=scale_factor,
            catalog_path=catalog_path,
        )
        return (map_grid, f"catalog_{map_name}")

    if preset_module is None:
        raise ValueError("Informe --map-name ou --preset-module.")

    factory = load_python_factory(preset_module, factory_name)
    map_grid = factory(scale_factor=scale_factor)
    return (map_grid, f"{preset_module}_{factory_name}")


def prompt_positive_int(label: str, default: int, *, allow_zero: bool = False) -> int:
    """Reads an integer option from terminal input."""
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


def prompt_non_negative_float(label: str, default: float) -> float:
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


def prompt_yes_no(label: str, default: bool) -> bool:
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


def prompt_map_name(catalog_path: Path, default_map_name: str | None) -> str:
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


def select_catalog_or_preset(
    args: object,
    *,
    default_preset_module: str = DEFAULT_LEGACY_PRESET_MODULE,
) -> None:
    """Mutates an argparse namespace with the selected catalog or preset source."""
    catalog_path = cast(Path, getattr(args, "catalog_path"))
    map_name = cast(str | None, getattr(args, "map_name"))
    available_maps = list_catalog_maps(catalog_path)
    if available_maps:
        setattr(args, "map_name", prompt_map_name(catalog_path, map_name))
        setattr(args, "preset_module", None)
        return

    print("Catalogo sem mapas. Usando modo legado por modulo Python.")
    setattr(args, "map_name", None)
    current_module = cast(str | None, getattr(args, "preset_module"))
    default_module = current_module or default_preset_module
    preset_module = input(f"Modulo preset Python [{default_module}]: ").strip()
    setattr(args, "preset_module", preset_module if preset_module else default_module)


def _has_obstacle_neighbor(map_grid: MapGrid, row: int, col: int) -> bool:
    """Returns whether one free cell touches any obstacle in its 8-neighborhood."""
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue

            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            if not map_grid.in_bounds(neighbor_row, neighbor_col):
                continue
            if not map_grid.is_free(neighbor_row, neighbor_col):
                return True

    return False


def _sampling_step(
    vertex_count: int,
    max_vertices: int,
    sampling_mode: BoundarySamplingMode,
) -> int:
    """Returns the historical sampling step used by each tool."""
    if sampling_mode == "ceil":
        return int(math.ceil(vertex_count / max_vertices))
    if sampling_mode == "floor":
        return max(1, vertex_count // max_vertices)
    raise ValueError("sampling_mode must be either 'ceil' or 'floor'.")


def extract_boundary_vertices(
    map_grid: MapGrid,
    boundary_stride: int,
    max_vertices: int | None,
    *,
    sampling_mode: BoundarySamplingMode = "ceil",
) -> list[GridPoint]:
    """Extracts free-space boundary vertices with optional stride and cap."""
    if boundary_stride <= 0:
        raise ValueError("boundary_stride must be greater than 0.")

    vertices: list[GridPoint] = []
    for row in range(map_grid.rows):
        for col in range(map_grid.cols):
            if not map_grid.is_free(row, col):
                continue
            if ((row + col) % boundary_stride) != 0:
                continue
            if _has_obstacle_neighbor(map_grid, row, col):
                vertices.append((row, col))

    if max_vertices is not None and len(vertices) > max_vertices:
        sampling_step = _sampling_step(len(vertices), max_vertices, sampling_mode)
        vertices = vertices[::sampling_step]

    return vertices


def build_visibility_graph_with_source(
    map_grid: MapGrid,
    vertex_source: str,
    boundary_stride: int,
    max_vertices: int | None,
    *,
    sampling_mode: BoundarySamplingMode = "ceil",
) -> nx.Graph[GridPoint]:
    """Builds a visibility graph from the selected vertex source strategy."""
    if vertex_source == "processor":
        processor = MapProcessor(map_grid)
        return processor.build_initial_visibility_graph()

    if vertex_source == "boundary":
        vertices = extract_boundary_vertices(
            map_grid=map_grid,
            boundary_stride=boundary_stride,
            max_vertices=max_vertices,
            sampling_mode=sampling_mode,
        )
        return build_visibility_graph(map_grid, vertices)

    raise ValueError(
        f"Unknown vertex_source={vertex_source!r}. Use 'processor' or 'boundary'."
    )


def coerce_max_vertices(raw_value: int) -> int | None:
    """Converts CLI max-vertices semantics where 0 means no limit."""
    return None if raw_value <= 0 else raw_value


__all__ = [
    "DEFAULT_FACTORY_NAME",
    "DEFAULT_LEGACY_PRESET_MODULE",
    "MapFactory",
    "build_visibility_graph_with_source",
    "coerce_max_vertices",
    "default_output_path",
    "extract_boundary_vertices",
    "load_matplotlib_modules",
    "load_map",
    "load_python_factory",
    "normalize_optional_map_name",
    "prompt_map_name",
    "prompt_non_negative_float",
    "prompt_positive_int",
    "prompt_yes_no",
    "select_catalog_or_preset",
]
