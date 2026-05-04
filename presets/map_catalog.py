"""JSON catalog storage for occupancy-grid map presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Sequence, TypedDict, cast

from core.map_grid import MapGrid

GridRectangle = tuple[int, int, int, int]

CATALOG_VERSION: Final[int] = 1
DEFAULT_CATALOG_PATH: Final[Path] = Path(__file__).with_name("maps_catalog.json")


class CatalogData(TypedDict):
    """Normalized JSON catalog structure."""

    version: int
    maps: dict[str, object]


def _validate_positive_int(name: str, value: object) -> int:
    """Validates that a value is a positive integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")
    return value


def _validate_scale_factor(scale_factor: object) -> int:
    """Validates and returns a positive integer map scale factor."""
    return _validate_positive_int("scale_factor", scale_factor)


def _validate_rectangle(
    rectangle: Sequence[object],
    rows: int,
    cols: int,
) -> GridRectangle:
    """Validates one rectangle in `[row_start, row_end, col_start, col_end]` format."""
    if len(rectangle) != 4:
        raise ValueError(
            "Each obstacle rectangle must contain exactly four integers: "
            "[row_start, row_end, col_start, col_end]."
        )

    row_start = _validate_non_negative_int("row_start", rectangle[0])
    row_end = _validate_non_negative_int("row_end", rectangle[1])
    col_start = _validate_non_negative_int("col_start", rectangle[2])
    col_end = _validate_non_negative_int("col_end", rectangle[3])

    if not (0 <= row_start < row_end <= rows and 0 <= col_start < col_end <= cols):
        raise ValueError(
            "Invalid rectangle bounds. Expected [start, end) ranges inside map limits."
        )

    return (row_start, row_end, col_start, col_end)


def _validate_non_negative_int(name: str, value: object) -> int:
    """Validates that a value is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; received {value!r}.")
    return value


def _normalize_map_name(map_name: str) -> str:
    """Normalizes and validates a map name used as catalog key."""
    normalized = map_name.strip()
    if not normalized:
        raise ValueError("map_name cannot be empty.")
    return normalized


def _read_catalog(catalog_path: Path | str = DEFAULT_CATALOG_PATH) -> CatalogData:
    """Reads catalog JSON and returns a normalized in-memory structure."""
    path = Path(catalog_path)
    if not path.exists():
        return {"version": CATALOG_VERSION, "maps": {}}

    raw_object: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_object, dict):
        raise ValueError("Catalog root must be a JSON object.")
    raw = cast(dict[str, object], raw_object)

    maps_object = raw.get("maps", {})
    if not isinstance(maps_object, dict):
        raise ValueError("Catalog field 'maps' must be a JSON object.")
    maps_untyped = cast(dict[object, object], maps_object)

    maps: dict[str, object] = {}
    for map_name, map_entry in maps_untyped.items():
        if not isinstance(map_name, str):
            raise ValueError("Catalog map names must be strings.")
        maps[map_name] = map_entry

    version_object = raw.get("version", CATALOG_VERSION)
    if not isinstance(version_object, int):
        raise ValueError("Catalog field 'version' must be an integer.")

    version = int(version_object)
    return {"version": version, "maps": maps}


def _write_catalog(
    data: CatalogData,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> None:
    """Writes catalog JSON to disk using stable formatting."""
    path = Path(catalog_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_map_entry(
    map_name: str,
    entry: object,
) -> tuple[int, int, list[GridRectangle]]:
    """Parses and validates one map entry from catalog data."""
    if not isinstance(entry, dict):
        raise ValueError(f"Map entry {map_name!r} must be a JSON object.")
    entry_dict = cast(dict[str, object], entry)

    rows = _validate_positive_int("rows", entry_dict.get("rows"))
    cols = _validate_positive_int("cols", entry_dict.get("cols"))

    rectangles_object = entry_dict.get("obstacle_rectangles", [])
    if not isinstance(rectangles_object, list):
        raise ValueError(
            f"Map entry {map_name!r} field 'obstacle_rectangles' must be a list."
        )
    rectangles_raw = cast(list[object], rectangles_object)

    rectangles: list[GridRectangle] = []
    for rectangle_raw in rectangles_raw:
        if not isinstance(rectangle_raw, list | tuple):
            raise ValueError(
                f"Map entry {map_name!r} contains a rectangle with invalid type."
            )
        rectangles.append(
            _validate_rectangle(
                cast(Sequence[object], rectangle_raw),
                rows=rows,
                cols=cols,
            )
        )

    return (rows, cols, rectangles)


def list_catalog_maps(catalog_path: Path | str = DEFAULT_CATALOG_PATH) -> list[str]:
    """Lists map names available in the JSON catalog."""
    data = _read_catalog(catalog_path)
    maps = data["maps"]
    return sorted(maps.keys())


def upsert_catalog_map(
    map_name: str,
    rows: int,
    cols: int,
    obstacle_rectangles: Sequence[GridRectangle],
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> None:
    """Creates or updates one map entry in the JSON catalog."""
    normalized_map_name = _normalize_map_name(map_name)
    valid_rows = _validate_positive_int("rows", rows)
    valid_cols = _validate_positive_int("cols", cols)

    validated_rectangles = [
        _validate_rectangle(rectangle, rows=valid_rows, cols=valid_cols)
        for rectangle in obstacle_rectangles
    ]

    data = _read_catalog(catalog_path)
    maps = data["maps"]
    maps[normalized_map_name] = {
        "rows": valid_rows,
        "cols": valid_cols,
        "obstacle_rectangles": [list(rectangle) for rectangle in validated_rectangles],
    }
    data["version"] = CATALOG_VERSION

    _write_catalog(data, catalog_path)


def create_map_from_catalog(
    map_name: str,
    scale_factor: int = 1,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> MapGrid:
    """Builds `MapGrid` from a named map entry in the JSON catalog."""
    normalized_map_name = _normalize_map_name(map_name)
    valid_scale_factor = _validate_scale_factor(scale_factor)

    data = _read_catalog(catalog_path)
    maps = data["maps"]

    if normalized_map_name not in maps:
        available = ", ".join(list_catalog_maps(catalog_path))
        raise ValueError(
            f"Map {normalized_map_name!r} not found in catalog. "
            f"Available maps: [{available}]"
        )

    rows, cols, rectangles = _parse_map_entry(
        normalized_map_name,
        maps[normalized_map_name],
    )

    map_grid = MapGrid(
        rows=rows * valid_scale_factor,
        cols=cols * valid_scale_factor,
    )

    for rectangle in rectangles:
        row_start, row_end, col_start, col_end = rectangle
        map_grid.add_obstacle_rect(
            row_start * valid_scale_factor,
            row_end * valid_scale_factor,
            col_start * valid_scale_factor,
            col_end * valid_scale_factor,
        )

    return map_grid


__all__ = [
    "CATALOG_VERSION",
    "DEFAULT_CATALOG_PATH",
    "create_map_from_catalog",
    "list_catalog_maps",
    "upsert_catalog_map",
]
