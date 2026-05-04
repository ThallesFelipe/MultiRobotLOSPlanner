"""Interactive occupancy-grid level editor for MapGrid presets.

This tool provides a small GUI to draw occupancy maps with the mouse and
export them as Python presets that match the project's `MapGrid` API.

Features:
- Configurable map dimensions (`rows`, `cols`) on startup.
- Load existing maps from the shared JSON catalog for further editing.
- Left-click + drag paints obstacle cells (`C_obs`).
- Right-click + drag erases cells back to free space (`C_free`).
- Status bar displays current grid cursor coordinates.
- Export writes `./presets/<name>.py` using compact rectangle definitions
  (`row_start`, `row_end`, `col_start`, `col_end`) for `add_obstacle_rect`.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass
import re
import tkinter as tk
from tkinter import messagebox, simpledialog
from typing import Any, Callable, cast

try:
    from tools._bootstrap import PROJECT_ROOT, ensure_project_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _bootstrap import PROJECT_ROOT, ensure_project_root_on_path

ensure_project_root_on_path()

from core.map_grid import MapGrid
from presets.map_catalog import (
    DEFAULT_CATALOG_PATH,
    create_map_from_catalog,
    list_catalog_maps,
    upsert_catalog_map,
)


PALETTE: dict[str, str] = {
    "background": "#F4F1E8",
    "panel": "#EAE4D6",
    "text": "#1F2321",
    "accent": "#1F4D3A",
    "accent_text": "#F4F1E8",
    "cell_free": "#F8F5ED",
    "cell_obstacle": "#1E2F24",
    "grid_line": "#D6CFBF",
    "status": "#E2DBC9",
}

DEFAULT_ROWS = 80
DEFAULT_COLS = 200
DEFAULT_CELL_SIZE = 12
MIN_CELL_SIZE = 4
MAX_CELL_SIZE = 48

GridRectangle = tuple[int, int, int, int]


@dataclass(frozen=True)
class EditorConfig:
    """Current map configuration for canvas dimensions."""

    rows: int
    cols: int
    cell_size: int


class MapEditorApp(tk.Tk):
    """Tkinter GUI app for drawing occupancy-grid presets."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Map Editor - Occupancy Grid")
        self.configure(bg=PALETTE["background"])
        self.geometry("1280x820")
        self.minsize(900, 620)

        self.config_var_rows = tk.StringVar(value=str(DEFAULT_ROWS))
        self.config_var_cols = tk.StringVar(value=str(DEFAULT_COLS))
        self.config_var_cell_size = tk.StringVar(value=str(DEFAULT_CELL_SIZE))
        self.status_var = tk.StringVar(value="Pronto.")

        self.config_state = EditorConfig(
            rows=DEFAULT_ROWS,
            cols=DEFAULT_COLS,
            cell_size=DEFAULT_CELL_SIZE,
        )

        self.obstacles: set[tuple[int, int]] = set()
        self.obstacle_items: dict[tuple[int, int], int] = {}

        self._build_layout()
        self._bind_canvas_events()

        self.create_map_from_inputs()

    def _build_layout(self) -> None:
        """Builds the control panel, drawing canvas, and status bar."""
        top_panel = tk.Frame(self, bg=PALETTE["panel"], padx=10, pady=8)
        top_panel.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(
            top_panel,
            text="Rows:",
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            top_panel,
            width=6,
            textvariable=self.config_var_rows,
            bg=PALETTE["cell_free"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            relief=tk.SOLID,
            bd=1,
        ).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )

        tk.Label(
            top_panel,
            text="Cols:",
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            top_panel,
            width=6,
            textvariable=self.config_var_cols,
            bg=PALETTE["cell_free"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            relief=tk.SOLID,
            bd=1,
        ).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )

        tk.Label(
            top_panel,
            text="Cell Size:",
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
        ).pack(
            side=tk.LEFT,
            padx=(0, 4),
        )
        tk.Entry(
            top_panel,
            width=6,
            textvariable=self.config_var_cell_size,
            bg=PALETTE["cell_free"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            relief=tk.SOLID,
            bd=1,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self._create_button(top_panel, "Criar / Redefinir", self.create_map_from_inputs).pack(
            side=tk.LEFT,
            padx=(4, 8),
        )
        self._create_button(
            top_panel,
            "Carregar Catalogo JSON",
            self.load_catalog_map,
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._create_button(top_panel, "Limpar", self.clear_obstacles).pack(
            side=tk.LEFT,
            padx=(0, 8),
        )
        self._create_button(top_panel, "Exportar Preset", self.export_preset).pack(side=tk.LEFT)
        self._create_button(
            top_panel,
            "Exportar Catalogo JSON",
            self.export_catalog_map,
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(
            top_panel,
            text="LMB: pintar  |  RMB: apagar",
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
        ).pack(side=tk.RIGHT, padx=(12, 0))

        canvas_wrapper = tk.Frame(self, bg=PALETTE["background"])
        canvas_wrapper.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(
            canvas_wrapper,
            bg=PALETTE["cell_free"],
            highlightthickness=1,
            highlightbackground=PALETTE["grid_line"],
        )

        scroll_y = tk.Scrollbar(
            canvas_wrapper,
            orient=tk.VERTICAL,
            command=self._canvas_yview,
        )
        scroll_x = tk.Scrollbar(
            canvas_wrapper,
            orient=tk.HORIZONTAL,
            command=self._canvas_xview,
        )
        self.canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")

        canvas_wrapper.grid_rowconfigure(0, weight=1)
        canvas_wrapper.grid_columnconfigure(0, weight=1)

        status_bar = tk.Label(
            self,
            textvariable=self.status_var,
            anchor=tk.W,
            bg=PALETTE["status"],
            fg=PALETTE["text"],
            padx=10,
            pady=6,
        )
        status_bar.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _create_button(
        self,
        parent: tk.Widget,
        text: str,
        command: Callable[[], Any],
    ) -> tk.Button:
        """Creates a palette-consistent button."""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=PALETTE["accent"],
            fg=PALETTE["accent_text"],
            activebackground="#2A634D",
            activeforeground=PALETTE["accent_text"],
            relief=tk.FLAT,
            padx=10,
            pady=6,
            cursor="hand2",
        )

    def _canvas_yview(self, *args: object) -> None:
        """Scrollbar callback proxy for vertical canvas scrolling."""
        canvas = cast(Any, self.canvas)
        canvas.yview(*args)

    def _canvas_xview(self, *args: object) -> None:
        """Scrollbar callback proxy for horizontal canvas scrolling."""
        canvas = cast(Any, self.canvas)
        canvas.xview(*args)

    def _bind_canvas_events(self) -> None:
        """Binds mouse events for painting, erasing, and coordinate tracking."""
        self.canvas.bind("<ButtonPress-1>", self._on_left_paint)
        self.canvas.bind("<B1-Motion>", self._on_left_paint)
        self.canvas.bind("<ButtonPress-3>", self._on_right_erase)
        self.canvas.bind("<B3-Motion>", self._on_right_erase)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def _parse_positive_int(self, value: str, field_name: str) -> int:
        """Parses a positive integer from a text entry value."""
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} deve ser inteiro.") from exc

        if parsed <= 0:
            raise ValueError(f"{field_name} deve ser maior que zero.")

        return parsed

    def _reset_canvas_map(self, rows: int, cols: int, cell_size: int) -> None:
        """Resets canvas and editor state for a map with given dimensions."""
        self.config_state = EditorConfig(rows=rows, cols=cols, cell_size=cell_size)
        self.obstacles.clear()
        self.obstacle_items.clear()

        self.canvas.delete("all")

        map_width = cols * cell_size
        map_height = rows * cell_size

        self.canvas.configure(scrollregion=(0, 0, map_width, map_height))
        self.canvas.create_rectangle(
            0,
            0,
            map_width,
            map_height,
            fill=PALETTE["cell_free"],
            outline="",
            tags=("background",),
        )
        self._draw_grid_lines(rows=rows, cols=cols, cell_size=cell_size)
        self._update_status(None)

    def _cell_size_for_loaded_map(self) -> int:
        """Returns a valid cell size when loading map data into the editor."""
        raw_cell_size = self.config_var_cell_size.get().strip()
        try:
            cell_size = self._parse_positive_int(raw_cell_size, "Cell Size")
        except ValueError:
            cell_size = DEFAULT_CELL_SIZE

        if not (MIN_CELL_SIZE <= cell_size <= MAX_CELL_SIZE):
            cell_size = DEFAULT_CELL_SIZE

        self.config_var_cell_size.set(str(cell_size))
        return cell_size

    def _load_map_grid(self, map_grid: MapGrid) -> None:
        """Loads an existing `MapGrid` instance into the editor canvas."""
        cell_size = self._cell_size_for_loaded_map()

        self.config_var_rows.set(str(map_grid.rows))
        self.config_var_cols.set(str(map_grid.cols))
        self._reset_canvas_map(rows=map_grid.rows, cols=map_grid.cols, cell_size=cell_size)

        for row_index in range(map_grid.rows):
            for col_index in range(map_grid.cols):
                if not map_grid.is_free(row_index, col_index):
                    self._paint_cell(row_index, col_index)

        self._update_status(None)

    def _resolve_catalog_map_choice(
        self,
        raw_choice: str,
        available_maps: list[str],
    ) -> str:
        """Resolves user input as a valid catalog map name."""
        normalized_choice = raw_choice.strip()
        if not normalized_choice:
            raise ValueError("Nome do mapa nao pode estar vazio.")

        if normalized_choice.isdigit():
            selected_index = int(normalized_choice)
            if 1 <= selected_index <= len(available_maps):
                return available_maps[selected_index - 1]

        if normalized_choice in available_maps:
            return normalized_choice

        lowered_choice = normalized_choice.lower()
        matching_names = [
            map_name for map_name in available_maps if map_name.lower() == lowered_choice
        ]
        if len(matching_names) == 1:
            return matching_names[0]

        raise ValueError(f"Mapa {normalized_choice!r} nao encontrado no catalogo.")

    def create_map_from_inputs(self) -> None:
        """Creates/recreates the drawing map based on panel inputs."""
        try:
            rows = self._parse_positive_int(self.config_var_rows.get().strip(), "Rows")
            cols = self._parse_positive_int(self.config_var_cols.get().strip(), "Cols")
            cell_size = self._parse_positive_int(
                self.config_var_cell_size.get().strip(),
                "Cell Size",
            )
        except ValueError as error:
            messagebox.showerror("Entrada invalida", str(error))
            return

        if not (MIN_CELL_SIZE <= cell_size <= MAX_CELL_SIZE):
            messagebox.showerror(
                "Entrada invalida",
                f"Cell Size deve estar entre {MIN_CELL_SIZE} e {MAX_CELL_SIZE}.",
            )
            return

        self._reset_canvas_map(rows=rows, cols=cols, cell_size=cell_size)

    def _draw_grid_lines(self, rows: int, cols: int, cell_size: int) -> None:
        """Draws grid lines over the map area for visual cell guidance."""
        height = rows * cell_size
        width = cols * cell_size

        for row_index in range(rows + 1):
            y = row_index * cell_size
            self.canvas.create_line(0, y, width, y, fill=PALETTE["grid_line"], tags=("grid",))

        for col_index in range(cols + 1):
            x = col_index * cell_size
            self.canvas.create_line(x, 0, x, height, fill=PALETTE["grid_line"], tags=("grid",))

    def _event_to_cell(self, event: tk.Event[tk.Misc]) -> tuple[int, int] | None:
        """Converts a canvas event position into `(row, col)` grid coordinates."""
        cell_size = self.config_state.cell_size
        canvas = cast(Any, self.canvas)
        event_x = int(cast(Any, event).x)
        event_y = int(cast(Any, event).y)
        canvas_x = int(canvas.canvasx(event_x))
        canvas_y = int(canvas.canvasy(event_y))

        col = canvas_x // cell_size
        row = canvas_y // cell_size

        if 0 <= row < self.config_state.rows and 0 <= col < self.config_state.cols:
            return (row, col)
        return None

    def _cell_bbox(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Returns pixel bounds for a grid cell on the canvas."""
        size = self.config_state.cell_size
        x1 = col * size
        y1 = row * size
        return (x1, y1, x1 + size, y1 + size)

    def _paint_cell(self, row: int, col: int) -> None:
        """Marks one cell as obstacle and updates its canvas item."""
        point = (row, col)
        if point in self.obstacles:
            return

        self.obstacles.add(point)
        x1, y1, x2, y2 = self._cell_bbox(row, col)
        item_id = self.canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=PALETTE["cell_obstacle"],
            outline="",
            tags=("obstacle",),
        )
        self.obstacle_items[point] = item_id

    def _erase_cell(self, row: int, col: int) -> None:
        """Restores one cell from obstacle back to free space."""
        point = (row, col)
        item_id = self.obstacle_items.pop(point, None)
        if item_id is None:
            return

        self.obstacles.discard(point)
        self.canvas.delete(item_id)

    def _on_left_paint(self, event: tk.Event[tk.Misc]) -> None:
        """Mouse handler: paints cells while left button is pressed."""
        cell = self._event_to_cell(event)
        if cell is not None:
            self._paint_cell(*cell)
        self._update_status(cell)

    def _on_right_erase(self, event: tk.Event[tk.Misc]) -> None:
        """Mouse handler: erases cells while right button is pressed."""
        cell = self._event_to_cell(event)
        if cell is not None:
            self._erase_cell(*cell)
        self._update_status(cell)

    def _on_mouse_move(self, event: tk.Event[tk.Misc]) -> None:
        """Mouse handler: keeps status bar cursor coordinates up to date."""
        self._update_status(self._event_to_cell(event))

    def _on_mouse_leave(self, _event: tk.Event[tk.Misc]) -> None:
        """Mouse handler: clears cursor coordinates when leaving canvas."""
        self._update_status(None)

    def _update_status(self, cell: tuple[int, int] | None) -> None:
        """Updates status bar text with map and cursor information."""
        rows = self.config_state.rows
        cols = self.config_state.cols
        obstacle_count = len(self.obstacles)

        if cell is None:
            cursor_info = "row=-, col=-"
        else:
            cursor_info = f"row={cell[0]}, col={cell[1]}"

        self.status_var.set(
            f"Mapa {rows}x{cols} | Obstaculos: {obstacle_count} | Cursor: {cursor_info}"
        )

    def clear_obstacles(self) -> None:
        """Clears all painted obstacle cells from the current map."""
        if not self.obstacles:
            self._update_status(None)
            return

        for item_id in self.obstacle_items.values():
            self.canvas.delete(item_id)
        self.obstacle_items.clear()
        self.obstacles.clear()
        self._update_status(None)

    def load_catalog_map(self) -> None:
        """Loads an existing map from the shared JSON map catalog."""
        try:
            available_maps = list_catalog_maps(DEFAULT_CATALOG_PATH)
        except ValueError as error:
            messagebox.showerror("Falha ao ler catalogo", str(error), parent=self)
            return

        if not available_maps:
            messagebox.showinfo(
                "Catalogo vazio",
                (
                    "Nenhum mapa encontrado em:\n"
                    f"{DEFAULT_CATALOG_PATH}\n\n"
                    "Use 'Exportar Catalogo JSON' para salvar um mapa primeiro."
                ),
                parent=self,
            )
            return

        preview_limit = 20
        preview_lines = [
            f"[{index}] {map_name}"
            for index, map_name in enumerate(available_maps[:preview_limit], start=1)
        ]
        if len(available_maps) > preview_limit:
            preview_lines.append(
                f"... e mais {len(available_maps) - preview_limit} mapa(s)."
            )

        raw_choice = simpledialog.askstring(
            "Carregar Catalogo JSON",
            (
                "Escolha o mapa salvo (numero ou nome):\n\n"
                + "\n".join(preview_lines)
            ),
            initialvalue="1",
            parent=self,
        )
        if raw_choice is None:
            return

        try:
            selected_map_name = self._resolve_catalog_map_choice(raw_choice, available_maps)
            map_grid = create_map_from_catalog(
                map_name=selected_map_name,
                scale_factor=1,
                catalog_path=DEFAULT_CATALOG_PATH,
            )
        except ValueError as error:
            messagebox.showerror("Falha ao carregar mapa", str(error), parent=self)
            return

        self._load_map_grid(map_grid)

        messagebox.showinfo(
            "Mapa carregado",
            (
                f"Mapa carregado do catalogo: {selected_map_name}\n\n"
                f"Dimensoes: {self.config_state.rows}x{self.config_state.cols}\n"
                f"Celulas de obstaculo: {len(self.obstacles)}"
            ),
            parent=self,
        )

    def _normalize_preset_name(self, raw_name: str) -> str:
        """Normalizes and validates a preset module name."""
        normalized = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
        normalized = re.sub(r"_+", "_", normalized)

        if not normalized:
            raise ValueError("Nome do preset nao pode estar vazio.")
        if not normalized.isidentifier() or keyword.iskeyword(normalized):
            raise ValueError(
                "Nome invalido. Use letras, numeros e underscore, sem iniciar por numero."
            )

        return normalized

    def _extract_row_runs(self, row_index: int) -> list[tuple[int, int]]:
        """Extracts contiguous obstacle runs `[col_start, col_end)` from one row."""
        runs: list[tuple[int, int]] = []
        col = 0
        cols = self.config_state.cols

        while col < cols:
            if (row_index, col) not in self.obstacles:
                col += 1
                continue

            start_col = col
            col += 1
            while col < cols and (row_index, col) in self.obstacles:
                col += 1
            runs.append((start_col, col))

        return runs

    def _build_rectangles_from_obstacles(self) -> list[GridRectangle]:
        """Builds rectangle list by vertically merging equal row runs.

        Heuristic:
        1. For each row, detect contiguous obstacle runs `[c_start, c_end)`.
        2. Merge runs with the same horizontal span across consecutive rows.

        This avoids exporting a giant 0/1 matrix and keeps output compact while
        preserving exact obstacle coverage.
        """
        rectangles: list[GridRectangle] = []

        active: dict[tuple[int, int], tuple[int, int]] = {}

        for row in range(self.config_state.rows):
            row_runs = self._extract_row_runs(row)
            current: dict[tuple[int, int], tuple[int, int]] = {}

            for run in row_runs:
                if run in active:
                    row_start, _ = active[run]
                    current[run] = (row_start, row + 1)
                else:
                    current[run] = (row, row + 1)

            for run, (row_start, row_end) in active.items():
                if run in current:
                    continue
                col_start, col_end = run
                rectangles.append((row_start, row_end, col_start, col_end))

            active = current

        for run, (row_start, row_end) in active.items():
            col_start, col_end = run
            rectangles.append((row_start, row_end, col_start, col_end))

        rectangles.sort(key=lambda rect: (rect[0], rect[2], rect[1], rect[3]))
        return rectangles

    def _render_preset_source(self, rectangles: list[GridRectangle]) -> str:
        """Renders the exported preset module source code."""
        rectangle_lines = (
            "\n".join(f"    {rectangle}," for rectangle in rectangles)
            if rectangles
            else "    # No obstacles were painted."
        )

        rows = self.config_state.rows
        cols = self.config_state.cols

        return f'''"""Custom map preset generated by map_editor.py.

This preset stores obstacle geometry as `[start, end)` rectangles compatible
with `MapGrid.add_obstacle_rect`.
"""

from typing import Final, TypeAlias

from core.map_grid import MapGrid

GridRectangle: TypeAlias = tuple[int, int, int, int]

BASE_MAP_ROWS: Final[int] = {rows}
BASE_MAP_COLS: Final[int] = {cols}

OBSTACLE_RECTANGLES: Final[tuple[GridRectangle, ...]] = (
{rectangle_lines}
)


def _validate_scale_factor(scale_factor: object) -> int:
    """Validates and returns a positive integer map scale factor."""
    if (
        isinstance(scale_factor, bool)
        or not isinstance(scale_factor, int)
        or scale_factor <= 0
    ):
        raise ValueError(
            "scale_factor must be a positive integer; "
            f"received {{scale_factor!r}}."
        )

    return scale_factor


def _scale_rectangle(rectangle: GridRectangle, scale_factor: int) -> GridRectangle:
    """Scales a `[start, end)` rectangle by the same factor in both axes."""
    row_start, row_end, col_start, col_end = rectangle
    return (
        row_start * scale_factor,
        row_end * scale_factor,
        col_start * scale_factor,
        col_end * scale_factor,
    )


def create_custom_map(scale_factor: int = 1) -> MapGrid:
    """Creates a custom `MapGrid` from exported obstacle rectangles."""
    valid_scale_factor = _validate_scale_factor(scale_factor)

    map_grid = MapGrid(
        rows=BASE_MAP_ROWS * valid_scale_factor,
        cols=BASE_MAP_COLS * valid_scale_factor,
    )

    for rectangle in OBSTACLE_RECTANGLES:
        map_grid.add_obstacle_rect(*_scale_rectangle(rectangle, valid_scale_factor))

    return map_grid


__all__ = ["create_custom_map"]
'''

    def export_preset(self) -> None:
        """Exports the painted map to `./presets/<preset_name>.py`."""
        raw_name = simpledialog.askstring(
            "Exportar Preset",
            "Nome do arquivo preset (sem .py):",
            initialvalue="my_custom_map",
            parent=self,
        )
        if raw_name is None:
            return

        try:
            preset_name = self._normalize_preset_name(raw_name)
        except ValueError as error:
            messagebox.showerror("Nome invalido", str(error))
            return

        rectangles = self._build_rectangles_from_obstacles()
        module_source = self._render_preset_source(rectangles)

        presets_dir = PROJECT_ROOT / "presets"
        presets_dir.mkdir(parents=True, exist_ok=True)

        output_path = presets_dir / f"{preset_name}.py"
        if output_path.exists():
            should_overwrite = messagebox.askyesno(
                "Arquivo existente",
                f"{output_path.name} ja existe. Deseja sobrescrever?",
                parent=self,
            )
            if not should_overwrite:
                return

        output_path.write_text(module_source, encoding="utf-8", newline="\n")

        messagebox.showinfo(
            "Exportacao concluida",
            (
                f"Preset salvo em:\n{output_path}\n\n"
                f"Retangulos exportados: {len(rectangles)}\n"
                f"Celulas de obstaculo: {len(self.obstacles)}"
            ),
            parent=self,
        )

    def export_catalog_map(self) -> None:
        """Exports the painted map into the shared JSON map catalog."""
        raw_name = simpledialog.askstring(
            "Exportar Catalogo JSON",
            "Nome do mapa no catalogo:",
            initialvalue="map_custom",
            parent=self,
        )
        if raw_name is None:
            return

        try:
            map_name = self._normalize_preset_name(raw_name)
        except ValueError as error:
            messagebox.showerror("Nome invalido", str(error))
            return

        rectangles = self._build_rectangles_from_obstacles()
        try:
            upsert_catalog_map(
                map_name=map_name,
                rows=self.config_state.rows,
                cols=self.config_state.cols,
                obstacle_rectangles=rectangles,
                catalog_path=DEFAULT_CATALOG_PATH,
            )
        except ValueError as error:
            messagebox.showerror("Falha na exportacao", str(error))
            return

        messagebox.showinfo(
            "Exportacao concluida",
            (
                f"Mapa salvo no catalogo JSON:\n{DEFAULT_CATALOG_PATH}\n\n"
                f"Nome do mapa: {map_name}\n"
                f"Retangulos exportados: {len(rectangles)}\n"
                f"Celulas de obstaculo: {len(self.obstacles)}"
            ),
            parent=self,
        )


def main() -> None:
    """Entrypoint for launching the map editor UI."""
    app = MapEditorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
