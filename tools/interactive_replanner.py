"""Interactive GUI demo for ordered_progression + reactive_replanning.

Launches a Tkinter window that embeds a Matplotlib map view where the user can:
1. Pick a map from the JSON catalog.
2. Click the map to define source (green) and target (red) points.
3. Run `ordered_progression` step by step, watching each robot move.
4. Click at any moment to insert a dynamic obstacle; as soon as it invalidates
   the current plan, `reactive_replanning` is automatically triggered.
5. See the original path, the blocked segment, the replanned path and the new
   per-step execution on the same canvas.

Every algorithmic event is logged in the side panel so the execution is fully
transparent.

Usage:
    python tools/interactive_replanner.py
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, cast

try:
    from tools._bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

import matplotlib

matplotlib.use("TkAgg")

import networkx as nx
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from algorithms.ordered_progression import (
    MovementSnapshot,
    ordered_progression,
)
from algorithms.connectivity_checks import temporary_los_connectivity_check
from algorithms.reactive_replanning import reactive_replanning
from algorithms.relay_dijkstra import (
    DEFAULT_RELAY_PENALTY_LAMBDA,
    INFINITE_PATH_COST,
    relay_dijkstra,
)
from core.map_grid import GridPoint, MapGrid
from core.map_processor import MapProcessor
from core.visibility import (
    DEFAULT_DIAGONAL_FLANK_POLICY,
    DiagonalFlankPolicy,
    bresenham,
    has_line_of_sight,
)
from core.visibility_graph import build_visibility_graph
from presets.map_catalog import (
    DEFAULT_CATALOG_PATH,
    create_map_from_catalog,
    list_catalog_maps,
)
from tools.common import extract_boundary_vertices

FREE_SPACE_COLOR = "#F4F1E8"
OBSTACLE_COLOR = "#1F2321"
DYNAMIC_OBSTACLE_COLOR = "#8E1A1A"
ORIGINAL_PATH_COLOR = "#1F4D3A"
REPLANNED_PATH_COLOR = "#C0681B"
BLOCKED_PATH_COLOR = "#7A7A7A"
SOURCE_COLOR = "#1F4D3A"
TARGET_COLOR = "#C7472D"
ROBOT_PALETTE: tuple[str, ...] = (
    "#E2572C",
    "#1F4D3A",
    "#C0681B",
    "#3E5C76",
    "#8E6C8A",
    "#5D7052",
    "#B5651D",
    "#4A6FA5",
)

DEFAULT_BOUNDARY_STRIDE = 1
DEFAULT_MAX_VERTICES = 600
DEFAULT_STEP_DELAY_MS = 350
DEFAULT_OBSTACLE_RADIUS = 1
DEFAULT_PLAN_PREFER_FEWER_RELAYS = False
DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY: DiagonalFlankPolicy = DEFAULT_DIAGONAL_FLANK_POLICY
DEFAULT_FLEET_ROBOT_COUNT = 4


def _euclidean_distance(p1: GridPoint, p2: GridPoint) -> float:
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _extract_boundary_vertices(
    map_grid: MapGrid,
    boundary_stride: int = DEFAULT_BOUNDARY_STRIDE,
    max_vertices: int | None = DEFAULT_MAX_VERTICES,
) -> list[GridPoint]:
    """Extracts free cells that touch obstacle cells to serve as graph vertices."""
    return extract_boundary_vertices(
        map_grid,
        boundary_stride,
        max_vertices,
        sampling_mode="floor",
    )


def _extract_visibility_vertices(map_grid: MapGrid) -> list[GridPoint]:
    """Extracts free visibility vertices using corners plus passage samples."""
    corner_vertices = MapProcessor(map_grid).extract_graph_vertices()
    passage_vertices = _extract_boundary_vertices(map_grid)
    return sorted(set(corner_vertices) | set(passage_vertices))


def _connect_endpoint(
    graph: nx.Graph[GridPoint],
    map_grid: MapGrid,
    endpoint: GridPoint,
) -> None:
    """Adds an endpoint node with LOS edges to the visibility graph."""
    if endpoint not in graph:
        graph.add_node(endpoint)
    for candidate in list(graph.nodes):
        if candidate == endpoint:
            continue
        if not has_line_of_sight(
            map_grid,
            endpoint,
            candidate,
            diagonal_flank_policy=DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY,
        ):
            continue
        if not graph.has_edge(endpoint, candidate):
            graph.add_edge(
                endpoint,
                candidate,
                weight=_euclidean_distance(endpoint, candidate),
            )


class InteractiveReplannerApp(tk.Tk):
    """Main application window for the interactive replanner demo."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Interactive Ordered Progression + Reactive Replanning")
        self.geometry("1500x900")
        self.minsize(1200, 760)

        self.catalog_path: Path = DEFAULT_CATALOG_PATH
        self.map_grid: MapGrid | None = None
        self.base_map_array: np.ndarray | None = None
        self.dynamic_obstacles: set[GridPoint] = set()
        self.pending_obstacle_events: list[tuple[set[GridPoint], GridPoint]] = []

        self.source_point: GridPoint | None = None
        self.target_point: GridPoint | None = None

        self.base_vis_graph: nx.Graph[GridPoint] | None = None
        self.planning_graph: nx.Graph[GridPoint] | None = None

        self.original_path: list[GridPoint] = []
        self.current_path: list[GridPoint] = []

        self.snapshots: list[MovementSnapshot] = []
        self.current_snapshot_index: int = 0
        self.current_positions: dict[int, GridPoint] = {}
        self.n_robots: int = 0

        self.click_mode: str = "source"
        self.playing: bool = False
        self.play_job: str | None = None
        self.replanning_active: bool = False

        self.map_name_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="source")
        self.status_var = tk.StringVar(
            value="Selecione um mapa e clique em 'Carregar mapa'."
        )
        self.step_info_var = tk.StringVar(value="Step: -")
        self.speed_var = tk.IntVar(value=DEFAULT_STEP_DELAY_MS)
        self.obstacle_radius_var = tk.IntVar(value=DEFAULT_OBSTACLE_RADIUS)

        self._build_layout()
        self._refresh_map_list()

    def _build_layout(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Mapa:").pack(side=tk.LEFT)
        self.map_combo = ttk.Combobox(
            top, textvariable=self.map_name_var, width=28, state="readonly"
        )
        self.map_combo.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(top, text="Carregar mapa", command=self._on_load_map).pack(
            side=tk.LEFT
        )
        ttk.Separator(top, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )

        ttk.Label(top, text="Clique define:").pack(side=tk.LEFT)
        for label, value in (
            ("Início", "source"),
            ("Destino", "target"),
            ("Obstáculo", "obstacle"),
        ):
            ttk.Radiobutton(
                top,
                text=label,
                value=value,
                variable=self.mode_var,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Raio obst.:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(
            top,
            from_=0,
            to=10,
            textvariable=self.obstacle_radius_var,
            width=4,
        ).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        ttk.Button(
            top, text="Planejar (Ordered)", command=self._on_plan
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="⏮", width=3, command=self._on_first).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(top, text="◀", width=3, command=self._on_prev).pack(
            side=tk.LEFT, padx=1
        )
        self.play_button = ttk.Button(
            top, text="▶ Play", width=8, command=self._on_play_pause
        )
        self.play_button.pack(side=tk.LEFT, padx=1)
        ttk.Button(top, text="▶", width=3, command=self._on_next).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(top, text="⏭", width=3, command=self._on_last).pack(
            side=tk.LEFT, padx=1
        )

        ttk.Label(top, text="Delay (ms):").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(
            top,
            from_=50,
            to=2000,
            increment=50,
            textvariable=self.speed_var,
            width=6,
        ).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        ttk.Button(
            top, text="Limpar obstáculos", command=self._on_clear_obstacles
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Resetar pontos", command=self._on_reset_points).pack(
            side=tk.LEFT, padx=2
        )

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure: Any = Figure(figsize=(9, 7), dpi=100)
        self.ax: Any = self.figure.add_subplot(111)
        self.ax.set_facecolor(FREE_SPACE_COLOR)

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        side = ttk.Frame(body, padding=6)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="Passo a passo", font=("TkDefaultFont", 10, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(side, textvariable=self.step_info_var).pack(anchor=tk.W)

        log_frame = ttk.Frame(side)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.log_text = tk.Text(
            log_frame, width=48, height=30, wrap=tk.WORD, state=tk.DISABLED
        )
        yview_command = cast(Callable[..., Any], getattr(self.log_text, "yview"))
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient=tk.VERTICAL,
            command=yview_command,
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text.tag_configure("info", foreground="#1F4D3A")
        self.log_text.tag_configure("warn", foreground="#C0681B")
        self.log_text.tag_configure("error", foreground="#8E1A1A")
        self.log_text.tag_configure("step", foreground="#3E5C76")

        status_bar = ttk.Frame(self, padding=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, textvariable=self.status_var, anchor=tk.W).pack(
            fill=tk.X
        )

    def _refresh_map_list(self) -> None:
        try:
            maps = list_catalog_maps(self.catalog_path)
        except Exception as exc:
            messagebox.showerror("Catálogo", f"Falha ao ler catálogo: {exc}")
            maps = []
        self.map_combo["values"] = maps
        if maps and not self.map_name_var.get():
            self.map_name_var.set(maps[0])

    def _on_load_map(self) -> None:
        map_name = self.map_name_var.get().strip()
        if not map_name:
            messagebox.showwarning("Mapa", "Selecione um mapa do catálogo.")
            return

        try:
            map_grid = create_map_from_catalog(map_name, catalog_path=self.catalog_path)
        except Exception as exc:
            messagebox.showerror("Mapa", f"Erro ao carregar mapa: {exc}")
            return

        self.map_grid = map_grid
        self.base_map_array = map_grid.grid.astype(np.int32).copy()
        self.dynamic_obstacles.clear()
        self.pending_obstacle_events.clear()
        self.source_point = None
        self.target_point = None
        self.snapshots = []
        self.current_snapshot_index = 0
        self.current_positions = {}
        self.original_path = []
        self.current_path = []
        self.replanning_active = False
        self._stop_playback()

        self._log(f"Mapa '{map_name}' carregado ({map_grid.rows}x{map_grid.cols}).", "info")
        self._set_status("Mapa carregado. Clique no mapa para definir origem e destino.")

        self._log(
            "Construindo grafo de visibilidade (Canny/cantos/DBSCAN + passagens livres)...",
            "info",
        )
        vertices = _extract_visibility_vertices(map_grid)
        self.base_vis_graph = build_visibility_graph(
            map_grid,
            vertices,
            diagonal_flank_policy=DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY,
        )
        self._log(
            f"Grafo base: {self.base_vis_graph.number_of_nodes()} vertices, "
            f"{self.base_vis_graph.number_of_edges()} arestas.",
            "info",
        )
        self.planning_graph = None

        self.mode_var.set("source")
        self._render()

    def _on_mode_change(self) -> None:
        self.click_mode = self.mode_var.get()
        self._set_status(
            {
                "source": "Clique no mapa para definir a ORIGEM.",
                "target": "Clique no mapa para definir o DESTINO.",
                "obstacle": "Clique no mapa para inserir um OBSTÁCULO dinâmico.",
            }.get(self.click_mode, "")
        )

    def _on_canvas_click(self, event: Event) -> None:
        if not isinstance(event, MouseEvent):
            return
        if event.inaxes is not self.ax:
            return
        if self.map_grid is None or event.xdata is None or event.ydata is None:
            return

        col = int(round(float(event.xdata)))
        row = int(round(float(event.ydata)))
        if not self.map_grid.in_bounds(row, col):
            return

        mode = self.mode_var.get()
        point: GridPoint = (row, col)

        if mode == "source":
            self._set_source(point)
        elif mode == "target":
            self._set_target(point)
        elif mode == "obstacle":
            self._insert_obstacle(point)

    def _set_source(self, point: GridPoint) -> None:
        assert self.map_grid is not None
        if self._is_blocked_cell(point):
            self._log(f"Origem {point} cai em obstáculo; escolha célula livre.", "error")
            return
        self.source_point = point
        self._log(f"Origem definida em {point}.", "info")
        self.mode_var.set("target" if self.target_point is None else "source")
        self._on_mode_change()
        self._render()

    def _set_target(self, point: GridPoint) -> None:
        assert self.map_grid is not None
        if self._is_blocked_cell(point):
            self._log(f"Destino {point} cai em obstáculo; escolha célula livre.", "error")
            return
        self.target_point = point
        self._log(f"Destino definido em {point}.", "info")
        self._render()

    def _insert_obstacle(self, point: GridPoint) -> None:
        """Inserts a dynamic obstacle. Triggers reactive replanning if needed."""
        assert self.map_grid is not None and self.base_map_array is not None
        radius = max(0, int(self.obstacle_radius_var.get()))

        new_cells: set[GridPoint] = set()
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                cell = (point[0] + d_row, point[1] + d_col)
                if not self.map_grid.in_bounds(cell[0], cell[1]):
                    continue
                if cell == self.source_point or cell == self.target_point:
                    continue
                new_cells.add(cell)

        if not new_cells:
            return

        inserted_cells: set[GridPoint] = set()
        for cell in new_cells:
            if cell in self.dynamic_obstacles:
                continue
            self.dynamic_obstacles.add(cell)
            try:
                self.map_grid.add_obstacle(cell[0], cell[1])
            except ValueError:
                continue
            inserted_cells.add(cell)

        if not inserted_cells:
            return

        self._log(
            f"Obstáculo dinâmico inserido em {point} (raio={radius}, "
            f"{len(inserted_cells)} células).",
            "warn",
        )
        self._render()

        if self.current_path:
            self._maybe_trigger_replanning(inserted_cells, point)

    def _count_blocked_steps(self, snapshots: list[MovementSnapshot]) -> int:
        """Counts blocked movement attempts, excluding the initial snapshot."""
        return sum(
            1
            for snapshot in snapshots[1:]
            if not snapshot["valid"]
        )

    def _count_valid_robot_moves(self, snapshots: list[MovementSnapshot]) -> int:
        """Counts effective robot displacements, excluding blocked attempts."""
        return sum(
            1
            for snapshot in snapshots[1:]
            if snapshot["valid"] and snapshot["robot_id"] is not None
        )

    def _build_plan_candidate(
        self,
        planning_graph: nx.Graph[GridPoint],
    ) -> tuple[float, list[GridPoint], list[MovementSnapshot], int] | None:
        """Builds one planning candidate and scores it by blocked steps."""
        assert self.source_point is not None and self.target_point is not None
        assert self.map_grid is not None

        cost, path = relay_dijkstra(
            planning_graph,
            self.source_point,
            self.target_point,
            lam=DEFAULT_RELAY_PENALTY_LAMBDA,
            prefer_fewer_relays=DEFAULT_PLAN_PREFER_FEWER_RELAYS,
        )
        if cost == INFINITE_PATH_COST or not path:
            return None

        robot_count = max(DEFAULT_FLEET_ROBOT_COUNT, len(path) - 1)
        snapshots = ordered_progression(
            path,
            grid_obj=self.map_grid,
            vis_graph=planning_graph,
            robot_count=robot_count,
        )
        blocked_steps = self._count_blocked_steps(snapshots)
        return cost, list(path), snapshots, blocked_steps

    def _build_planning_graph_for_current_map(
        self,
        extra_vertices: set[GridPoint] | None = None,
    ) -> nx.Graph[GridPoint]:
        """Builds a planning graph from the current occupancy map state."""
        assert self.map_grid is not None

        vertices = _extract_visibility_vertices(self.map_grid)
        if extra_vertices:
            vertices = sorted(set(vertices) | set(extra_vertices))
        planning_graph = build_visibility_graph(
            self.map_grid,
            vertices,
            diagonal_flank_policy=DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY,
        )

        if self.source_point is not None:
            _connect_endpoint(planning_graph, self.map_grid, self.source_point)
        if self.target_point is not None:
            _connect_endpoint(planning_graph, self.map_grid, self.target_point)

        return planning_graph

    def _on_plan(self) -> None:
        if self.map_grid is None:
            messagebox.showwarning("Planejar", "Carregue um mapa primeiro.")
            return
        if self.source_point is None or self.target_point is None:
            messagebox.showwarning(
                "Planejar", "Defina origem e destino clicando no mapa."
            )
            return

        self._stop_playback()
        self.pending_obstacle_events.clear()
        self._log(
            "=== Planejamento inicial (ordered_progression) ===", "info"
        )
        self._log(
            "Construindo grafo para o estado atual do mapa (inclui obstáculos dinâmicos).",
            "info",
        )

        planning_graph = self._build_planning_graph_for_current_map()
        self.planning_graph = planning_graph

        candidate = self._build_plan_candidate(
            planning_graph,
        )

        if candidate is None:
            self._log("Nenhum caminho viável encontrado.", "error")
            self._set_status("Sem caminho. Altere pontos ou limpe obstáculos.")
            self.snapshots = []
            self.current_path = []
            self.original_path = []
            self._render()
            return

        cost, path, snapshots, blocked_steps = candidate

        self.original_path = list(path)
        self.current_path = list(path)
        self.replanning_active = False

        self._log("Plano escolhido: menor custo distância + lambda.", "info")
        self._log(
            f"Caminho encontrado: {len(path)} vertices, custo={cost:.3f}.",
            "info",
        )
        self._log(f"Caminho: {path}", "step")

        if blocked_steps > 0:
            self._log(
                f"Atenção: plano escolhido ainda possui {blocked_steps} bloqueio(s) "
                "no ordered_progression.",
                "warn",
            )

        self._install_snapshots(snapshots, f"ordered_progression ({len(snapshots) - 1} movimentos)")

    def _install_snapshots(self, snapshots: list[MovementSnapshot], label: str) -> None:
        """Installs a snapshot sequence and resets the playback cursor."""
        self.snapshots = snapshots
        self.current_snapshot_index = 0
        if snapshots:
            self.current_positions = dict(snapshots[0]["positions"])
            self.n_robots = len(self.current_positions)
            self._log(f"Cronograma instalado: {label}.", "info")
            self._log_snapshot(snapshots[0])
        else:
            self.current_positions = {}
            self.n_robots = 0
        self._update_step_info()
        self._render()

    def _on_play_pause(self) -> None:
        if not self.snapshots:
            return
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _has_terminal_failure(self) -> bool:
        """Returns whether the installed schedule ended with mission failure."""
        if not self.snapshots:
            return False

        final_snapshot = self.snapshots[-1]
        return (
            final_snapshot["robot_id"] is None
            and not final_snapshot["valid"]
            and final_snapshot["description"].startswith("Missão interrompida:")
        )

    def _start_playback(self) -> None:
        if self._has_terminal_failure() and self.current_snapshot_index >= len(self.snapshots) - 1:
            return
        if self.current_snapshot_index >= len(self.snapshots) - 1:
            self.current_snapshot_index = 0
            self._apply_current_snapshot()
        self.playing = True
        self.play_button.configure(text="⏸ Pause")
        self._schedule_next_step()

    def _stop_playback(self) -> None:
        self.playing = False
        self.play_button.configure(text="▶ Play")
        if self.play_job is not None:
            try:
                self.after_cancel(self.play_job)
            except Exception:
                pass
            self.play_job = None

    def _schedule_next_step(self) -> None:
        delay = max(50, int(self.speed_var.get()))
        self.play_job = self.after(delay, self._play_tick)

    def _play_tick(self) -> None:
        if not self.playing:
            return
        if self.current_snapshot_index >= len(self.snapshots) - 1:
            self._stop_playback()
            return
        self._on_next()
        if self.playing:
            self._schedule_next_step()

    def _on_first(self) -> None:
        if not self.snapshots:
            return
        self.current_snapshot_index = 0
        self._apply_current_snapshot()

    def _on_last(self) -> None:
        if not self.snapshots:
            return
        while self.current_snapshot_index < len(self.snapshots) - 1:
            previous_index = self.current_snapshot_index
            self._on_next()
            if self.current_snapshot_index == previous_index:
                break
            if self._has_terminal_failure():
                break

    def _on_prev(self) -> None:
        if not self.snapshots or self.current_snapshot_index <= 0:
            return
        self.current_snapshot_index -= 1
        self._apply_current_snapshot()

    def _on_next(self) -> None:
        if not self.snapshots:
            return
        if self.current_snapshot_index >= len(self.snapshots) - 1:
            return

        next_snapshot = self.snapshots[self.current_snapshot_index + 1]
        blocked_point = self._leader_runtime_blocking_point(next_snapshot)
        if blocked_point is not None:
            self._handle_leader_runtime_block(next_snapshot, blocked_point)
            return

        self.current_snapshot_index += 1
        self._apply_current_snapshot()

    def _apply_current_snapshot(self) -> None:
        snapshot = self.snapshots[self.current_snapshot_index]
        self.current_positions = dict(snapshot["positions"])
        self._log_snapshot(snapshot)
        self._update_step_info()
        self._render()
        self._check_pending_obstacles_for_leader()

    def _leader_runtime_blocking_point(
        self,
        snapshot: MovementSnapshot,
    ) -> GridPoint | None:
        """Detects a blocked leader advance before applying a stale snapshot."""
        if self.map_grid is None or self.source_point is None:
            return None
        if snapshot["robot_id"] != 0 or not snapshot["valid"]:
            return None
        if snapshot["from_pos"] is None or snapshot["to_pos"] is None:
            return None
        if snapshot["from_pos"] == snapshot["to_pos"]:
            return None

        from_pos = snapshot["from_pos"]
        to_pos = snapshot["to_pos"]
        segment_cells = list(bresenham(from_pos[0], from_pos[1], to_pos[0], to_pos[1]))

        for cell in segment_cells[1:]:
            if not self.map_grid.in_bounds(*cell):
                return cell
            if not self.map_grid.is_free(*cell):
                return cell

        if not has_line_of_sight(
            self.map_grid,
            from_pos,
            to_pos,
            diagonal_flank_policy=DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY,
        ):
            return to_pos

        simulated_positions = dict(self.current_positions)
        simulated_positions[0] = to_pos
        if not temporary_los_connectivity_check(
            self.map_grid,
            list(simulated_positions.values()),
            self.source_point,
        ):
            return to_pos

        return None

    def _handle_leader_runtime_block(
        self,
        snapshot: MovementSnapshot,
        obstacle_point: GridPoint,
    ) -> None:
        """Stops the old plan when the leader's next attempted advance fails."""
        self._log(
            "Líder tentou avançar "
            f"{snapshot['from_pos']}->{snapshot['to_pos']} e encontrou bloqueio "
            f"em {obstacle_point}. Disparando reactive_replanning.",
            "warn",
        )
        self._set_status("Líder encontrou obstáculo. Replanejamento reativo em execução...")
        self._stop_playback()
        self.pending_obstacle_events.clear()
        self._trigger_reactive_replanning(obstacle_point)

    def _halt_execution_at_current_state(
        self,
        description: str,
        status_message: str,
    ) -> None:
        """Stops execution and replaces future snapshots with a terminal failure."""
        self._stop_playback()
        self.pending_obstacle_events.clear()

        current_step = 0
        retained_snapshots: list[MovementSnapshot] = []
        if self.snapshots:
            current_index = min(self.current_snapshot_index, len(self.snapshots) - 1)
            retained_snapshots = list(self.snapshots[: current_index + 1])
            current_step = int(retained_snapshots[-1]["step"])

        failure_snapshot: MovementSnapshot = {
            "step": current_step + 1,
            "robot_id": None,
            "from_pos": None,
            "to_pos": None,
            "positions": dict(self.current_positions),
            "valid": False,
            "description": description,
        }

        self.snapshots = [*retained_snapshots, failure_snapshot]
        self.current_snapshot_index = len(self.snapshots) - 1
        self.replanning_active = True
        self._log_snapshot(failure_snapshot)
        self._set_status(status_message)
        self._update_step_info()
        self._render()

    def _leader_next_path_vertex(self) -> GridPoint | None:
        """Returns the next path vertex the leader is trying to reach."""
        if not self.current_path:
            return None

        leader_position = self.current_positions.get(0, self.current_path[0])
        if leader_position not in self.current_path:
            return None

        leader_index = self.current_path.index(leader_position)
        if leader_index + 1 >= len(self.current_path):
            return None
        return self.current_path[leader_index + 1]

    def _leader_detected_obstacle_point(
        self,
        obstacle_cells: set[GridPoint],
        fallback_obstacle_point: GridPoint,
    ) -> GridPoint | None:
        """Returns the obstacle point only if it blocks the leader's next edge."""
        assert self.map_grid is not None

        if not self.current_path:
            return None

        leader_position = self.current_positions.get(0, self.current_path[0])
        leader_next_vertex = self._leader_next_path_vertex()
        if leader_next_vertex is None:
            return None

        leader_edge_cells = set(
            bresenham(
                leader_position[0],
                leader_position[1],
                leader_next_vertex[0],
                leader_next_vertex[1],
            )
        )
        intersecting_cells = sorted(leader_edge_cells & obstacle_cells)
        if intersecting_cells:
            return intersecting_cells[0]

        if not has_line_of_sight(
            self.map_grid,
            leader_position,
            leader_next_vertex,
            diagonal_flank_policy=DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY,
        ):
            return fallback_obstacle_point

        return None

    def _maybe_trigger_replanning(
        self,
        obstacle_cells: set[GridPoint],
        fallback_obstacle_point: GridPoint,
    ) -> bool:
        """Registers obstacle cells until a leader movement actually hits them."""
        assert self.map_grid is not None

        detected_point = self._leader_detected_obstacle_point(
            obstacle_cells,
            fallback_obstacle_point,
        )
        if detected_point is None:
            self._log(
                "Obstáculo registrado, mas ainda não foi encontrado pelo líder.",
                "info",
            )
        else:
            self._log(
                "Obstáculo registrado no próximo corredor do líder; "
                "o replanejamento será disparado quando ele tentar avançar.",
                "warn",
            )

        self.pending_obstacle_events.append(
            (set(obstacle_cells), fallback_obstacle_point)
        )
        return False

    def _check_pending_obstacles_for_leader(self) -> None:
        """Keeps deferred obstacles registered for runtime leader validation."""
        if not self.pending_obstacle_events or self.map_grid is None:
            return

    def _spread_overlapping_positions(
        self,
        positions: dict[int, GridPoint],
        connectivity_graph: nx.Graph[GridPoint],
        base: GridPoint,
    ) -> dict[int, GridPoint]:
        """Compatibility shim: current robot positions are never redistributed."""
        return dict(positions)

    def _trigger_reactive_replanning(self, obstacle_point: GridPoint) -> None:
        """Rebuilds the visibility graph and runs reactive_replanning."""
        assert self.map_grid is not None
        assert self.source_point is not None and self.target_point is not None

        self._log("Reconstruindo grafo de visibilidade com obstáculos atualizados...", "info")
        new_base_graph = self._build_planning_graph_for_current_map(
            extra_vertices={obstacle_point}
        )

        initial_positions: dict[int, GridPoint] = {}
        for robot_id, pos in self.current_positions.items():
            if not self.map_grid.is_free(pos[0], pos[1]):
                self._log(
                    f"Robô r{robot_id + 1} está em célula inválida {pos}; "
                    "replanning abortado sem reposicionamento artificial.",
                    "error",
                )
                self._halt_execution_at_current_state(
                    (
                        "Missão interrompida: robô em célula ocupada após "
                        "detecção de obstáculo; nenhum reposicionamento "
                        "artificial foi executado."
                    ),
                    "Replanning inviável: robô em célula ocupada.",
                )
                return
            initial_positions[robot_id] = pos

        if 0 not in initial_positions:
            initial_positions[0] = self.source_point

        expected_ids = set(range(len(initial_positions)))
        if set(initial_positions) != expected_ids:
            self._log(
                "IDs de robôs não são contíguos; replanning abortado sem remapeamento.",
                "error",
            )
            self._halt_execution_at_current_state(
                (
                    "Missão interrompida: IDs de robôs não contíguos; "
                    "nenhum remapeamento artificial foi executado."
                ),
                "Replanning inviável: IDs de robôs não contíguos.",
            )
            return

        if len(set(initial_positions.values())) != len(initial_positions):
            self._log(
                "Posições sobrepostas preservadas; nenhum robô será redistribuído "
                "artificialmente antes do replanning.",
                "warn",
            )

        try:
            cost, new_path, new_snapshots, updated_graph = reactive_replanning(
                new_base_graph,
                source=self.source_point,
                target=self.target_point,
                initial_positions=initial_positions,
                obstacle_point=obstacle_point,
                grid_obj=self.map_grid,
                lam=DEFAULT_RELAY_PENALTY_LAMBDA,
                record_blocked_attempts=False,
            )
        except Exception as exc:
            self._log(f"Erro em reactive_replanning: {exc}", "error")
            self._halt_execution_at_current_state(
                f"Missão interrompida: erro em reactive_replanning ({exc}).",
                "Replanning falhou; execução interrompida.",
            )
            return

        self.planning_graph = updated_graph

        if cost == INFINITE_PATH_COST or not new_path:
            self._log(
                "Replanning não encontrou caminho viável após atualização do mapa.",
                "error",
            )
            self._halt_execution_at_current_state(
                (
                    "Missão interrompida: sem caminho viável após o obstáculo "
                    "encontrado pelo líder. Nenhum movimento antigo será "
                    "executado."
                ),
                "Sem caminho após obstáculo. Execução interrompida.",
            )
            return

        self.replanning_active = True
        self.current_path = list(new_path)
        self._log(
            f"Novo plano: {len(new_path)} vertices, custo={cost:.3f}.", "info"
        )
        self._log(f"Path: {new_path}", "step")
        effective_moves = self._count_valid_robot_moves(new_snapshots)
        self._install_snapshots(
            new_snapshots,
            f"reactive_replanning ({effective_moves} movimentos)",
        )
        self._set_status("Replanejamento concluído. Continue a execução passo a passo.")

    def _is_blocked_cell(self, point: GridPoint) -> bool:
        assert self.map_grid is not None
        row, col = point
        if not self.map_grid.in_bounds(row, col):
            return True
        return not self.map_grid.is_free(row, col)

    def _on_clear_obstacles(self) -> None:
        if self.map_grid is None or self.base_map_array is None:
            return
        self.map_grid.grid[:, :] = self.base_map_array
        self.dynamic_obstacles.clear()
        self.pending_obstacle_events.clear()
        self._log("Obstáculos dinâmicos removidos.", "info")
        vertices = _extract_visibility_vertices(self.map_grid)
        self.base_vis_graph = build_visibility_graph(
            self.map_grid,
            vertices,
            diagonal_flank_policy=DEFAULT_RUNTIME_DIAGONAL_FLANK_POLICY,
        )
        self.planning_graph = None
        self._render()

    def _on_reset_points(self) -> None:
        self.source_point = None
        self.target_point = None
        self.snapshots = []
        self.current_snapshot_index = 0
        self.current_positions = {}
        self.current_path = []
        self.original_path = []
        self.replanning_active = False
        self.pending_obstacle_events.clear()
        self._stop_playback()
        self._log("Pontos, plano e execução resetados.", "info")
        self.mode_var.set("source")
        self._on_mode_change()
        self._render()

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _log(self, message: str, level: str = "info") -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n", level)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_snapshot(self, snapshot: MovementSnapshot) -> None:
        level = "step" if snapshot["valid"] else "warn"
        self._log(
            f"[step {snapshot['step']}] {snapshot['description']}", level
        )

    def _update_step_info(self) -> None:
        if not self.snapshots:
            self.step_info_var.set("Step: -")
            return
        snap = self.snapshots[self.current_snapshot_index]
        self.step_info_var.set(
            f"Step {self.current_snapshot_index}/{len(self.snapshots) - 1} "
            f"(sim step={snap['step']})"
        )

    def _render(self) -> None:
        self.ax.clear()
        if self.map_grid is None:
            self.ax.text(
                0.5,
                0.5,
                "Carregue um mapa para começar.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                fontsize=12,
                color="#7A7A7A",
            )
            self.canvas.draw_idle()
            return

        display_grid = self.map_grid.grid.astype(np.int32).copy()
        for r, c in self.dynamic_obstacles:
            if self.map_grid.in_bounds(r, c):
                display_grid[r, c] = 2

        cmap = mcolors.ListedColormap(
            [FREE_SPACE_COLOR, OBSTACLE_COLOR, DYNAMIC_OBSTACLE_COLOR]
        )
        norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
        self.ax.imshow(
            display_grid,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
            zorder=0,
        )

        if self.original_path and self.replanning_active:
            self._draw_path(
                self.original_path,
                BLOCKED_PATH_COLOR,
                "Rota original (bloqueada)",
                linestyle="--",
                linewidth=1.6,
                alpha=0.75,
                zorder=2,
            )
        if self.current_path:
            label = "Rota replanejada" if self.replanning_active else "Rota planejada"
            color = REPLANNED_PATH_COLOR if self.replanning_active else ORIGINAL_PATH_COLOR
            self._draw_path(
                self.current_path, color, label, linewidth=2.4, zorder=3
            )

        if self.source_point is not None:
            self._draw_point(self.source_point, SOURCE_COLOR, "Origem", "o", size=130, zorder=5)
        if self.target_point is not None:
            self._draw_point(self.target_point, TARGET_COLOR, "Destino", "*", size=220, zorder=5)

        if self.current_positions:
            for robot_id, position in self.current_positions.items():
                color = ROBOT_PALETTE[robot_id % len(ROBOT_PALETTE)]
                self.ax.scatter(
                    [position[1]],
                    [position[0]],
                    s=180,
                    c=color,
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                )
                self.ax.text(
                    position[1],
                    position[0],
                    f"r{robot_id + 1}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    zorder=7,
                )

        legend_handles = [
            mpatches.Patch(color=FREE_SPACE_COLOR, label="Livre"),
            mpatches.Patch(color=OBSTACLE_COLOR, label="Obstáculo estático"),
            mpatches.Patch(color=DYNAMIC_OBSTACLE_COLOR, label="Obstáculo dinâmico"),
        ]
        self.ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
        )

        self.ax.set_title(
            f"Mapa: {self.map_name_var.get()} | modo clique: {self.mode_var.get()}",
            fontsize=10,
        )
        self.ax.set_xlabel("col")
        self.ax.set_ylabel("row")
        self.ax.set_xlim(-0.5, self.map_grid.cols - 0.5)
        self.ax.set_ylim(self.map_grid.rows - 0.5, -0.5)
        self.canvas.draw_idle()

    def _draw_path(
        self,
        path: list[GridPoint],
        color: str,
        label: str,
        linestyle: str = "-",
        linewidth: float = 2.0,
        alpha: float = 1.0,
        zorder: int = 3,
    ) -> None:
        if len(path) < 2:
            return
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        self.ax.plot(
            cols,
            rows,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            marker="o",
            markersize=4,
            label=label,
            zorder=zorder,
        )

    def _draw_point(
        self,
        point: GridPoint,
        color: str,
        label: str,
        marker: str,
        size: int,
        zorder: int,
    ) -> None:
        self.ax.scatter(
            [point[1]],
            [point[0]],
            s=size,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=1.2,
            zorder=zorder,
            label=label,
        )


def main() -> None:
    app = InteractiveReplannerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
