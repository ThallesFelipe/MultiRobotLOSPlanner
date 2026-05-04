"""Core primitives for LOS-constrained multi-robot motion planning.

The package exposes the stable public API for occupancy-grid representation and
line-of-sight (LOS) validation used by the planner.
"""

from .map_grid import MapGrid
from .map_processor import (
    MapProcessingConfig,
    MapProcessor,
    compute_blocked_corridor_segment,
    extract_graph_vertices,
)
from .visibility import has_line_of_sight

__all__ = [
    "MapGrid",
    "MapProcessingConfig",
    "MapProcessor",
    "compute_blocked_corridor_segment",
    "extract_graph_vertices",
    "has_line_of_sight",
]
