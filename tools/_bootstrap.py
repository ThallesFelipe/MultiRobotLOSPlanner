"""Shared import-path bootstrap for standalone tool scripts."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"


def ensure_project_root_on_path() -> None:
    """Makes project-local packages importable when a tool is run directly."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


ensure_project_root_on_path()

__all__ = ["PROJECT_ROOT", "TOOLS_DIR", "ensure_project_root_on_path"]
