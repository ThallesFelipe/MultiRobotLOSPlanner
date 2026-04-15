"""Interactive terminal launcher for map tools.

Usage:
    python tools/terminal_menu.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"


def _run_script(script_name: str, extra_args: list[str] | None = None) -> int:
    """Runs a tool script with the project root as current directory."""
    command = [sys.executable, str(TOOLS_DIR / script_name)]
    if extra_args:
        command.extend(extra_args)

    print("\nExecutando:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def _show_menu() -> None:
    """Prints the available tool actions."""
    print("\n=== MultiRobot LOS Planner - Menu de Ferramentas ===")
    print("[1] Abrir editor de mapa (GUI)")
    print("[2] Exportar occupancy grid")
    print("[3] Exportar grafo de visibilidade")
    print("[4] Planejar caminho (interativo)")
    print("[5] Sair")


def main() -> None:
    """Entrypoint for interactive terminal menu."""
    while True:
        _show_menu()
        choice = input("Escolha uma opcao [1-5]: ").strip().lower()

        if choice == "1":
            exit_code = _run_script("map_editor.py")
            print(f"Editor finalizado com codigo: {exit_code}")
            continue

        if choice == "2":
            exit_code = _run_script("export_map_visualization.py", ["--interactive"])
            print(f"Exportacao de mapa finalizada com codigo: {exit_code}")
            continue

        if choice == "3":
            exit_code = _run_script("export_visibility_graph.py", ["--interactive"])
            print(f"Exportacao do grafo finalizada com codigo: {exit_code}")
            continue

        if choice == "4":
            exit_code = _run_script("export_planned_path.py", ["--interactive"])
            print(f"Planejamento finalizado com codigo: {exit_code}")
            continue

        if choice in {"5", "q", "quit", "s", "sair"}:
            print("Encerrando menu.")
            return

        print("Opcao invalida. Digite 1, 2, 3, 4 ou 5.")


if __name__ == "__main__":
    main()
