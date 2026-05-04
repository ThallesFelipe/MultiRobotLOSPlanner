# MultiRobot LOS Planner

Ferramentas Python para planejamento multi-robô com restrição de linha de visada
(LOS), geração de grafo de visibilidade, progressão ordenada e replanejamento
reativo com obstáculos dinâmicos.

## Requisitos

- Python 3.10+
- Tkinter disponível no Python local para as ferramentas GUI
- Dependências diretas em `requirements.txt`

OpenCV é opcional. Se instalado com `pip install .[opencv]`, o processamento de
bordas usa `cv2`; sem OpenCV, o projeto usa o fallback em SciPy já existente.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Também é possível instalar como pacote local:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Execução

Menu interativo:

```powershell
.\.venv\Scripts\python.exe tools\terminal_menu.py
```

Ferramentas principais:

```powershell
.\.venv\Scripts\python.exe tools\map_editor.py
.\.venv\Scripts\python.exe tools\export_map_visualization.py --help
.\.venv\Scripts\python.exe tools\export_visibility_graph.py --help
.\.venv\Scripts\python.exe tools\export_planned_path.py --help
.\.venv\Scripts\python.exe tools\interactive_replanner.py
```

## Validação

```powershell
.\.venv\Scripts\python.exe -m compileall -q algorithms core presets tools tests
.\.venv\Scripts\python.exe -m pytest
```

Os artefatos exportados são gerados em `exports/` e não devem ser versionados.
