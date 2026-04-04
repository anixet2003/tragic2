# TRAGIC

Traffic Analysis with Generative Intelligence and Crowd Simulation.

TRAGIC simulates crowd evacuation on raster floorplans, applies hazards, and generates analytics plus floor-plan improvement suggestions.

Note: DXF support has been removed. The project is image-only.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py bc.jpeg
```

## CLI Usage

### Run from floorplan

```bash
python main.py path/to/floorplan.jpg
```

Supported formats:
- `.png`, `.jpg`, `.jpeg`, `.bmp`

When you run with a floorplan, TRAGIC auto-generates a reusable config in `generated_configs/`.

### Run from generated config (no floorplan needed)

```bash
python main.py --config generated_configs/your_floorplan_xxxxxxxx.yaml
```

Optional custom generated-config path during floorplan run:

```bash
python main.py bc.jpeg --save-config generated_configs/bc_custom.yaml
```

### Common flags

```bash
python main.py floorplan.jpg --agents 500 --duration 300 --scale 10 --batch
```

- `--scale`
- `--agents`
- `--duration`
- `--time-step`
- `--no-viz`
- `--batch`
- `--config`
- `--save-config`

## Streamlit App

Run:

```bash
python ui_runner.py
```

Current Streamlit modes:
- Floorplan file: upload floorplan, run simulation, and auto-save generated config.
- Generated config file: run directly from a previously generated config.

Tip: Run once in "Floorplan file" mode to create entries under `generated_configs/`, then reuse them in "Generated config file" mode.

## Outputs

After each run, check `output/` for:
- `floorplan_analytics.csv`
- `agent_paths.png`
- `heatmaps/density_heatmap.png`
- `heatmaps/panic_heatmap.png`
- `floorplan_improvement_suggestions.md`
- `floorplan_improvement_overlay.png`

## Key Folders

- `generated_configs/`: per-floorplan reusable configs.
- `output/`: analytics and visual artifacts from runs.
- `src/`: simulation engine and core modules.

Both `generated_configs/` and `output/` are created automatically at runtime when needed.