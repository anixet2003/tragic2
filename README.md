# TRAGIC

Traffic Analysis with Generative Intelligence and Crowd Simulation.

TRAGIC is a floorplan-aware evacuation simulator for crowd movement, hazards, and egress analytics. It supports image and DXF floorplans, realistic motion models, hazard propagation, and automatic post-run recommendations for layout improvements.

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Who This Is For](#who-this-is-for)
3. [Quick Start (3 Minutes)](#quick-start-3-minutes)
4. [How To Run](#how-to-run)
5. [Streamlit Dashboard](#streamlit-dashboard)
6. [Outputs You Get Automatically](#outputs-you-get-automatically)
7. [Configuration Guide](#configuration-guide)
8. [How The Recommendation System Works](#how-the-recommendation-system-works)
9. [Project Structure](#project-structure)
10. [Performance Tips](#performance-tips)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)
13. [Contributing](#contributing)

## What This Project Does

TRAGIC simulates evacuation in a 2D environment derived from floorplans and provides:

- Agent-based movement using social-force and collision-avoidance ideas.
- Hazard effects (fire, smoke, exit failures).
- Time-series evacuation analytics.
- Visual outputs (paths, density heatmap, panic heatmap).
- Automatic floor-plan improvement suggestions as:
  - Markdown report.
  - Color-coded overlay image drawn on the floorplan.

## Who This Is For

- Safety and egress analysts.
- Architects and building planners.
- Researchers prototyping evacuation algorithms.
- Students learning crowd simulation.

## Quick Start (3 Minutes)

### 1) Setup environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run on the sample floorplan

```bash
python main.py bc.jpeg
```

### 3) Check generated outputs

```bash
ls -lh output
ls -lh output/heatmaps
```

Look for:

- `output/floorplan_analytics.csv`
- `output/agent_paths.png`
- `output/heatmaps/density_heatmap.png`
- `output/heatmaps/panic_heatmap.png`
- `output/floorplan_improvement_suggestions.md`
- `output/floorplan_improvement_overlay.png`

## How To Run

### A) Default config run

```bash
python main.py
```

### B) Floorplan file run

```bash
python main.py path/to/floorplan.jpg
```

Supported inputs:

- Image: `.png`, `.jpg`, `.jpeg`, `.bmp`
- CAD: `.dxf`

### C) Run with custom config file

```bash
python main.py --config config.yaml
```

### D) Common command-line flags

```bash
python main.py floorplan.jpg --agents 500 --duration 300 --scale 10 --batch
```

- `--config`: config file path.
- `--scale`: floorplan scale (pixels or units per meter).
- `--agents`: target agent count.
- `--duration`: simulation duration in seconds.
- `--time-step`: physics step size.
- `--no-viz`: disable live matplotlib rendering.
- `--batch`: non-interactive mode.

## Streamlit Dashboard

Run the dashboard:

```bash
python ui_runner.py
```

In the dashboard you can:

- Upload floorplans or edit YAML config.
- Launch simulation runs.
- View analytics preview.
- View heatmaps and movement paths.
- View the auto-generated floor-plan improvement report.
- View the color-coded improvement overlay image.

## Outputs You Get Automatically

After each completed run:

### Core analytics

- `output/floorplan_analytics.csv`
  - Columns: `Time`, `Active_Agents`, `Evacuated`, `Deceased`, `Avg_Panic`, `Avg_Speed`

### Visual diagnostics

- `output/agent_paths.png`
- `output/heatmaps/density_heatmap.png`
- `output/heatmaps/panic_heatmap.png`

### Recommendation package

- `output/floorplan_improvement_suggestions.md`
  - Structured sections: key results, findings, bottlenecks, prioritized recommendations.
- `output/floorplan_improvement_overlay.png`
  - Color-coded markers and callouts directly on the floorplan.

## Configuration Guide

Main file: `config.yaml`

### 1) Simulation

```yaml
simulation:
  duration: 300.0
  time_step: 0.1
  seed: 42
```

### 2) Environment

```yaml
environment:
  width: 50.0
  height: 50.0
  grid_resolution: 0.5
```

### 3) Agents

```yaml
agents:
  count: 500
  speed_range: [0.8, 1.8]
  radius_range: [0.2, 0.4]
  visibility_range: [3.0, 15.0]
  panic_threshold: 0.3
```

### 4) Motion model

```yaml
motion:
  model: hybrid
```

Available model values include: `sfm`, `rvo`, `pathfinding`, `hybrid`.

### 5) Hazards

```yaml
hazards:
  fire:
    enabled: true
  smoke:
    enabled: true
  exit_failures:
    enabled: false
```

### 6) Visualization

```yaml
visualization:
  enabled: true
  fps: 30
```

### 7) Analytics and outputs

```yaml
analytics:
  enabled: true
  export_csv: true
  csv_path: output/floorplan_analytics.csv
  compute_heatmaps: true

output:
  directory: output
```

## How The Recommendation System Works

At simulation finalize:

1. Reads CSV trends and computes evacuation phase behavior.
2. Uses density and panic heatmaps to identify hotspots.
3. Checks exit utilization imbalance.
4. Generates prioritized, location-specific floor-plan changes.
5. Draws a visual overlay with color-coded improvement markers.

Generated files:

- `floorplan_improvement_suggestions.md`
- `floorplan_improvement_overlay.png`

## Project Structure

```text
tragic2/
  main.py
  ui_app.py
  ui_runner.py
  config.yaml
  requirements.txt
  analyze_floorplan_yolo.py
  yolov8n.pt
  README.md

  src/
    __init__.py
    agent.py
    analytics.py
    environment.py
    egress_advisor.py
    floorplan_parser.py
    hazard_manager.py
    motion_models.py
    simulation_engine.py
    visualizer.py

  output/
    floorplan_analytics.csv
    agent_paths.png
    floorplan_improvement_suggestions.md
    floorplan_improvement_overlay.png
    heatmaps/
      density_heatmap.png
      panic_heatmap.png
```

## Performance Tips

For faster runs:

- Reduce agents (`--agents 100` for quick tests).
- Use larger `time_step` (for example `0.2`) with caution.
- Disable live visualization (`--no-viz`).
- Increase `grid_resolution` (coarser grid) if acceptable.

For higher fidelity:

- Use smaller `time_step`.
- Use smaller `grid_resolution`.
- Enable visuals and heatmaps.

## Troubleshooting

### Virtual environment activation issues

Use:

```bash
source .venv/bin/activate
```

### Missing dependencies

Reinstall:

```bash
pip install -r requirements.txt
```

### Slow simulation with many agents

Try:

- Lower agent count.
- Disable live rendering.
- Run in batch mode.

### No improvement report generated

Check:

- Simulation completed normally (not interrupted).
- Output directory is writable.
- Analytics CSV exists.

### Streamlit run exits with code 130

Exit code 130 often means user interruption (Ctrl+C). Re-run:

```bash
python ui_runner.py
```

## FAQ

### Does the improvement report generate automatically?

Yes. It is generated at the end of simulation finalize.

### Does Streamlit also show the report and overlay?

Yes. The Latest Outputs panel renders both automatically after a completed run.

### Can I run without a floorplan file?

Yes. You can run using `config.yaml` dimensions/exits/obstacles.

### Which floorplan format is best?

Use clear high-contrast images for quickest onboarding. DXF is better when CAD geometry is available and clean.

## Contributing

Suggested contribution areas:

- New motion models.
- Better route assignment strategies.
- Additional hazard types.
- More robust floorplan parsing and calibration.
- Better recommendation logic and evaluation metrics.

## License, Authors, Citation

Add your project-specific details here.

- License: add your chosen license.
- Authors: add team details.
- Citation: add publication or repository citation format.