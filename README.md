# TRAGIC - TRaffic Analysis with Generative Intelligence and Crowd simulation

A sophisticated crowd simulation and evacuation system that models realistic human behavior, hazard propagation, and emergency scenarios using advanced motion models and deep learning-based floorplan analysis.

## 🎯 Overview

TRAGIC is a comprehensive simulation framework designed to:
- Simulate large-scale crowd dynamics and evacuation scenarios
- Model realistic human behavior including panic, social forces, and collision avoidance
- Analyze emergency situations with hazards like fire, smoke, and exit failures
- Process real-world floorplans using YOLOv8 deep learning
- Generate detailed analytics, heatmaps, and visualizations
- Support multiple motion models: Social Force Model (SFM), Reciprocal Velocity Obstacles (RVO), A* Pathfinding, and hybrid approaches

## ✨ Key Features

### 🚶 Agent Simulation
- **Realistic Behavior**: Agents with individual speeds, visibility ranges, and panic responses
- **Panic Dynamics**: Panic spreads through crowds based on proximity and hazard exposure
- **Social Forces**: Natural movement patterns considering personal space and social interactions
- **Collision Avoidance**: RVO-based obstacle and agent avoidance

### 🔥 Hazard Modeling
- **Fire Simulation**: Dynamic fire spread with ignition points and growth rates
- **Smoke Diffusion**: Realistic smoke propagation affecting visibility and agent health
- **Exit Failures**: Simulate emergency scenarios where exits become blocked or unusable
- **Damage Systems**: Health tracking for agents exposed to hazards

### 🗺️ Floorplan Analysis
- **YOLOv8 Detection**: Deep learning-based detection of walls, doors, windows, and exits
- **Multiple Formats**: Support for DXF (CAD) and image formats (PNG, JPG, JPEG)
- **Auto-Configuration**: Automatic scale detection and environment setup
- **Grid-Based Navigation**: Efficient spatial representation for pathfinding

### 📊 Analytics & Visualization
- **Real-time Visualization**: Animated simulation with agent trails and hazard overlays
- **Heatmap Generation**: Density, panic, and evacuation flow heatmaps
- **Statistical Analysis**: CSV export with time-series data on evacuations, casualties, and panic levels

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Virtual Environment**: Recommended for dependency isolation

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/tragic
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate on macOS/Linux
   source .venv/bin/activate

   # Activate on Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**:

   **Option A - Using pip** (traditional):
   ```bash
   pip install -r requirements.txt
   ```

   **Option B - Using uv** (faster, recommended):
   ```bash
   # Install uv first (macOS/Linux)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env

   # Install dependencies with uv
   uv pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python main.py --help
   ```

## 📖 Usage

### Basic Simulation

Run a simulation with default configuration:
```bash
python main.py
```

### Floorplan-Based Simulation

Analyze and simulate using a floorplan image:
```bash
python main.py path/to/floorplan.jpg
```

Example with the provided floorplan:
```bash
python main.py bc.jpeg
```

### Configuration Options

Customize simulation parameters via command-line arguments:
```bash
python main.py path/to/floorplan.jpg \
    --agents 500 \
    --duration 300 \
    --scale 10.0 \
    --config custom_config.yaml
```

**Available Arguments**:
- `floorplan`: Path to floorplan file (PNG, JPG, JPEG, or DXF)
- `--config`: Path to custom YAML configuration file (default: `config.yaml`)
- `--scale`: Scale in pixels/units per meter (default: auto-detect)
- `--agents`: Number of agents (default: from config)
- `--duration`: Simulation duration in seconds (default: from config)
- `--no-viz`: Disable visualization (faster)
- `--batch`: Run in batch mode without interactive prompts

### Floorplan Analysis Only

Analyze a floorplan without running a simulation:
```bash
python analyze_floorplan_yolo.py path/to/floorplan.jpg
```

## ⚙️ Configuration

The simulation behavior is controlled via `config.yaml`. Key configuration sections:

### Simulation Settings
```yaml
simulation:
  duration: 300.0      # Total simulation time (seconds)
  time_step: 0.1       # Physics time step (seconds)
  seed: 42             # Random seed for reproducibility
```

### Environment
```yaml
environment:
  width: 50.0          # Environment width (meters)
  height: 50.0         # Environment height (meters)
  grid_resolution: 0.5 # Grid cell size (meters)
```

### Agents
```yaml
agents:
  count: 500
  speed_range: [0.8, 1.8]      # Min/max speed (m/s)
  radius_range: [0.2, 0.4]     # Agent size range (meters)
  panic_threshold: 0.3         # Panic activation threshold
```

### Motion Models
```yaml
motion:
  model: "hybrid"              # Options: sfm, rvo, pathfinding, hybrid
  sfm:
    relaxation_time: 0.5
    agent_strength: 2000.0
    wall_strength: 2000.0
  rvo:
    time_horizon: 2.0
    neighbor_dist: 5.0
  pathfinding:
    congestion_weight: 0.3
```

### Hazards
```yaml
hazards:
  fire:
    enabled: true
    start_time: 30.0
    ignition_points: [[25.0, 25.0]]
    spread_rate: 0.05
  smoke:
    enabled: true
    diffusion_rate: 0.3
  exit_failures:
    enabled: true
    failure_times: [60.0, 120.0]
```

### Visualization
```yaml
visualization:
  enabled: true
  fps: 30
  show_trajectories: true
```

## 📁 Project Structure

```
tragic/
├── main.py                          # Main entry point
├── analyze_floorplan_yolo.py        # Standalone floorplan analyzer
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── yolov8n.pt                       # YOLOv8 model weights
├── README.md                        # This file
│
├── src/                             # Core source modules
│   ├── __init__.py
│   ├── agent.py                     # Agent behavior and state
│   ├── analytics.py                 # Data collection and analysis
│   ├── environment.py               # Spatial environment and grid
│   ├── floorplan_parser.py          # DXF and image parsing
│   ├── hazard_manager.py            # Fire, smoke, and hazards
│   ├── motion_models.py             # SFM, RVO, pathfinding
│   ├── simulation_engine.py         # Main simulation loop
│   └── visualizer.py                # Rendering and heatmap export
│
└── output/                          # Generated outputs
    ├── analytics.csv                # Analytics data (path configurable)
    ├── agent_paths.png              # Agent movement paths
    └── heatmaps/                    # Generated heatmap images
        ├── density_heatmap.png
        └── panic_heatmap.png
```

## 🔧 Core Components

### 1. Simulation Engine (`src/simulation_engine.py`)
- Orchestrates the entire simulation loop
- Manages agents, environment, hazards, and analytics
- Handles time stepping and event coordination
- Integrates motion models and collision detection

### 2. Agent System (`src/agent.py`)
- Individual agent behavior and state management
- Panic dynamics and health tracking
- Goal selection and path following
- Exit assignment and evacuation logic

### 3. Motion Models (`src/motion_models.py`)
- **Social Force Model (SFM)**: Physics-based pedestrian dynamics
- **RVO (Reciprocal Velocity Obstacles)**: Collision-free velocity selection
- **A* Pathfinding**: Grid-based optimal path planning
- **Hybrid Model**: Combines pathfinding with local collision avoidance

### 4. Hazard Manager (`src/hazard_manager.py`)
- Fire ignition, spread, and growth simulation
- Smoke diffusion using grid-based propagation
- Exit failure management
- Agent damage calculation

### 5. Environment (`src/environment.py`)
- Spatial grid representation
- Exit and obstacle management
- Distance field computation for navigation
- Boundary and collision detection

### 6. Floorplan Parser (`src/floorplan_parser.py`)
- DXF file parsing with `ezdxf`
- Image processing with PIL/OpenCV
- Automatic scale detection and calibration

### 7. Visualizer (`src/visualizer.py`)
- Real-time matplotlib rendering
- Agent trail rendering
- Hazard overlay visualization
- Heatmap generation

### 8. Analytics (`src/analytics.py`)
- Time-series data collection
- Evacuation statistics
- Panic and casualty tracking
- CSV export for post-analysis

## 📊 Output Files

After running a simulation, you'll find:

### 1. **Analytics CSV** (`output/analytics.csv`)
Columns include:
- `Time`: Simulation timestamp
- `Active_Agents`: Current agent count
- `Evacuated`: Number of evacuated agents
- `Deceased`: Number of casualties
- `Avg_Panic`: Average panic level
- `Avg_Speed`: Average agent speed

### 2. **Heatmaps** (`output/heatmaps/`)
- `density_heatmap.png`: Agent density over time
- `panic_heatmap.png`: Panic level distribution

### 3. **Agent Paths** (`output/agent_paths.png`)
- Combined visualization of all agent trajectories

## 🧪 Example Workflows

### Workflow 1: Quick Test Simulation
```bash
# Run with default settings
python main.py

# Check output
ls -lh output/
cat output/analytics.csv
```

### Workflow 2: Custom Building Analysis
```bash
# Analyze a custom floorplan
python main.py my_building.jpg --agents 1000 --duration 600

# View results
open output/heatmaps/density_heatmap.png
```

### Workflow 3: Batch Processing
```bash
# Run multiple scenarios without visualization
for agents in 100 500 1000; do
    python main.py building.jpg --agents $agents --batch
    mv output/analytics.csv output/analytics_${agents}.csv
done
```

### Workflow 4: Configuration Experiments
```bash
# Create custom config
cp config.yaml config_high_panic.yaml
# Edit config_high_panic.yaml to increase panic parameters

# Run with custom config
python main.py building.jpg --config config_high_panic.yaml
```

## 📦 Dependencies

Core libraries (from `requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.2.6 | Numerical computations |
| matplotlib | 3.10.8 | Visualization and plotting |
| opencv-python | 4.12.0.88 | Image processing |
| pillow | 12.0.0 | Image loading and manipulation |
| scipy | 1.16.3 | Scientific algorithms |
| torch | 2.9.1 | Deep learning framework |
| torchvision | 0.24.1 | Computer vision models |
| ultralytics | 8.3.239 | YOLOv8 object detection |
| ezdxf | 1.4.3 | DXF CAD file parsing |
| pyyaml | 6.0.3 | YAML configuration |
| polars | 1.36.1 | Fast dataframe processing |
| networkx | 3.6.1 | Graph algorithms for pathfinding |

## 🎓 Technical Details

### Motion Model: Social Force Model (SFM)

The Social Force Model simulates pedestrian dynamics using forces:
- **Desired force**: Moves agent toward goal
- **Repulsive forces**: Avoids other agents and obstacles
- **Random noise**: Adds natural movement variation

$$\mathbf{F}_i = \mathbf{F}_i^{goal} + \sum_j \mathbf{F}_{ij}^{agent} + \sum_w \mathbf{F}_{iw}^{wall} + \mathbf{F}_i^{noise}$$

### RVO (Reciprocal Velocity Obstacles)

Collision-free velocity selection based on:
- Time horizon for collision prediction
- Velocity obstacles from nearby agents
- Optimal velocity selection within constraints

### Panic Dynamics

Panic spreads through the crowd:
```
panic_increase = panic_spread_rate × nearby_panic_level
panic_level = min(max_panic, current_panic + hazard_panic + spread_panic)
```

Affects agent behavior:
- Increased movement speed (up to 50% faster)
- Reduced decision-making quality
- Higher likelihood of congestion

## 🐛 Troubleshooting

### Issue: "Permission denied" when activating virtual environment
**Solution**: Use `source` command:
```bash
source .venv/bin/activate
```

### Issue: YOLOv8 model not found
**Solution**: The model will auto-download on first run. Ensure internet connection.

### Issue: DXF files not loading
**Solution**: Install ezdxf:
```bash
pip install ezdxf
```

### Issue: Simulation runs too slow
**Solutions**:
- Reduce agent count: `--agents 100`
- Increase time step in `config.yaml`: `time_step: 0.2`
- Disable visualization: `--no-viz`
- Use lower resolution: `grid_resolution: 1.0`

## 🔬 Research Applications

TRAGIC is suitable for:
- **Emergency Planning**: Evaluate building evacuation strategies
- **Crowd Management**: Analyze high-density event scenarios
- **Safety Assessment**: Test fire safety and exit capacity
- **Urban Design**: Optimize public space layouts
- **Training**: Generate realistic emergency scenarios

## 📝 Citation

If you use TRAGIC in your research, please cite:
```
TRAGIC - TRaffic Analysis with Generative Intelligence and Crowd simulation
GitHub: [Your Repository URL]
Year: 2025
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional motion models (e.g., cellular automata)
- More hazard types (structural damage, flooding)
- Multi-floor building support
- Real-time agent decision-making with AI
- Integration with building information models (BIM)

## 📄 License

[Add your license information here]

## 👥 Authors

[Add author information here]

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics for object detection
- Social Force Model by Helbing et al.
- RVO library concepts from UNC-Chapel Hill
- Community contributions and feedback

---

**Version**: 1.0.0  
**Last Updated**: December 17, 2025  
**Status**: Active Development

For questions, issues, or suggestions, please open an issue on the project repository.