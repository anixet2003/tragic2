"""
Main Application Entry Point
Command-line interface for running simulations
"""

import sys
import os
import yaml
import argparse
import hashlib
import copy
from pathlib import Path

from src.simulation_engine import SimulationEngine

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: Path) -> None:
    """Persist configuration to YAML."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = to_serializable_config(config)
    with open(config_path, 'w') as f:
        yaml.safe_dump(serializable_config, f, sort_keys=False)


def to_serializable_config(config: dict) -> dict:
    """Convert runtime config values (objects, numpy types) into YAML-safe primitives."""

    def _normalize(value):
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]

        # Serialize environment obstacle objects
        if hasattr(value, 'x') and hasattr(value, 'y') and hasattr(value, 'width') and hasattr(value, 'height'):
            return {
                'x': float(value.x),
                'y': float(value.y),
                'width': float(value.width),
                'height': float(value.height),
            }

        # Serialize exit objects
        if hasattr(value, 'id') and hasattr(value, 'position') and hasattr(value, 'width'):
            pos = value.position
            if hasattr(pos, 'tolist'):
                pos = pos.tolist()
            return {
                'id': int(value.id),
                'position': _normalize(pos),
                'width': float(value.width),
                'capacity': int(getattr(value, 'capacity', 100)),
            }

        # Normalize numpy scalars/arrays if present
        if hasattr(value, 'item') and callable(getattr(value, 'item', None)):
            try:
                return value.item()
            except Exception:
                pass

        if hasattr(value, 'tolist') and callable(getattr(value, 'tolist', None)):
            try:
                return value.tolist()
            except Exception:
                pass

        return value

    return _normalize(copy.deepcopy(config))


def build_generated_config_path(floorplan_path: str) -> Path:
    """Build a stable per-floorplan config filename."""
    floorplan = Path(floorplan_path)
    safe_stem = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in floorplan.stem).strip('_') or 'floorplan'
    digest = hashlib.sha1(str(floorplan.resolve()).encode('utf-8')).hexdigest()[:8]
    return Path('generated_configs') / f"{safe_stem}_{digest}.yaml"


def apply_streamlit_overrides(config: dict) -> None:
    output_dir = os.getenv("TRAGIC_OUTPUT_DIR")
    frame_output = os.getenv("TRAGIC_FRAME_OUTPUT")
    frame_stride = os.getenv("TRAGIC_FRAME_STRIDE")

    if output_dir:
        config.setdefault("output", {})
        config["output"]["directory"] = output_dir
        config.setdefault("analytics", {})
        config["analytics"]["csv_path"] = str(Path(output_dir) / "floorplan_analytics.csv")

    if frame_output:
        config.setdefault("visualization", {})
        config["visualization"]["enabled"] = True
        config["visualization"]["show_live_window"] = False
        config["visualization"]["capture_frames"] = True
        config["visualization"]["frame_output_path"] = frame_output
        if frame_stride:
            try:
                config["visualization"]["frame_stride"] = max(1, int(frame_stride))
            except ValueError:
                config["visualization"]["frame_stride"] = 1


def detect_floorplan_type(filepath: str) -> str:
    """Detect floorplan file type."""
    ext = Path(filepath).suffix.lower()
    if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        return 'image'
    return None


def auto_detect_scale(filepath: str, floorplan_type: str) -> float:
    """Auto-detect scale from floorplan."""
    if floorplan_type == 'image' and PIL_AVAILABLE:
        img = Image.open(filepath)
        width_px, height_px = img.size
        # Assume typical building is 50m, so estimate scale
        estimated_scale = max(width_px, height_px) / 50.0
        return estimated_scale
    return 10.0  # Default: 10 pixels per meter


def configure_from_floorplan(filepath: str, scale: float = None, 
                             agent_count: int = None, duration: float = None,
                             batch_mode: bool = False) -> dict:
    """Auto-configure simulation from floorplan file."""
    
    if not Path(filepath).exists():
        print(f"Error: Floorplan file not found: {filepath}")
        return None
    
    floorplan_type = detect_floorplan_type(filepath)
    
    if floorplan_type is None:
        print(f"Error: Unsupported file type. Use .png, .jpg, .jpeg, or .bmp")
        return None
    
    if floorplan_type == 'image' and not PIL_AVAILABLE:
        print("Error: Image support requires Pillow. Install with: pip install Pillow")
        return None
    
    print(f"[+] Detected {floorplan_type.upper()} floorplan")
    
    # Auto-detect or use provided scale
    if scale is None:
        scale = auto_detect_scale(filepath, floorplan_type)
        print(f"[+] Auto-detected scale: {scale:.2f} {'pixels' if floorplan_type == 'image' else 'units'} per meter")
    
    # Calculate dimensions
    img = Image.open(filepath)
    width_px, height_px = img.size
    width = width_px / scale
    height = height_px / scale
    print(f"[+] Image size: {width_px}x{height_px} pixels -> {width:.1f}x{height:.1f} meters")
    
    # Configuration for agents and duration
    area = width * height
    suggested_agents = int(area * 0.7)
    
    print(f"\n{'='*60}")
    print("SIMULATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Building area: {area:.1f} m²")
    print(f"Suggested density: 0.7 agents/m² = {suggested_agents} agents")
    
    if agent_count is None and not batch_mode:
        # Interactive mode - ask user for number of agents
        try:
            user_input = input(f"\nNumber of agents [{suggested_agents}]: ").strip()
            if user_input:
                agent_count = int(user_input)
            else:
                agent_count = suggested_agents
        except (ValueError, EOFError, KeyboardInterrupt):
            agent_count = suggested_agents
    elif agent_count is None:
        agent_count = suggested_agents
    
    print(f"[+] Using {agent_count} agents")
    
    if duration is None and not batch_mode:
        # Interactive mode - ask user for simulation duration
        try:
            user_input = input(f"Simulation duration in seconds [300]: ").strip()
            if user_input:
                duration = float(user_input)
            else:
                duration = 300.0
        except (ValueError, EOFError, KeyboardInterrupt):
            duration = 300.0
    elif duration is None:
        duration = 300.0
    
    print(f"[+] Simulation duration: {duration:.0f} seconds")
    
    # Optional: adjust scale interactively
    if not batch_mode and scale is not None:
        try:
            user_input = input(f"\nAdjust scale? Current: {scale:.1f} pixels/meter [press Enter to keep]: ").strip()
            if user_input:
                new_scale = float(user_input)
                # Recalculate dimensions with new scale
                if floorplan_type == 'image':
                    width = width_px / new_scale
                    height = height_px / new_scale
                    scale = new_scale
                    print(f"[+] New dimensions: {width:.1f}x{height:.1f} meters")
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
    
    print(f"{'='*60}\n")
    
    # Parse floorplan to extract obstacles and exits
    print("Parsing floorplan geometry...")
    from src.floorplan_parser import ImageParser
    
    obstacles_list = []
    exits_list = []
    
    try:
        parser = ImageParser(filepath, scale)
        obstacles_list, exits_list = parser.parse()
        
        print(f"[+] Parsed {len(obstacles_list)} obstacles and {len(exits_list)} exits from floorplan")
    except Exception as e:
        print(f"Warning: Failed to parse floorplan geometry: {e}")
        print("Continuing with empty environment...")
    
    # Generate configuration
    config = {
        'floorplan_path': filepath,  # Store floorplan path for visualization overlay
        'simulation': {
            'duration': duration,
            'time_step': 0.1,
            'seed': 42
        },
        'environment': {
            'width': width,
            'height': height,
            'grid_resolution': 0.5,
            'floorplan_obstacles': obstacles_list,  # Pass parsed obstacles
            'floorplan_exits': exits_list  # Pass parsed exits
        },
        'agents': {
            'count': agent_count,
            'speed_range': [1.0, 1.6],
            'radius_range': [0.25, 0.35],
            'visibility_range': [max(width, height) * 1.5, max(width, height) * 2.0],  # Scale with building size
            'panic_threshold': 0.3
        },
        'motion': {
            'model': 'hybrid',
            'sfm': {
                'relaxation_time': 0.5,
                'agent_strength': 2000.0,
                'agent_range': 0.08,
                'wall_strength': 2000.0,
                'wall_range': 0.08,
                'noise_factor': 0.1
            },
            'rvo': {
                'time_horizon': 2.0,
                'neighbor_dist': 5.0,
                'max_neighbors': 10
            },
            'pathfinding': {
                'replan_interval': 1.5,
                'congestion_weight': 0.4,
                'hazard_weight': 0.6
            }
        },
        'hazards': {
            'fire': {
                'enabled': True,
                'start_time': 30.0,
                'ignition_points': [[width/2, height/2]],
                'spread_rate': 0.08,
                'damage_rate': 0.15,
                'growth_rate': 2.0
            },
            'smoke': {
                'enabled': True,
                'diffusion_rate': 0.4,
                'visibility_reduction': 0.85,
                'damage_rate': 0.03
            },
            'exit_failures': {
                'enabled': False,
                'failure_times': [],
                'failure_exits': []
            }
        },
        'exits': {
            'positions': [
                [width * 0.1, height * 0.5],
                [width * 0.9, height * 0.5],
                [width * 0.5, height * 0.1],
                [width * 0.5, height * 0.9]
            ],
            'widths': [2.5, 2.5, 2.5, 2.5],
            'capacities': [200, 200, 200, 200]
        },
        'obstacles': {
            'rectangles': []
        },
        'visualization': {
            'enabled': True,
            'fps': 30,
            'show_trajectories': False,
            'show_panic_levels': True,
            'show_hazards': True
        },
        'analytics': {
            'enabled': True,
            'sampling_rate': 0.5,
            'export_csv': True,
            'csv_path': 'output/floorplan_analytics.csv',
            'compute_heatmaps': True,
            'bottleneck_threshold': 5.0
        },
        'output': {
            'directory': 'output'
        }
    }
    
    print(f"✓ Configuration complete!\n")
    
    return config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Crowd Simulation & Multi-Hazard Evacuation System',
        epilog='Examples:\n'
               '  python main.py myfloorplan.png\n'
               '  python main.py bc.jpeg --scale 10 --agents 100\n'
               '  python main.py --config generated_configs/bc_1234abcd.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional argument - just pass your floorplan file directly!
    parser.add_argument(
        'floorplan',
        type=str,
        nargs='?',
        help='Path to your floorplan file (PNG, JPG, JPEG, or BMP)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Run from a previously generated config file'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        help='Optional output path for generated config when using a floorplan'
    )

    parser.add_argument(
        '--scale',
        type=float,
        help='Floorplan scale in pixels/meter (will be prompted if not specified)'
    )
    parser.add_argument(
        '--agents',
        type=int,
        help='Number of agents (will be prompted if not specified)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        help='Simulation duration in seconds (will be prompted if not specified)'
    )
    parser.add_argument(
        '--time-step',
        type=float,
        help='Simulation time step in seconds'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization (faster)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode - skip interactive prompts, use defaults'
    )
    
    args = parser.parse_args()

    if args.floorplan and args.config:
        print("Error: Use either a floorplan path or --config, not both.")
        sys.exit(1)

    if not args.floorplan and not args.config:
        print("Error: Provide a floorplan path or --config.")
        print("Usage: python main.py <floorplan.(png|jpg|jpeg|bmp)> [--scale ... --agents ... --duration ...]")
        print("   or: python main.py --config generated_configs/<name>.yaml")
        sys.exit(1)

    if args.floorplan:
        print(f"Loading floorplan: {args.floorplan}\n")
        config = configure_from_floorplan(
            args.floorplan,
            scale=args.scale,
            agent_count=args.agents,
            duration=args.duration,
            batch_mode=args.batch
        )
        if config is None:
            print("Error: Failed to configure from floorplan.")
            sys.exit(1)
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        print(f"Loading config: {config_path}\n")
        config = load_config(str(config_path))

        if args.agents:
            config.setdefault('agents', {})
            config['agents']['count'] = args.agents

        if args.duration:
            config.setdefault('simulation', {})
            config['simulation']['duration'] = args.duration
    
    if args.no_viz:
        config['visualization']['enabled'] = False

    if args.time_step:
        config['simulation']['time_step'] = args.time_step

    if args.floorplan:
        generated_config_path = Path(args.save_config) if args.save_config else build_generated_config_path(args.floorplan)
        save_config(config, generated_config_path)
        print(f"[+] Saved generated config: {generated_config_path}")
        print("    You can rerun without the floorplan using:")
        print(f"    python main.py --config {generated_config_path}")

    apply_streamlit_overrides(config)
    
    # Create output directory
    output_dir = Path(config['output']['directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'heatmaps').mkdir(exist_ok=True)
    
    # Initialize and run simulation
    print("Initializing simulation engine...")
    engine = SimulationEngine(config)
    
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
