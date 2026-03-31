"""
Streamlit UI for running TRAGIC simulations.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import psutil
import streamlit as st
import yaml

try:
	from streamlit_autorefresh import st_autorefresh
except Exception:
	st_autorefresh = None

from main import configure_from_floorplan, load_config
from src.simulation_engine import SimulationEngine


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OUTPUT_DIR = Path("output")


def load_base_config() -> dict:
	if DEFAULT_CONFIG_PATH.exists():
		return load_config(str(DEFAULT_CONFIG_PATH))
	return {
		"simulation": {"duration": 300.0, "time_step": 0.1, "seed": 42},
		"environment": {"width": 50.0, "height": 50.0, "grid_resolution": 0.5},
		"agents": {
			"count": 200,
			"speed_range": [0.8, 1.8],
			"radius_range": [0.2, 0.4],
			"visibility_range": [3.0, 15.0],
			"panic_threshold": 0.3,
		},
		"motion": {"model": "hybrid"},
		"hazards": {},
		"exits": {"positions": [], "widths": [], "capacities": []},
		"obstacles": {"rectangles": []},
		"visualization": {"enabled": False, "fps": 30},
		"analytics": {
			"enabled": True,
			"sampling_rate": 0.5,
			"export_csv": True,
			"csv_path": "output/analytics.csv",
			"compute_heatmaps": True,
			"bottleneck_threshold": 4.0,
		},
		"output": {"directory": "output"},
	}


def write_uploaded_file(uploaded_file) -> Optional[Path]:
	if uploaded_file is None:
		return None

	suffix = Path(uploaded_file.name).suffix
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp.write(uploaded_file.getbuffer())
		return Path(tmp.name)


def ensure_output_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)
	(path / "heatmaps").mkdir(exist_ok=True)
	(path / "live").mkdir(exist_ok=True)


def build_run_output_dir(base_dir: Path) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return base_dir / f"run_{timestamp}"


def apply_overrides(config: dict, agent_count: int, duration: float, time_step: float) -> None:
	config["agents"]["count"] = agent_count
	config["simulation"]["duration"] = duration
	config["simulation"]["time_step"] = time_step


def apply_hazard_toggles(config: dict, fire: bool, smoke: bool, exit_failures: bool) -> None:
	config.setdefault("hazards", {})
	config["hazards"].setdefault("fire", {})
	config["hazards"]["fire"]["enabled"] = fire
	config["hazards"].setdefault("smoke", {})
	config["hazards"]["smoke"]["enabled"] = smoke
	config["hazards"].setdefault("exit_failures", {})
	config["hazards"]["exit_failures"]["enabled"] = exit_failures


def run_simulation(config: dict, output_dir: Path) -> None:
	config.setdefault("output", {})
	config["output"]["directory"] = str(output_dir)
	config.setdefault("analytics", {})
	config["analytics"]["csv_path"] = str(output_dir / "floorplan_analytics.csv")

	ensure_output_dir(output_dir)
	engine = SimulationEngine(config)
	engine.run()


def write_config_file(config: dict, output_path: Path) -> None:
	output_path.write_text(yaml.safe_dump(config, sort_keys=False))


def launch_background_simulation(
	*,
	output_dir: Path,
	frame_path: Path,
	frame_stride: int,
	floorplan_path: Optional[Path],
	scale_value: Optional[float],
	agent_count: int,
	duration: float,
	time_step: float,
	config_text: Optional[str],
	use_overrides: bool,
	fire_enabled: bool,
	smoke_enabled: bool,
	exit_failures_enabled: bool,
) -> int:
	command = [sys.executable, "main.py"]

	if floorplan_path is not None:
		command.append(str(floorplan_path))
		if scale_value is not None:
			command.extend(["--scale", str(scale_value)])
		command.extend(["--agents", str(agent_count)])
		command.extend(["--duration", str(duration)])
		command.extend(["--time-step", str(time_step)])
		command.append("--batch")
	else:
		config_path = output_dir / "run_config.yaml"
		if config_text:
			config_obj, error_message = parse_config_with_error(config_text)
			if config_obj is None:
				raise ValueError(error_message or "Invalid config YAML.")
			if use_overrides:
				apply_overrides(config_obj, int(agent_count), float(duration), float(time_step))
				apply_hazard_toggles(config_obj, fire_enabled, smoke_enabled, exit_failures_enabled)
			write_config_file(config_obj, config_path)
		else:
			config_obj = load_base_config()
			if use_overrides:
				apply_overrides(config_obj, int(agent_count), float(duration), float(time_step))
				apply_hazard_toggles(config_obj, fire_enabled, smoke_enabled, exit_failures_enabled)
			write_config_file(config_obj, config_path)
		command.extend(["--config", str(config_path)])

	child_env = dict(os.environ)
	child_env["TRAGIC_OUTPUT_DIR"] = str(output_dir)
	child_env["TRAGIC_FRAME_OUTPUT"] = str(frame_path)
	child_env["TRAGIC_FRAME_STRIDE"] = str(frame_stride)

	process = subprocess.Popen(command, env=child_env, cwd=Path(__file__).parent)
	return process.pid


def is_process_running(pid: Optional[int]) -> bool:
	if pid is None:
		return False
	try:
		process = psutil.Process(pid)
		return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
	except psutil.NoSuchProcess:
		return False


def stop_process(pid: Optional[int]) -> None:
	if pid is None:
		return
	try:
		process = psutil.Process(pid)
		process.terminate()
		process.wait(timeout=5)
	except (psutil.NoSuchProcess, psutil.TimeoutExpired):
		return


def read_csv_rows(csv_path: Path, max_rows: int = 200) -> list:
	if not csv_path.exists():
		return []

	with csv_path.open("r", newline="") as handle:
		reader = csv.reader(handle)
		rows = list(reader)

	if not rows:
		return []

	headers = rows[0]
	data_rows = rows[1:][:max_rows]
	return [dict(zip(headers, row)) for row in data_rows]


def parse_config_with_error(config_text: str) -> Tuple[Optional[dict], Optional[str]]:
	try:
		parsed = yaml.safe_load(config_text)
	except yaml.YAMLError as exc:
		return None, format_yaml_error(exc, config_text)
	if not isinstance(parsed, dict):
		return None, "Config root must be a YAML mapping (key/value dictionary)."
	return parsed, None


def format_yaml_error(exc: yaml.YAMLError, config_text: str) -> str:
	marker = getattr(exc, "problem_mark", None)
	if marker is None:
		return "Invalid YAML."

	line_no = marker.line + 1
	col_no = marker.column + 1
	lines = config_text.splitlines()
	bad_line = lines[line_no - 1] if 0 <= line_no - 1 < len(lines) else ""
	caret = " " * (col_no - 1) + "^"
	return f"YAML error at line {line_no}, column {col_no}:\n{bad_line}\n{caret}"


def main() -> None:
	st.set_page_config(page_title="TRAGIC Simulation", layout="wide")
	st.title("TRAGIC Simulation Runner")

	base_config = load_base_config()

	with st.sidebar:
		st.header("Run Settings")
		source_mode = st.radio("Configuration source", ["Floorplan file", "config.yaml"], index=0)
		use_overrides = True
		if source_mode == "config.yaml":
			use_overrides = st.checkbox("Apply sidebar overrides", value=True)
		agent_count = st.number_input(
			"Agents",
			min_value=1,
			max_value=5000,
			value=int(base_config["agents"]["count"]),
			step=10,
		)
		duration = st.number_input(
			"Duration (seconds)",
			min_value=5.0,
			max_value=3600.0,
			value=float(base_config["simulation"]["duration"]),
			step=10.0,
		)
		time_step = st.number_input(
			"Time step (seconds)",
			min_value=0.01,
			max_value=5.0,
			value=min(max(float(base_config["simulation"].get("time_step", 0.1)), 0.01), 5.0),
			step=0.01,
		)

		st.subheader("Hazards")
		fire_enabled = st.checkbox("Fire", value=True)
		smoke_enabled = st.checkbox("Smoke", value=True)
		exit_failures_enabled = st.checkbox("Exit failures", value=False)

		st.subheader("Visualization")
		enable_visuals = st.checkbox("Enable matplotlib visuals", value=False)
		live_preview = st.checkbox("Live preview in Streamlit (beta)", value=False)
		frame_stride = 1
		if live_preview:
			frame_stride = st.number_input(
				"Frame stride (capture every Nth frame)",
				min_value=1,
				max_value=30,
				value=1,
				step=1,
			)

	floorplan_path: Optional[Path] = None
	scale_value: Optional[float] = None

	if source_mode == "Floorplan file":
		st.subheader("Floorplan")
		uploaded = st.file_uploader("Upload a floorplan image or DXF", type=["png", "jpg", "jpeg", "bmp", "dxf"])
		scale_value = st.number_input("Scale (pixels per meter)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
		floorplan_path = write_uploaded_file(uploaded)
		if uploaded is not None:
			st.caption(f"Using uploaded file: {uploaded.name}")
	else:
		st.subheader("Config")
		default_yaml = yaml.safe_dump(base_config, sort_keys=False)
		config_text = st.text_area(
			"Edit config.yaml",
			value=st.session_state.get("config_text", default_yaml),
			height=360,
		)
		col_left, col_middle, col_right = st.columns([1, 1, 3])
		with col_left:
			save_config = st.button("Save config.yaml")
		with col_middle:
			reset_config = st.button("Reset to default")
		if reset_config:
			st.session_state["config_text"] = default_yaml
			st.success("Editor reset to default config.")
		if save_config:
			parsed_config, error_message = parse_config_with_error(config_text)
			if parsed_config is None:
				st.error("Config YAML is invalid.")
				if error_message:
					st.code(error_message)
			else:
				DEFAULT_CONFIG_PATH.write_text(config_text)
				st.session_state["config_text"] = config_text
				st.success("config.yaml updated.")

	run_clicked = st.button("Run simulation", type="primary")

	if run_clicked:
		if source_mode == "Floorplan file" and floorplan_path is None:
			st.error("Please upload a floorplan file to continue.")
			return

		with st.spinner("Running simulation. This can take a while for large agent counts."):
			output_dir = build_run_output_dir(DEFAULT_OUTPUT_DIR)
			frame_path = output_dir / "live" / "frame.png"
			ensure_output_dir(output_dir)

			if live_preview:
				config_text = st.session_state.get("config_text") if source_mode == "config.yaml" else None
				try:
					pid = launch_background_simulation(
						output_dir=output_dir,
						frame_path=frame_path,
						frame_stride=int(frame_stride),
						floorplan_path=floorplan_path if source_mode == "Floorplan file" else None,
						scale_value=scale_value,
						agent_count=int(agent_count),
						duration=float(duration),
						time_step=float(time_step),
						config_text=config_text,
						use_overrides=use_overrides,
						fire_enabled=fire_enabled,
						smoke_enabled=smoke_enabled,
						exit_failures_enabled=exit_failures_enabled,
					)
				except ValueError as exc:
					st.error(str(exc))
					return

				st.session_state["live_pid"] = pid
				st.session_state["live_output_dir"] = str(output_dir)
				st.session_state["last_output_dir"] = None
				st.success("Simulation started in the background.")
			else:
				if source_mode == "Floorplan file":
					config = configure_from_floorplan(
						str(floorplan_path),
						scale=scale_value,
						agent_count=int(agent_count),
						duration=float(duration),
						batch_mode=True,
					)
					if config is None:
						st.error("Failed to configure from floorplan. See logs for details.")
						return
					apply_overrides(config, int(agent_count), float(duration), float(time_step))
				else:
					config_text = st.session_state.get("config_text")
					if config_text:
						config, error_message = parse_config_with_error(config_text)
						if config is None:
							st.error("Config YAML is invalid. Fix the errors and try again.")
							if error_message:
								st.code(error_message)
							return
					else:
						config = load_base_config()
					if use_overrides:
						apply_overrides(config, int(agent_count), float(duration), float(time_step))

				if use_overrides or source_mode == "Floorplan file":
					config.setdefault("visualization", {})
					config["visualization"]["enabled"] = bool(enable_visuals)
					apply_hazard_toggles(config, fire_enabled, smoke_enabled, exit_failures_enabled)

				run_simulation(config, output_dir)
				st.session_state["last_output_dir"] = str(output_dir)
				st.success("Simulation complete.")

	live_output_dir = st.session_state.get("live_output_dir")
	live_pid = st.session_state.get("live_pid")
	if live_output_dir:
		st.subheader("Live Preview")
		output_dir = Path(live_output_dir)
		frame_path = output_dir / "live" / "frame.png"
		running = is_process_running(live_pid)
		if running and st_autorefresh:
			st_autorefresh(interval=1000, key="live_preview")
		elif running:
			st.caption("Auto-refresh unavailable. Click refresh in the browser to update.")

		if frame_path.exists():
			st.image(str(frame_path), caption="Live simulation frame", use_container_width=True)
		else:
			st.info("Waiting for the first frame...")

		col_left, col_right = st.columns([1, 3])
		with col_left:
			if running and st.button("Stop live simulation"):
				stop_process(live_pid)
				st.session_state["live_pid"] = None
				st.session_state["last_output_dir"] = live_output_dir
				st.success("Simulation stopped.")

		if not running and live_pid:
			st.session_state["live_pid"] = None
			st.session_state["last_output_dir"] = live_output_dir
			st.success("Simulation finished.")

	last_output_dir = st.session_state.get("last_output_dir")
	if last_output_dir:
		st.subheader("Latest Outputs")
		output_dir = Path(last_output_dir)
		csv_rows = read_csv_rows(output_dir / "floorplan_analytics.csv")

		if csv_rows:
			st.write("Analytics preview")
			st.dataframe(csv_rows, use_container_width=True)
		else:
			st.info("No analytics CSV found yet.")

		heatmap_density = output_dir / "heatmaps" / "density_heatmap.png"
		heatmap_panic = output_dir / "heatmaps" / "panic_heatmap.png"
		agent_paths = output_dir / "agent_paths.png"

		cols = st.columns(2)
		with cols[0]:
			if heatmap_density.exists():
				st.image(str(heatmap_density), caption="Density heatmap", use_container_width=True)
			else:
				st.caption("Density heatmap not available.")
		with cols[1]:
			if heatmap_panic.exists():
				st.image(str(heatmap_panic), caption="Panic heatmap", use_container_width=True)
			else:
				st.caption("Panic heatmap not available.")

		if agent_paths.exists():
			st.image(str(agent_paths), caption="Agent movement paths", use_container_width=True)
		else:
			st.caption("Agent paths image not available.")


if __name__ == "__main__":
	main()
