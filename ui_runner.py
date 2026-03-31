"""
Helper to launch the Streamlit UI using the current Python environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
	app_path = Path(__file__).parent / "ui_app.py"
	if not app_path.exists():
		print("Error: ui_app.py not found.")
		return 1

	command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
	try:
		return subprocess.call(command)
	except KeyboardInterrupt:
		print("Streamlit UI stopped by user.")
		return 130


if __name__ == "__main__":
	raise SystemExit(main())
