"""One-command launcher. Creates only the GUI venv; never installs GPU drivers."""
from pathlib import Path
import subprocess
import sys

if sys.version_info < (3, 12):
    raise SystemExit("Install Python 3.12 or newer to run the desktop frontend.")
root = Path(__file__).resolve().parent
environment = root / ".venv"
python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if not python.is_file():
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
check = subprocess.run([str(python), "-c", "import glm_desktop, slint"], capture_output=True)
if check.returncode:
    subprocess.run([str(python), "-m", "pip", "install", "-e", str(root)], check=True)
raise SystemExit(subprocess.call([str(python), "-m", "glm_desktop.app", *sys.argv[1:]]))
