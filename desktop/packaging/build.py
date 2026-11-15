"""Build a self-contained GUI; GPU runtime stays separate."""
from pathlib import Path
import subprocess
import sys
from collect_notices import collect
root=Path(__file__).resolve().parents[1]
sep=";" if sys.platform=="win32" else ":"
notices = collect(root.parent / "build" / "package-notices")
subprocess.run([sys.executable,"-m","PyInstaller","--noconfirm","--clean","--windowed","--name","Distributed-Compute","--collect-all","slint","--collect-all","keyring","--paths",str(root),"--add-data",str(root/"glm_desktop")+sep+"glm_desktop","--add-data",str(root.parent/"training")+sep+"training","--add-data",str(root.parent/"hivemind-tools")+sep+"hivemind-tools","--add-data",str(notices)+sep+"third-party/package-notices","--add-data",str(root.parent/"THIRD_PARTY_LICENSES")+sep+"third-party/licenses","--add-data",str(root.parent/"THIRD_PARTY_NOTICES.md")+sep+"third-party","--add-data",str(root.parent/"LICENSE")+sep+"project-license","--add-data",str(root.parent/"NOTICE")+sep+"project-license",str(root/"packaging/entry.py")],check=True)
