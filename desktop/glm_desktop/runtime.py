import json
import os
import re
import secrets
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

BRIDGE = Path(__file__).with_name("bridge.py").resolve()


def decode_output(raw):
    """WSL launcher diagnostics may be UTF-16LE; Linux workers use UTF-8."""
    if isinstance(raw, str):
        return raw.replace("\x00", "").strip()
    if raw.count(b"\x00") > len(raw) // 5:
        # readline splits at LF, before the UTF-16 newline's final NUL.
        raw = raw.lstrip(b"\x00")
        if len(raw) % 2:
            raw += b"\x00"
        return raw.decode("utf-16-le", errors="replace").strip("\ufeff\r\n ")
    return raw.decode("utf-8", errors="replace").replace("\x00", "").strip()


def wsl_path(path):
    parsed = PureWindowsPath(str(path))
    if not re.fullmatch(r"[A-Za-z]:", parsed.drive):
        raise ValueError("The desktop app must be installed on a local Windows drive for WSL launch.")
    return "/mnt/" + parsed.drive[0].lower() + "/" + "/".join(parsed.parts[1:])


@dataclass(frozen=True)
class Runtime:
    python: str
    wsl: bool = False
    distro: str = "Ubuntu"

    @staticmethod
    def discover(wsl=False, distro="Ubuntu"):
        if wsl and (not distro.strip() or distro.startswith("-")):
            raise ValueError("Enter your WSL distribution name first.")
        code = "from pathlib import Path;import shutil; candidates=[Path.home()/'.local/share/glm/training-venv/bin/python',Path.home()/'gatorllm-venv/bin/python'];print(next((str(p) for p in candidates if p.is_file()),shutil.which('python3') or ''))"
        if wsl:
            command=["wsl.exe","--distribution",distro,"--","python3","-c",code]
            result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=25,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if result.returncode:
                raise ValueError("Could not open that WSL distribution. Install Ubuntu/WSL and confirm its name.")
            found=result.stdout.strip()
        else:
            import shutil
            candidate=Path.home()/".local/share/glm/training-venv/bin/python"
            if not candidate.is_file():
                candidate=Path.home()/"gatorllm-venv"/("Scripts/python.exe" if os.name == "nt" else "bin/python")
            found=str(candidate) if candidate.is_file() else shutil.which("python3") or shutil.which("python")
        if not found:
            raise ValueError("Python was not found. Install the training environment using the setup guide.")
        return found

    def command(self):
        if not self.python.strip() or self.python.startswith("-"):
            raise ValueError("Choose the Python executable from your installed training environment.")
        if self.wsl and (not self.distro.strip() or self.distro.startswith("-")):
            raise ValueError("Enter a valid WSL distribution name.")
        return (["wsl.exe", "--distribution", self.distro, "--", self.python, "-u", wsl_path(BRIDGE)] if self.wsl else [self.python, "-u", str(BRIDGE)])


class Job:
    """Owns exactly one subprocess; never shells out, stops WSL, or kills unrelated jobs."""
    def __init__(self, runtime, payload, emit):
        self.runtime, self.emit = runtime, emit
        self.nonce = secrets.token_hex(16)
        self.secret = payload.get("key", "")
        self.group = None
        self.stopping = False
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(runtime.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=flags, start_new_session=os.name != "nt")
        try:
            self.process.stdin.write((json.dumps({**payload, "nonce": self.nonce}) + "\n").encode("utf-8"))
            self.process.stdin.close()
        except (OSError, ValueError):
            self.process.terminate()
            self.process.wait(timeout=10)
            self.process.stdout.close()
            self.process.stderr.close()
            raise RuntimeError("The training runtime exited before accepting its settings. Check its Python path.") from None
        self.error_thread = threading.Thread(target=self.read_errors, daemon=True)
        self.error_thread.start()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            for raw in self.process.stdout:
                line = decode_output(raw[:32000])
                if self.secret:
                    line = line.replace(self.secret, "[access key hidden]")
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                if isinstance(row, dict) and row.get("desktop_event") == "started" and row.get("nonce") == self.nonce:
                    group = row.get("group")
                    if type(group) is int and group > 1:
                        self.group = group
                    continue
                self.emit("line", (line, row))
            code = self.process.wait()
            self.error_thread.join(timeout=5)
            self.emit("exit", dict(code=code, stopped=self.stopping))
        finally:
            self.secret = ""
            self.process.stdout.close()

    def read_errors(self):
        try:
            for raw in self.process.stderr:
                line = decode_output(raw[:32000])
                if self.secret:
                    line = line.replace(self.secret, "[access key hidden]")
                if line:
                    self.emit("line", (line, None))
        finally:
            self.process.stderr.close()

    @property
    def running(self):
        return self.process.poll() is None

    def stop(self):
        if not self.running:
            return
        self.stopping = True
        if self.runtime.wsl:
            # The bridge's own process group includes all of its Hivemind children.
            # Wait briefly for the authenticated startup record before signalling.
            for _ in range(300):
                if self.group or not self.running:
                    break
                threading.Event().wait(.1)
            if self.group:
                self._wsl_signal("TERM", group=False)
            else:
                self.process.terminate()
        elif os.name != "nt":
            os.kill(self.process.pid, signal.SIGTERM)
        else:
            self.process.terminate()  # Native Windows training is rejected by the bridge.
        try:
            self.process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            if self.runtime.wsl and self.group:
                self._wsl_signal("KILL")
            elif os.name != "nt":
                os.killpg(self.process.pid, signal.SIGKILL)
            else:
                self.process.kill()
            self.process.wait(timeout=10)

    def _wsl_signal(self, name, *, group=True):
        subprocess.run(["wsl.exe", "--distribution", self.runtime.distro, "--", "kill", "-" + name, "--", ("-" if group else "") + str(self.group)],
            capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
