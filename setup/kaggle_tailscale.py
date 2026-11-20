"""Paste into a private, Internet-enabled Kaggle notebook."""

import atexit
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
from uuid import uuid4


VERSION = "1.104.7"
ARCHIVE_SHA256 = "e56693ff74a64f2591b7e819c280635a100aae1ddfb0ef058fcd052b7cd8d16c"
PEER_PORT = 40190
SOCKS_PORT = 40191
TAILNET_RANGE = IPv4Network("100.64.0.0/10")
# Optional. Leave blank if you are still waiting for your teammate's IP.
PEER_IP = ""


def status(step, message):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] [{step}] {message}", flush=True)


class ProbeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not re.fullmatch(r"/probe/[0-9a-f]{32}", self.path):
            self.send_error(404)
            return
        body = self.path.rsplit("/", 1)[-1].encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class KaggleTailnet:
    def __init__(self):
        self.directory = TemporaryDirectory(prefix="glm-tailnet-")
        self.root = Path(self.directory.name)
        self.cli = [str(self.root / "tailscale"), f"--socket={self.root / 'tailscale.sock'}"]
        self.process = None
        self.server = None
        self.server_thread = None
        atexit.register(self.close)

    def start(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Use a Linux x86-64 Kaggle notebook.")

        name = f"tailscale_{VERSION}_amd64"
        archive = self.root / f"{name}.tgz"
        status("01/download", "Fetching Tailscale from its official package server.")
        subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--max-time", "60",
            "--proto", "=https", "--output", str(archive),
            f"https://pkgs.tailscale.com/stable/{name}.tgz",
        ], check=True, timeout=65)
        with archive.open("rb") as payload:
            digest = sha256()
            for block in iter(lambda: payload.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError("Downloaded Tailscale archive failed its checksum.")
        status("02/verify", "Archive checksum matched. Extracting the two binaries.")
        with tarfile.open(archive) as bundle:
            for binary in ("tailscale", "tailscaled"):
                with bundle.extractfile(f"{name}/{binary}") as source, (self.root / binary).open("wb") as target:
                    shutil.copyfileobj(source, target)
                (self.root / binary).chmod(0o700)

        status("03/daemon", "Starting tailscaled in userspace.")
        with (self.root / "daemon.log").open("w") as log:
            self.process = subprocess.Popen([
                str(self.root / "tailscaled"), "--tun=userspace-networking", "--state=mem:",
                f"--socket={self.root / 'tailscale.sock'}",
                f"--socks5-server=127.0.0.1:{SOCKS_PORT}",
            ], stdout=log, stderr=subprocess.STDOUT)
        deadline = monotonic() + 10
        while not (self.root / "tailscale.sock").exists():
            if self.process.poll() is not None or monotonic() >= deadline:
                raise RuntimeError("Tailscale daemon did not start. Restart the Kaggle kernel and retry.")
            sleep(0.2)
        status("04/daemon", "Local control socket is available.")

        hostname = f"glm-kaggle-{uuid4().hex[:8]}"
        status("05/login", "Open the sign-in link in a separate tab and choose your team's tailnet.")
        # Relay CLI output to the notebook as it arrives so login can finish interactively.
        with subprocess.Popen([
            *self.cli, "up", "--timeout=3m", f"--hostname={hostname}",
            "--accept-dns=false", "--accept-routes=false",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as login:
            try:
                for line in login.stdout:
                    print(line, end="", flush=True)
                if login.wait() != 0:
                    raise RuntimeError("Tailscale sign-in failed or expired.")
            except BaseException:
                if login.poll() is None:
                    login.terminate()
                raise
        status("06/login", "tailscale up returned successfully.")

        status("07/listener", "Starting the local probe server.")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
        self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        status("08/forward", f"Configuring the tailnet TCP forwarder on port {PEER_PORT}.")
        forward = subprocess.run([
            *self.cli, "serve", "--bg", f"--tcp={PEER_PORT}",
            f"tcp://127.0.0.1:{self.server.server_port}",
        ], capture_output=True, text=True, timeout=15)
        if forward.returncode:
            raise RuntimeError(f"Tailscale Serve failed: {forward.stderr.strip() or forward.stdout.strip()}")
        status("09/address", "TCP forwarder configured. Reading this node's tailnet IP.")
        address = subprocess.check_output([*self.cli, "ip", "-4"], text=True, timeout=10).strip()
        self.address = IPv4Address(address)
        print(f"Ready: {self.address} on TCP {PEER_PORT}. Keep this notebook running.", flush=True)
        print('To check your teammate, run test_kaggle_peer("THEIR_READY_IP") in a new cell.', flush=True)

    def probe(self, address):
        peer = IPv4Address(address.strip())
        if peer not in TAILNET_RANGE or peer == self.address:
            raise ValueError("Enter the other notebook's Tailscale IPv4 address.")
        challenge = uuid4().hex
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--max-time", "15",
            "--noproxy", "", "--socks5", f"127.0.0.1:{SOCKS_PORT}",
            f"http://{peer}:{PEER_PORT}/probe/{challenge}",
        ], capture_output=True, text=True, timeout=20)
        if result.returncode or result.stdout != challenge:
            raise RuntimeError(f"Peer did not answer. Check both notebooks are ready and the tailnet allows TCP {PEER_PORT}.")
        print(f"TCP PASS: {self.address} → {peer}:{PEER_PORT}", flush=True)

    def close(self):
        atexit.unregister(self.close)
        if self.server is not None:
            if self.server_thread is not None and self.server_thread.is_alive():
                self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.directory.cleanup()


def test_kaggle_peer(address):
    kaggle_tailnet.probe(address)


def stop_kaggle_tailnet():
    kaggle_tailnet.close()


if __name__ == "__main__":
    status("00/start", "Kaggle Tailscale diagnostics v1. This cell never waits for IP input.")
    if "kaggle_tailnet" in globals():
        stop_kaggle_tailnet()
    kaggle_tailnet = KaggleTailnet()
    try:
        kaggle_tailnet.start()
    except BaseException as error:
        status("error", f"{type(error).__name__}: {error}")
        kaggle_tailnet.close()
        raise
    if PEER_IP.strip():
        status("10/probe", "Checking the configured peer.")
        test_kaggle_peer(PEER_IP)
    status("done", "Setup cell finished. Tailscale stays active while this kernel is alive.")
