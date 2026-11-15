"""Executed by the user's existing training Python, not by the GUI interpreter.

Credentials arrive only on stdin. No changes to the training libraries or API.
"""
import importlib
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
from types import SimpleNamespace


def send(**values):
    print(json.dumps(values), flush=True)


def request_stop(signum, frame):
    # Some Hivemind versions ignore SIGINT while forking. SIGTERM to the
    # bridge parent still unwinds the engine's existing finally/cleanup blocks.
    raise KeyboardInterrupt


def training_source(project, bridge_path=None):
    from pathlib import Path
    bridge = Path(bridge_path or __file__).resolve()
    # Source checkout: repo/desktop/glm_desktop; packaged: app/_internal/glm_desktop.
    for base in (bridge.parents[1], bridge.parents[2]):
        candidate = base / project
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Training source is missing from this installation. Reinstall the complete app bundle or use the automated setup guide.")


def main():
    payload = json.loads(sys.stdin.readline(65536))
    if os.name != "nt":
        signal.signal(signal.SIGTERM, request_stop)
        try:
            os.setsid()
        except PermissionError:
            if os.getpgrp() != os.getpid():
                raise RuntimeError("Could not create an isolated training process group")
    send(desktop_event="started", nonce=payload.pop("nonce"), group=os.getpgrp() if os.name != "nt" else None)
    # Older worker installations remain usable during the rollout.
    for current, legacy in (("glmbench", "gatorbench"), ("glmtrain", "gatortrain")):
        if current not in sys.modules and importlib.util.find_spec(current) is None and importlib.util.find_spec(legacy) is not None:
            sys.modules[current] = importlib.import_module(legacy)
    operation = payload.pop("operation")
    if operation == "install":
        for index, project in enumerate(("training", "hivemind-tools")):
            source = training_source(project)
            send(event="install_stage", progress=index/2, label="Installing " + project + ". Live download output is in Debug.")
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-input", "-e", str(source)], check=True)
        send(event="install_complete")
        return
    if operation == "check_network":
        ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=15)
        if ip.returncode:
            raise RuntimeError("Tailscale needs to be installed, running and signed into the team tailnet. Use the Tailscale setup bar.")
        send(event="desktop_network", host=ip.stdout.strip().splitlines()[0])
        return
    if operation == "check_gpu":
        from glmbench.device import select_device, describe
        selected, _ = select_device(payload.get("device", "auto"), 0)
        send(event="desktop_gpu", device=describe(selected)["name"])
        return
    if operation == "check":
        if not (3, 11) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError("The training environment needs Python 3.11 or 3.12. The GUI may use a newer Python.")
        from importlib import metadata
        packages = {}
        for name in ("torch", "hivemind", "glm-hivemind-tools", "glm-training"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                legacy = {"glm-hivemind-tools": "gatorllm-hivemind-tools", "glm-training": "gatorllm-training"}.get(name, name)
                packages[name] = metadata.version(legacy)
        import torch
        from glmbench.device import select_device, describe
        selected, _ = select_device(payload.get("device", "auto"), 0)
        address = ""
        try:
            ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10)
            if ip.returncode == 0:
                address = ip.stdout.strip().splitlines()[0]
        except (OSError, subprocess.TimeoutExpired):
            pass
        send(event="desktop_check", packages=packages, python=sys.executable, platform=platform.system(), host=address,
            device=describe(selected)["name"], hip=bool(torch.version.hip))
        return
    if platform.system() == "Windows":
        raise RuntimeError("Run the training tools in Linux/WSL. The desktop app itself can run on Windows.")
    if operation in ("agent", "coordinator"):
        from glmbench.client import Client
        from glmbench.benchmark import agent, coordinator
        client = Client(payload.pop("api"), payload.pop("key"))
        if operation == "agent" and (client.get().get("session") or {}).get("id") != payload.get("session_id"):
            raise RuntimeError("The lobby changed. Refresh the desktop app before joining.")
        args = SimpleNamespace(host=payload["host"], port=0 if operation == "agent" else 40100,
            identity=None, device=payload.get("device", "auto"), device_index=0,
            threads=2, ready=False, output=payload["output"])
        (agent if operation == "agent" else coordinator)(args, client)
        return
    if operation in ("inspect", "train", "room", "bootstrap"):
        from glmtrain.data import load_bundle
        manifest = load_bundle(payload["bundle"])
        settings = manifest["training"]
        send(event="desktop_bundle", run=manifest["run_id"], target_tokens=settings["target_tokens"],
            fingerprint=manifest["fingerprint"], shards=settings["data_shards"], batch_tokens=settings["target_batch_tokens"])
        if operation == "bootstrap":
            from glmtrain.network import bootstrap
            return bootstrap(SimpleNamespace(host=payload["host"], port=40101, identity=None, peers_file=None), manifest)
        if operation == "room":
            from glmtrain import __main__ as cli
            from room_worker import run
            return run(payload, manifest, cli)
        if operation == "inspect":
            return
        selected_device = payload.get("device", "auto")
        if selected_device not in ("auto", "cpu", "cuda", "rocm"):
            raise ValueError("Choose Automatic, NVIDIA, AMD or CPU in Get started")
        if selected_device in ("cuda", "rocm"):
            from glmbench.device import select_device
            select_device(selected_device, 0)  # Enforce the requested vendor / no CPU fallback.
        from glmtrain import __main__ as cli
        shard = int(payload["shard"])
        if not 0 <= shard < settings["data_shards"]:
            raise ValueError(f"Your organizer must assign a shard from 0 to {settings['data_shards']-1}")
        argv = ["glmtrain", "train", "--bundle", payload["bundle"], "--host", payload["host"],
            "--peer", payload["peer"], "--name", payload["name"], "--shard", str(shard),
            "--output", payload["output"], "--microbatch", "1", "--device", "cpu" if selected_device == "cpu" else "auto", "--precision", "fp32"]
        if payload.get("resume"):
            argv.append("--resume")
        # Join an existing run; creating a new model is deliberately organizer-only CLI work.
        sys.argv = argv
        cli.main()
        return
    raise ValueError("Unsupported desktop operation")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Corpus training resumes from its last completed checkpoint.", flush=True)
    except Exception as error:
        send(event="desktop_error", message=str(error))
        sys.exit(1)
