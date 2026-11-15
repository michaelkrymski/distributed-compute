"""Real two-worker corpus training through a local HTTP/D1 room. Linux only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

from glmbench.client import Client
from glmtrain.data import prepare


def main():
    base = os.environ.get("GLM_API_BASE", "http://127.0.0.1:5181")
    if urlsplit(base).hostname not in ("127.0.0.1", "localhost"):
        raise RuntimeError("Local test backend only")
    repo = Path(__file__).resolve().parents[2]
    keys = json.loads(Path(os.environ.get("GLM_KEYS_FILE", str(repo / "team-dashboard/work/team-access-keys.json"))).read_text())
    client = Client(base, keys["michaelkrymski"])
    client.url = base + "/api/rooms"
    project = client.get().get("project")
    processes, handles = [], []
    with tempfile.TemporaryDirectory(prefix="glm-room-") as temp:
        root = Path(temp)
        (root / "train.txt").write_text("A small model learns words by the river.\n" * 400)
        (root / "val.txt").write_text("Birds rest in the trees.\n" * 100)
        config = json.loads((repo / "training/configs/smoke.json").read_text())
        config["training"].update(target_batch_tokens=4096, target_tokens=8192, data_shards=2)
        if project:
            config["model"] = project["model"]
            config["training"]["target_tokens"] = project["targetTokens"]
        (root / "config.json").write_text(json.dumps(config))
        manifest = prepare(root / "train.txt", root / "val.txt", root / "config.json", root / "bundle")

        def launch(name, payload):
            log = open(root / (name + ".log"), "w")
            handles.append(log)
            p = subprocess.Popen([sys.executable, str(repo / "desktop/glm_desktop/bridge.py")],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            p.stdin.write((json.dumps(dict(nonce="acceptance", **payload)) + "\n").encode())
            p.stdin.close()
            processes.append(p)
            return p

        try:
            common = dict(bundle=str(root / "bundle"), host="127.0.0.1", device="cpu")
            launch("bootstrap", dict(common, operation="bootstrap", output=str(root / "bootstrap")))
            deadline = time.monotonic() + 60
            peer = None
            while time.monotonic() < deadline and not peer:
                for line in (root / "bootstrap.log").read_text().splitlines():
                    try:
                        peer = json.loads(line).get("initial_peers", [None])[0]
                    except (ValueError, AttributeError):
                        pass
                time.sleep(.2)
            assert peer, "Bootstrap did not publish a peer"
            old = client.get().get("room")
            if old and old["phase"] not in ("cancelled", "finished"):
                client.post(action="cancel", roomId=old["id"])
            client.post(action="create", name="Real CPU room", fingerprint=manifest["fingerprint"],
                peer=peer, targetTokens=8192, shards=2)
            room_id = client.get()["room"]["id"]
            workers = [launch(str(i), dict(common, operation="room", api=base,
                key=keys["michaelkrymski"], name=f"cpu-{i}", output=str(root / str(i)), resume=False)) for i in range(2)]
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                members = client.get()["members"]
                if len(members) == 2:
                    break
                assert all(p.poll() is None for p in workers), "Worker exited before readiness"
                time.sleep(.5)
            assert len(members) == 2
            for m in members:
                client.post(action="ready", roomId=room_id, instanceId=m["instance_id"], ready=True)
            client.post(action="start", roomId=room_id)
            for p in workers:
                assert p.wait(timeout=180) == 0, "Training worker failed"
            for i in range(2):
                log = (root / (str(i) + ".log")).read_text()
                assert '"event": "epoch"' in log, log
            print("Real two-worker room training passed")
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
