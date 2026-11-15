"""Real CPU worker stops below target, resumes, and produces another checkpoint."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import torch
from glmtrain.data import prepare


def main():
    processes, handles = [], []
    with tempfile.TemporaryDirectory(prefix="glm-checkpoint-") as temp:
        root = Path(temp)
        (root / "train.txt").write_text("A gator swims through the river.\n" * 200)
        (root / "val.txt").write_text("A bird rests beside the lake.\n" * 100)
        config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
        config["training"].update(target_batch_tokens=16, target_tokens=48)
        (root / "config.json").write_text(json.dumps(config))
        prepare(root / "train.txt", root / "val.txt", root / "config.json", root / "bundle")

        def launch(name, args):
            handle = open(root / f"{name}.log", "w")
            handles.append(handle)
            process = subprocess.Popen([sys.executable, "-m", "glmtrain", *args], stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            processes.append(process)
            return process

        try:
            bootstrap = launch("bootstrap", ["bootstrap", "--bundle", str(root / "bundle"),
                "--port", "0", "--identity", str(root / "bootstrap.key"),
                "--peers-file", str(root / "peers.json")])
            deadline = time.monotonic() + 40
            while not (root / "peers.json").exists():
                if bootstrap.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Bootstrap did not start")
                time.sleep(0.2)
            peer = json.loads((root / "peers.json").read_text())[0]
            common = ["train", "--bundle", str(root / "bundle"), "--peer", peer,
                "--device", "cpu", "--microbatch", "1", "--threads", "1", "--name", "alice",
                "--shard", "0", "--output", str(root / "alice"), "--checkpoint-every", "1",
                "--matchmaking-time", "1", "--averaging-timeout", "30", "--max-microsteps", "1"]
            first = launch("first", common + ["--start-new"])
            assert first.wait(timeout=90) == 0
            checkpoint = root / "alice/latest.pt"
            before = torch.load(checkpoint, map_location="cpu", weights_only=True)
            epoch = before["optimizer"]["state"]["local_epoch"]
            assert 0 < epoch < 3 and before["local_tokens"] > 0
            resumed = launch("resume", common + ["--resume"])
            assert resumed.wait(timeout=90) == 0
            ready = [json.loads(line) for line in (root / "resume.log").read_text().splitlines()
                     if line.startswith("{") and '"event": "ready"' in line]
            assert ready and ready[0]["epoch"] == epoch
            after = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert after["optimizer"]["state"]["local_epoch"] > epoch
            assert after["local_tokens"] > before["local_tokens"]
            before_state, after_state = before["optimizer"]["state"], after["optimizer"]["state"]
            parameter_ids = [key for key in before_state if isinstance(key, int)]
            assert parameter_ids, "Checkpoint has no Adam state"
            for key in parameter_ids:
                assert after_state[key]["step"] > before_state[key]["step"]
            assert any(not torch.equal(before["model"][key], after["model"][key]) for key in before["model"])
            rows = [json.loads(line) for line in (root / "alice/metrics.jsonl").read_text().splitlines()]
            assert rows[-1]["epoch"] == after["optimizer"]["state"]["local_epoch"]
            assert rows[-1]["local_tokens"] == after["local_tokens"]
            print(f"PASS: resumed epoch {epoch} -> {rows[-1]['epoch']}, tokens {before['local_tokens']} -> {after['local_tokens']}, model updated")
        except BaseException:
            for path in root.glob("*.log"):
                print(f"--- {path.name} ---\n{path.read_text()[-14000:]}")
            raise
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
