"""A real CPU rendezvous + two-worker averaging test, isolated to loopback."""

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
    processes = []
    with tempfile.TemporaryDirectory(prefix="glm-swarm-") as temp:
        root = Path(temp)
        (root / "train.txt").write_text(
            (
                "A gator studies language by the lake.\nAnother gator swims through the river.\n"
                * 200
            )
        )
        (root / "val.txt").write_text(
            ("The bird rests on a branch far from the river.\n" * 100)
        )
        config = json.loads(
            (Path(__file__).parents[1] / "configs" / "smoke.json").read_text()
        )
        config["training"].update(target_batch_tokens=4096, target_tokens=8192)
        (root / "config.json").write_text(json.dumps(config))
        prepare(
            root / "train.txt", root / "val.txt", root / "config.json", root / "bundle"
        )
        handles = []

        def launch(name, arguments):
            f = open(root / f"{name}.log", "w")
            handles.append(f)
            p = subprocess.Popen(
                [sys.executable, "-m", "glmtrain", *arguments],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            )
            processes.append(p)
            return p

        try:
            bootstrap = launch(
                "bootstrap",
                [
                    "bootstrap",
                    "--bundle",
                    str(root / "bundle"),
                    "--port",
                    "0",
                    "--identity",
                    str(root / "bootstrap.key"),
                    "--peers-file",
                    str(root / "peers.json"),
                ],
            )
            until = time.monotonic() + 40
            while not (root / "peers.json").exists():
                if bootstrap.poll() is not None or time.monotonic() > until:
                    raise RuntimeError("Bootstrap did not start")
                time.sleep(0.2)
            peer = json.loads((root / "peers.json").read_text())[0]
            common = [
                "train",
                "--bundle",
                str(root / "bundle"),
                "--peer",
                peer,
                "--device",
                "cpu",
                "--microbatch",
                "1",
                "--threads",
                "1",
                "--min-peers",
                "2",
                "--matchmaking-time",
                "1",
                "--averaging-timeout",
                "30",
                "--join-timeout",
                "60",
                "--checkpoint-every",
                "1",
            ]
            first = launch(
                "alice",
                common
                + [
                    "--name",
                    "alice",
                    "--shard",
                    "0",
                    "--output",
                    str(root / "alice"),
                    "--start-new",
                ],
            )
            second = launch(
                "bob",
                common
                + ["--name", "bob", "--shard", "1", "--output", str(root / "bob")],
            )
            deadline = time.monotonic() + 150
            while first.poll() is None or second.poll() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("Two-worker test timed out")
                if any(p.poll() not in (None, 0) for p in (first, second)):
                    raise RuntimeError("Worker failed")
                time.sleep(0.5)
            assert first.returncode == second.returncode == 0
            logs = (root / "alice.log").read_text() + (root / "bob.log").read_text()
            assert "Averaged gradients with 2 peers" in logs, (
                "No real two-peer gradient averaging was observed"
            )
            a = torch.load(
                root / "alice" / "latest.pt", map_location="cpu", weights_only=True
            )
            b = torch.load(
                root / "bob" / "latest.pt", map_location="cpu", weights_only=True
            )
            assert a["optimizer"]["state"]["local_epoch"] >= 2
            assert b["optimizer"]["state"]["local_epoch"] >= 2
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=0.01, atol=0.002
                )
            assert a["local_tokens"] > 0 and b["local_tokens"] > 0
            resumed = launch(
                "resume",
                [
                    "train",
                    "--bundle",
                    str(root / "bundle"),
                    "--peer",
                    peer,
                    "--device",
                    "cpu",
                    "--microbatch",
                    "1",
                    "--threads",
                    "1",
                    "--name",
                    "alice",
                    "--shard",
                    "0",
                    "--output",
                    str(root / "alice"),
                    "--resume",
                    "--matchmaking-time",
                    "1",
                    "--averaging-timeout",
                    "30",
                ],
            )
            assert resumed.wait(timeout=40) == 0, (root / "resume.log").read_text()
            ready = [
                json.loads(line)
                for line in (root / "resume.log").read_text().splitlines()
                if line.startswith("{") and '"event": "ready"' in line
            ]
            assert ready and ready[0]["epoch"] >= 2, "Resume reset the global epoch"
            print(
                "PASS: real loopback bootstrap, two distinct data shards, state download, two-peer gradient averaging, completed epochs, compatible checkpoints, and checkpoint resume"
            )
        except BaseException:
            for name in ("bootstrap", "alice", "bob", "resume"):
                path = root / f"{name}.log"
                if path.exists():
                    print(f"--- {name} ---\n" + path.read_text()[-14000:])
            raise
        finally:
            for p in reversed(processes):
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGINT)
            for p in processes:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
