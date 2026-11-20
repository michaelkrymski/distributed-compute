"""Paste this entire file into one Kaggle Python cell, then run it."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from urllib.error import URLError
from urllib.request import urlopen


def kaggle_smoke():
    revision = "11898d27ec28bb4dc834675921b19f29aa305f27"
    source = f"https://raw.githubusercontent.com/michaelkrymski/distributed-compute/{revision}/training"
    output = Path("/kaggle/working")
    if not output.is_dir():
        output = Path.cwd()
    report = output / "glm-kaggle-smoke.json"
    result = {"status": "incomplete", "revision": revision, "events": []}
    started = monotonic()
    # Reserve a margin under five minutes after the cell gets a running kernel.
    deadline = started + 240

    def remaining():
        seconds = deadline - monotonic()
        if seconds <= 0:
            raise TimeoutError("The four-minute test budget expired.")
        return seconds

    # Replace any prior PASS before starting so a failed rerun cannot look green.
    report.write_text(json.dumps(result), encoding="utf-8")
    try:
        print("Loading the pinned training smoke test from GitHub…", flush=True)
        with TemporaryDirectory(prefix="glm-kaggle-") as directory:
            root = Path(directory)
            for filename in ("glmtrain/model.py", "configs/baseline.json", "tests/smoke_gpu.py"):
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(f"{source}/{filename}", timeout=min(15, remaining())) as response:
                    path.write_bytes(response.read())

            print("Testing one CUDA GPU with Kaggle's installed PyTorch…", flush=True)
            completed = subprocess.run(
                [sys.executable, "-u", "tests/smoke_gpu.py", "--backend", "cuda"],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=remaining(),
            )
            for line in completed.stdout.splitlines():
                print(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    result["events"].append(event)
            if completed.stderr:
                print(completed.stderr, end="")
            if completed.returncode or not any(e.get("event") == "PASS" for e in result["events"]):
                raise RuntimeError("GPU smoke test failed. Check Accelerator and the output above.")
        result["status"] = "pass"
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        result.update(status="fail", error="The four-minute test budget expired.")
        raise RuntimeError(result["error"]) from error
    except URLError as error:
        result.update(status="fail", error="Could not fetch GitHub source. Turn Internet on in Kaggle.")
        raise RuntimeError(result["error"]) from error
    except Exception as error:
        result.update(status="fail", error=str(error))
        raise
    finally:
        result["elapsedSeconds"] = round(monotonic() - started, 2)
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Result: {result['status'].upper()} in {result['elapsedSeconds']}s. Report: {report}")


if __name__ == "__main__":
    kaggle_smoke()
