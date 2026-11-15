"""Actual HTTP API + D1 + five Hivemind processes, full 60s+300s timing.

Run against a freshly migrated local backend only. No fake HTTP server or DHT.
"""

import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
from urllib.parse import urlparse
from glmbench.client import Client


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:5175")
    p.add_argument("--keys-file", required=True)
    p.add_argument("--output", default="results/e2e")
    p.add_argument("--agents", type=int, default=5)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--measure", type=int, default=300)
    p.add_argument(
        "--workload", choices=("synthetic", "baseline-training"), default="synthetic"
    )
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument("--training-profile", choices=("reference-fp32", "optimized-fp32"), default="reference-fp32")
    p.add_argument("--microbatch", type=int, default=1)
    a = p.parse_args()
    if urlparse(a.api).hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("E2E tests must use an isolated local backend")
    if not 1 <= a.agents <= 10:
        raise ValueError("Choose 1–10 agents")
    keys = json.loads(Path(a.keys_file).read_text())
    admin = Client(a.api, keys["michaelkrymski"])
    old = admin.get().get("session")
    if old and old["phase"] not in ("finished", "cancelled"):
        admin.post(action="cancel", sessionId=old["id"])
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []

    def launch(name, args):
        f = open(output / (name + ".log"), "w")
        handles.append(f)
        proc = subprocess.Popen(
            [sys.executable, "-m", "glmbench", *args],
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "GLM_TEAM_KEY": keys[name]},
        )
        processes.append(proc)
        return proc

    try:
        coordinator = launch(
            "michaelkrymski",
            [
                "coordinator",
                "--api",
                a.api,
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--identity",
                str(output / "coordinator.key"),
            ],
        )
        deadline = time.monotonic() + 45
        while True:
            state = admin.get()
            if state["session"] and state["session"]["id"] != (old or {}).get("id"):
                break
            if coordinator.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Coordinator failed to create a real lobby")
            time.sleep(0.5)
        session = state["session"]
        admin.post(
            action="configure",
            sessionId=session["id"],
            warmupSeconds=a.warmup,
            measureSeconds=a.measure,
            workload=a.workload,
            batchTokens=a.batch_tokens,
            trainingProfile=a.training_profile,
            microbatch=a.microbatch,
        )
        workers = []
        for i in range(a.agents):
            name = list(keys)[i % len(keys)]
            # Separate log for the coordinator's own benchmark agent.
            f = open(output / (name + f"-agent-{i}.log"), "w")
            handles.append(f)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "glmbench",
                    "agent",
                    "--api",
                    a.api,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--device",
                    "cpu",
                    "--threads",
                    "1",
                    "--ready",
                    "--output",
                    str(output),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "GLM_TEAM_KEY": keys[name]},
            )
            processes.append(proc)
            workers.append(proc)
        deadline = time.monotonic() + 100
        while True:
            lobby = admin.get()
            if len(lobby["agents"]) == a.agents and lobby["session"]["canStart"]:
                break
            if (
                any(w.poll() is not None for w in workers)
                or time.monotonic() > deadline
            ):
                raise RuntimeError("The real agents did not become ready")
            time.sleep(0.5)
        wall_start = time.monotonic()
        admin.post(action="start", sessionId=session["id"])
        print(
            f"Started real 15s countdown + {a.warmup}s warmup + {a.measure}s measurement across {a.agents} local processes",
            flush=True,
        )
        deadline = time.monotonic() + 15 + a.warmup + a.measure + 90
        while any(w.poll() is None for w in workers):
            if time.monotonic() > deadline:
                raise RuntimeError("Full benchmark timed out")
            time.sleep(2)
        assert all(w.returncode == 0 for w in workers), "An agent failed"
        wall_seconds = time.monotonic() - wall_start
        assert wall_seconds >= 15 + a.warmup + a.measure - 1, (
            f"Only {wall_seconds:.3f}s elapsed on the independent monotonic clock; expected at least {15 + a.warmup + a.measure - 1}s. Check host clock adjustments."
        )
        state = admin.get()
        reports = state["agents"]
        assert len(reports) == a.agents
        for report in reports:
            r = report["result"]
            assert r and a.measure - 1 <= r["measuredSeconds"] <= a.measure
            assert (a.warmup == 0 or r["warmupRounds"] > 0) and r[
                "measuredRounds"
            ] > 0, (
                report["userId"],
                r,
            )
            assert r["complete"], (report["userId"], r)
            if a.workload == "baseline-training":
                assert r["parameterCount"] == state["session"]["config"]["payloadElements"]
                assert r["processedTokens"] > 0 and r["tokensPerSecond"] > 0
                assert r["optimizerSteps"] >= r["measuredRounds"]
                assert (
                    r["processedTokens"] == r["measuredRounds"] * r["localBatchTokens"]
                )
                assert r["meanLoss"] > 0
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "test": f"{a.agents} local CPU processes against real API and D1, not separate human/GPU participants",
                    "observedWallSeconds": wall_seconds,
                    "session": state["session"],
                    "agents": reports,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"PASS: real API/D1, {a.agents} actual agents, full warmup and measurement windows, persistent result uploads",
            flush=True,
        )
    except BaseException:
        for path in output.glob("*.log"):
            print(f"--- {path.name} ---\n" + path.read_text()[-2500:])
        raise
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for f in handles:
            f.close()


if __name__ == "__main__":
    main()
