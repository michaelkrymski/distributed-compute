"""Paste this entire file into one Kaggle Python cell, then run it."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic


# Set to "benchmark" on both notebooks to compare their independent GPU throughput.
MODE = "smoke"
BACKEND = "compute"  # Set to "team2" for the original GatorLLM dashboard.
MEASURE_SECONDS = 60
MODEL_REVISION = "11898d27ec28bb4dc834675921b19f29aa305f27"
BACKEND_HOSTS = {
    "compute": "distributed-compute-platform.mkrymski.chatgpt.site",
    "team2": "gatorllm-team-2-lab.netlify.app",
}


def fetch_project(key, backend):
    from http.client import HTTPSConnection

    # Use the exact first-party host and do not follow redirects with a key.
    connection = HTTPSConnection(BACKEND_HOSTS[backend], timeout=15)
    try:
        connection.request("GET", "/api/benchmark", headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "glmbench/0.4.0",
        })
        response = connection.getresponse()
        if response.status in (401, 403):
            raise RuntimeError(f"Backend returned HTTP {response.status}. Check your access key for {BACKEND_HOSTS[backend]}.")
        if response.status != 200:
            raise RuntimeError(f"Backend returned HTTP {response.status}.")
        state = json.loads(response.read())
        if not isinstance(state.get("user"), dict) or not state["user"].get("id"):
            raise RuntimeError("The backend did not return an authenticated user.")
        if backend == "team2":
            # The legacy API authenticates team members but has no projects.
            return {
                "id": "gatorllm-team-2-baseline",
                "name": "GatorLLM Team 2 baseline",
                "runtime": "glm-decoder-v1",
                "model": json.loads(BUNDLED_FILES["configs/baseline.json"])["model"],
                "parameters": 24_388_992,
            }
        return state.get("project")
    except OSError:
        raise RuntimeError("Could not reach the selected backend. Enable Internet in this Kaggle notebook and retry.") from None
    finally:
        connection.close()


def kaggle_benchmark():
    from getpass import getpass
    from hashlib import sha256
    from types import ModuleType

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select GPU T4 x2 in Kaggle's Accelerator settings.")
    if type(MEASURE_SECONDS) is not int or not 10 <= MEASURE_SECONDS <= 180:
        raise ValueError("Use a measurement duration of 10–180 seconds.")
    if BACKEND not in BACKEND_HOSTS:
        raise ValueError('Set BACKEND to "compute" or "team2".')
    key_kind = "Team 2 access key" if BACKEND == "team2" else "Compute project worker key"
    key = getpass(f"{key_kind}: ").strip()
    if not key or len(key) > 249 or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError(f"Enter your own {key_kind}.")
    if BACKEND == "compute" and not key.startswith("dc_worker_"):
        raise ValueError("Use your own Compute project worker key starting with dc_worker_.")
    project = fetch_project(key, BACKEND)
    del key
    if not project or project.get("runtime") != "glm-decoder-v1":
        raise RuntimeError("Select an included decoder project in Compute.")

    module = ModuleType("_glm_kaggle_model")
    sys.modules[module.__name__] = module
    exec(compile(BUNDLED_FILES["glmtrain/model.py"], "glmtrain/model.py", "exec"), module.__dict__)
    config = module.ModelConfig(**project["model"])
    count = config.parameter_count()
    if count != project["parameters"] or count > 25_000_000 or config.context > 512:
        raise RuntimeError("This notebook supports matching decoder projects up to 25M parameters and 512 context.")
    fingerprint = sha256(json.dumps(project["model"], sort_keys=True).encode()).hexdigest()
    output = Path("/kaggle/working/glm-kaggle-benchmark.json")
    if not output.parent.is_dir():
        output = Path.cwd() / output.name
    report = {
        "status": "incomplete", "mode": "independent-single-gpu",
        "backend": BACKEND, "projectId": project["id"],
    }
    output.write_text(json.dumps(report), encoding="utf-8")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    model = module.LanguageModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.1)

    def step():
        tokens = torch.randint(config.vocab_size, (1, config.context + 1), device=device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        return loss

    print(f"{project['name']} | {torch.cuda.get_device_name(0)} | {count:,} parameters", flush=True)
    print(f"Warming up for 5 seconds, then measuring for {MEASURE_SECONDS} seconds…", flush=True)
    warmup_end = monotonic() + 5
    while monotonic() < warmup_end:
        step()
    started = monotonic()
    steps = 0
    while monotonic() - started < MEASURE_SECONDS:
        loss = step()
        steps += 1
    elapsed = monotonic() - started
    if not torch.isfinite(loss).item():
        raise RuntimeError("The training loss is nonfinite.")
    processed = steps * config.context
    report.update(
        status="pass", modelRevision=MODEL_REVISION, modelConfigHash=fingerprint,
        device=torch.cuda.get_device_name(0), torchVersion=torch.__version__,
        parameters=count, context=config.context, microbatch=1, precision="fp32",
        measuredSeconds=elapsed, steps=steps, processedTokens=processed,
        tokensPerSecond=processed / elapsed, finalLoss=loss.item(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Independent GPU throughput: {report['tokensPerSecond']:,.1f} tokens/s | report: {output}")


def kaggle_smoke():
    output = Path("/kaggle/working")
    if not output.is_dir():
        output = Path.cwd()
    report = output / "glm-kaggle-smoke.json"
    result = {"status": "incomplete", "revision": MODEL_REVISION, "events": []}
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
        print("Loading the bundled training smoke test…", flush=True)
        with TemporaryDirectory(prefix="glm-kaggle-") as directory:
            root = Path(directory)
            for filename, source in BUNDLED_FILES.items():
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")

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
    except Exception as error:
        result.update(status="fail", error=str(error))
        raise
    finally:
        result["elapsedSeconds"] = round(monotonic() - started, 2)
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Result: {result['status'].upper()} in {result['elapsedSeconds']}s. Report: {report}")


# Snapshot of the three training files at the revision recorded in the report.
BUNDLED_FILES = {
    'glmtrain/model.py': r'''"""Bias-free decoder: dense attention, RoPE, RMSNorm, GELU FFN."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    width: int = 384
    layers: int = 12
    heads: int = 6
    expansion: int = 4
    context: int = 512
    tied: bool = True

    def __post_init__(self):
        for key in ("vocab_size", "width", "layers", "heads", "expansion", "context"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 257 <= self.vocab_size <= 65536:
            raise ValueError("Byte BPE requires vocabulary between 257 and 65536")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width / heads must be an even integer for RoPE")
        if type(self.tied) is not bool:
            raise ValueError("tied must be boolean")

    def parameter_count(self):
        d, f = self.width, self.width * self.expansion
        return (
            self.vocab_size * d * (1 if self.tied else 2)
            + self.layers * (4 * d * d + 2 * d * f + 2 * d)
            + d
        )


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        scaled = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        return scaled.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.width // c.heads
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        positions = torch.arange(c.context).float()
        frequency = 10000 ** (
            -torch.arange(0, self.head_dim, 2).float() / self.head_dim
        )
        angles = positions[:, None] * frequency[None, :]
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    def rotate(self, x):
        cos, sin = (
            self.cos[:, :, : x.size(2)].to(x.dtype),
            self.sin[:, :, : x.size(2)].to(x.dtype),
        )
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            a.view(b, t, self.heads, self.head_dim).transpose(1, 2) for a in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            self.rotate(q), self.rotate(k), v, is_causal=True
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.width), RMSNorm(c.width)
        self.attention = Attention(c)
        self.ffn = nn.Sequential(
            nn.Linear(c.width, c.width * c.expansion, bias=False),
            nn.GELU(),
            nn.Linear(c.width * c.expansion, c.width, bias=False),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = RMSNorm(config.width)
        self.output = (
            None
            if config.tied
            else nn.Linear(config.width, config.vocab_size, bias=False)
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, tokens, targets=None):
        if tokens.ndim != 2 or tokens.size(1) > self.config.context:
            raise ValueError("Expected [batch, sequence] within configured context")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = (
            F.linear(x, self.embedding.weight)
            if self.output is None
            else self.output(x)
        )
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        )
        return logits, loss
''',
    'configs/baseline.json': r'''{
  "model": {"vocab_size":8192,"width":384,"layers":12,"heads":6,"expansion":4,"context":512,"tied":true},
  "training": {"seed":42,"learning_rate":0.0003,"weight_decay":0.1,"target_batch_tokens":32768,"target_tokens":500000000,"data_shards":5}
}
''',
    'tests/smoke_gpu.py': r'''"""Opt-in local GPU test of the real baseline model; never falls back to CPU.

Run from training/: python tests/smoke_gpu.py --backend rocm
This tests FP32 model training and AdamW checkpointing, not Hivemind or a corpus.
"""

import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
from glmtrain.model import LanguageModel, ModelConfig


BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"


def select_gpu(backend="auto", index=0):
    if index < 0:
        raise ValueError("GPU index must be nonnegative")
    if backend not in ("auto", "cuda", "rocm"):
        raise ValueError("Backend must be auto, cuda, or rocm")
    detected = "rocm" if torch.version.hip else "cuda" if torch.version.cuda else None
    if detected is None:
        raise RuntimeError("CPU-only PyTorch installed; install a CUDA or ROCm build")
    if backend != "auto" and backend != detected:
        raise RuntimeError(f"Requested {backend}, but installed PyTorch uses {detected}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{detected} GPU unavailable; check drivers and device access")
    if index >= torch.cuda.device_count():
        raise ValueError("GPU index exceeds the number of visible GPUs")
    # ROCm deliberately uses PyTorch's CUDA device namespace too.
    return torch.device("cuda", index), detected


def make_optimizer(model, settings):
    return torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=settings["weight_decay"],
    )


def training_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.device != x.device or parameter.grad is None:
            raise RuntimeError(f"{name}: missing GPU parameter/gradient")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"{name}: nonfinite gradient")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    if not bool(norm > 0):
        raise RuntimeError("All gradients are zero")
    before = model.embedding.weight.detach().clone()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"{name}: nonfinite parameter after update")
    if torch.equal(before, model.embedding.weight):
        raise RuntimeError("Optimizer did not change embedding weights")
    torch.cuda.synchronize(x.device)
    return {"loss": loss.item(), "gradientNorm": norm.item()}


def run(args):
    if args.steps < 2 or args.microbatch < 1:
        raise ValueError("Use at least two steps and a positive microbatch")
    device, backend = select_gpu(args.backend, args.device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    config = json.loads(BASELINE.read_text())
    cfg = ModelConfig(**config["model"])
    print(json.dumps({
        "event": "gpu_selected",
        "backend": backend,
        "device": str(device),
        "name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "precision": "fp32",
        "parameters": cfg.parameter_count(),
        "context": cfg.context,
        "microbatch": args.microbatch,
    }), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = LanguageModel(cfg).to(device).train()
    optimizer = make_optimizer(model, config["training"])
    # Shifted next-token targets, at the full baseline context/vocabulary sizes.
    # Reuse one synthetic batch: this is a functionality test, not convergence.
    tokens = torch.randint(0, cfg.vocab_size, (args.microbatch, cfg.context + 1))
    x = tokens[:, :-1].contiguous().to(device)
    y = tokens[:, 1:].contiguous().to(device)
    for step in range(args.steps):
        metrics = training_step(model, optimizer, x, y)
        print(json.dumps({"event": "step", "step": step + 1, **metrics}), flush=True)

    with tempfile.TemporaryDirectory(prefix="glm-gpu-smoke-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        restored = LanguageModel(cfg).to(device).train()
        restored.load_state_dict(saved["model"])
        resumed_optimizer = make_optimizer(restored, config["training"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        for original, resumed in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(original, resumed, rtol=0, atol=0)
            for key, value in optimizer.state[original].items():
                torch.testing.assert_close(
                    value, resumed_optimizer.state[resumed][key], rtol=0, atol=0
                )
        if optimizer.state_dict()["param_groups"] != resumed_optimizer.state_dict()["param_groups"]:
            raise RuntimeError("Optimizer settings changed during checkpoint reload")
        with torch.no_grad():
            expected, _ = model(x)
            actual, _ = restored(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        metrics = training_step(restored, resumed_optimizer, x, y)
        print(json.dumps({"event": "resumed_step", **metrics}), flush=True)

    print(json.dumps({
        "event": "PASS",
        "backend": backend,
        "trainingSteps": args.steps + 1,
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "peakAllocatedMiB": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "scope": "Single-GPU FP32 baseline model + AdamW + checkpoint resume; not Hivemind, mixed precision, or training quality",
    }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cuda", "rocm"), default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=1)
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, AssertionError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
''',
}


if __name__ == "__main__":
    if MODE == "benchmark":
        kaggle_benchmark()
    elif MODE == "smoke":
        kaggle_smoke()
    else:
        raise ValueError('Set MODE to "smoke" or "benchmark".')
