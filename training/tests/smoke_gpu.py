"""Opt-in local GPU test of the real baseline model; never falls back to CPU.

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
