"""Reference and optimized model/gradient/AdamW benchmark workloads.

Generated token batches isolate training throughput from corpus/tokenizer I/O.
Weights are disposable; this is not a resumable corpus-training run.
"""

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from .model import LanguageModel, ModelConfig


def allocate_proportional_tokens(batch_tokens, context, members, speeds=None, microbatch=1):
    """Split batch_tokens across active members proportionally to observed compute speeds."""
    ordered = list(members)
    if not ordered:
        return {}
    if context <= 0 or batch_tokens <= 0:
        raise ValueError("batch_tokens and context must be positive")
    if len(ordered) == 1:
        return {ordered[0]: math.ceil(batch_tokens / context) * context}
    total_sequences = max(len(ordered), math.ceil(batch_tokens / context))
    if not speeds or any(
        m not in speeds
        or isinstance(speeds[m], bool)
        or not isinstance(speeds[m], (int, float))
        or not math.isfinite(speeds[m])
        or speeds[m] <= 0
        for m in ordered
    ):
        equal_tokens = math.ceil(batch_tokens / (len(ordered) * context)) * context
        return {m: equal_tokens for m in ordered}
    unique = sorted(ordered)
    weights = {m: max(float(speeds[m]), 1.0) for m in unique}
    total_weight = sum(weights.values())
    seqs = {m: 1 for m in unique}
    remaining = total_sequences - len(unique)
    raw = {m: remaining * (weights[m] / total_weight) for m in unique}
    floors = {m: math.floor(raw[m]) for m in unique}
    for m in unique:
        seqs[m] += floors[m]
    leftover = remaining - sum(floors.values())
    if leftover > 0:
        ranked = sorted(unique, key=lambda m: (-(raw[m] - floors[m]), -weights[m], m))
        for m in ranked[:leftover]:
            seqs[m] += 1
    return {m: seqs[m] * context for m in ordered}


class TrainingWorkload:
    def __init__(self, config, device, seed):
        self.cfg = ModelConfig(**config["model"])
        self.device = device
        self.target_batch = config["batchTokens"]
        self.profile = config.get("trainingProfile", "reference-fp32")
        if self.profile not in ("reference-fp32", "optimized-fp32", "optimized-bf16", "compiled-bf16"):
            raise ValueError("Unknown training profile")
        self.optimized = self.profile != "reference-fp32"
        self.gradient_part_size = 2 * 1024 * 1024 if self.optimized else 512 * 1024
        self.compiled_model = self.profile == "compiled-bf16"
        self.compile_seconds = 0.0
        self.precision = "bf16" if self.profile.endswith("bf16") else "fp32"
        self.microbatch = config.get("microbatch", 16 if self.compiled_model else 4 if self.optimized else 1)
        if type(self.microbatch) is not int or self.microbatch not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("Microbatch must be a power of two from 1 to 64")
        if not self.optimized and self.microbatch != 1:
            raise ValueError("Reference profile requires microbatch 1")
        self.local_steps = config.get("localSteps", 1)
        if type(self.local_steps) is not int or self.local_steps not in (1, 2, 4, 8, 16, 32):
            raise ValueError("localSteps must be a power of two from 1 to 32")
        if not self.optimized and self.local_steps != 1:
            raise ValueError("Reference profile requires localSteps 1")
        if self.compiled_model and (device.type != "cuda" or torch.version.hip is not None):
            raise RuntimeError("Compiled BF16 currently requires one NVIDIA CUDA GPU; use optimized-bf16 for ROCm or multiple nodes")
        if self.precision == "bf16" and (
            device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("BF16 profile requires a BF16-capable CUDA/ROCm GPU")
        if not 1 <= self.target_batch <= 262144:
            raise ValueError("Invalid effective batch size")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = LanguageModel(self.cfg)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        digest = hashlib.sha256(
            Path(__file__).with_name("model.py").read_text().encode()
        )
        digest.update(Path(__file__).read_text().encode())
        digest.update(json.dumps(config["model"], sort_keys=True).encode())
        digest.update(self.profile.encode())
        if self.local_steps != 1:
            digest.update(f":localSteps={self.local_steps}".encode())
        for parameter in self.model.parameters():
            digest.update(parameter.detach().numpy().tobytes())
        self.fingerprint = digest.hexdigest()
        self.model = self.model.to(device).train()
        self.parameters = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=0.0003, betas=(0.9, 0.95), weight_decay=0.1,
            fused=True if self.optimized and device.type == "cuda" else None,
        )
        self.flat_grad = None
        if self.optimized:
            # Stable GPU gradient views allow a single transfer for a collective,
            # and no CPU gradient transfers at all for a single training node.
            self.flat_grad = torch.zeros(self.parameter_count, device=device)
            offset = 0
            for parameter in self.parameters:
                parameter.grad = self.flat_grad[offset:offset + parameter.numel()].view_as(parameter)
                offset += parameter.numel()
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.local_tokens = 0
        self.last_loss = 0.0
        self.last_compute_seconds = 0.0
        self.last_compute_tps = 0.0
        self.steps = 0
        self._prepared_participants = 0
        if self.compiled_model:
            self._warm_compile()

    def _warm_compile(self):
        print(json.dumps({"event": "model_compile_started", "microbatch": self.microbatch}), flush=True)
        started = time.perf_counter()
        state = self.generator.get_state()
        self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)
        sequences = math.ceil(self.target_batch / self.cfg.context)
        sizes = {min(self.microbatch, sequences)}
        if sequences % self.microbatch:
            sizes.add(sequences % self.microbatch)
        # Compile/capture both full and short-tail shapes before reporting Ready.
        # Weights and optimizer state never change, and the data RNG is restored.
        for batch in sorted(sizes, reverse=True):
            tokens = torch.randint(self.cfg.vocab_size, (batch, self.cfg.context + 1), device=self.device, generator=self.generator)
            for _ in range(3):
                self.flat_grad.zero_()
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = self.model(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
                loss.backward()
        torch.cuda.synchronize(self.device)
        self.flat_grad.zero_()
        self.generator.set_state(state)
        self.compile_seconds = time.perf_counter() - started
        print(json.dumps({"event": "model_compile_finished", "seconds": self.compile_seconds}), flush=True)

    def prepare(self, participants=1, *, local_tokens=None):
        if type(participants) is not int or not 1 <= participants <= 10:
            raise ValueError("Expected 1–10 training participants")
        if local_tokens is not None and (
            type(local_tokens) is not int
            or local_tokens <= 0
            or local_tokens % self.cfg.context != 0
        ):
            raise ValueError("local_tokens must be a positive multiple of context")
        self._prepared_participants = participants
        if self.optimized:
            return self._prepare_optimized(participants, local_tokens=local_tokens)
        accumulation = (
            local_tokens // self.cfg.context
            if local_tokens is not None
            else math.ceil(self.target_batch / (participants * self.cfg.context))
        )
        self.local_tokens = accumulation * self.cfg.context
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accumulation):
            tokens = torch.randint(
                self.cfg.vocab_size,
                (1, self.cfg.context + 1),
                generator=self.generator,
                device=self.device,
            )
            _, loss = self.model(
                tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite training loss")
            loss_sum += loss.detach().item()
            (loss / accumulation).backward()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(1e-6, time.perf_counter() - started)
        self.last_compute_seconds = elapsed
        self.last_compute_tps = self.local_tokens / elapsed
        self.last_loss = loss_sum / accumulation
        flat = torch.cat([p.grad.detach().reshape(-1) for p in self.parameters]).cpu()
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError("Nonfinite gradients")
        return flat

    def _prepare_optimized(self, participants, *, local_tokens=None):
        if self.compiled_model and participants != 1:
            raise ValueError("Compiled BF16 currently supports a single NVIDIA training node")
        sequences = (
            local_tokens // self.cfg.context
            if local_tokens is not None
            else math.ceil(self.target_batch / (participants * self.cfg.context))
        )
        self.local_tokens = sequences * self.cfg.context * self.local_steps
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        loss_total = 0.0
        for step_idx in range(self.local_steps):
            self.flat_grad.zero_()
            loss_sum = torch.zeros((), device=self.device)
            for offset in range(0, sequences, self.microbatch):
                if self.compiled_model:
                    torch.compiler.cudagraph_mark_step_begin()
                batch = min(self.microbatch, sequences - offset)
                tokens = torch.randint(
                    self.cfg.vocab_size, (batch, self.cfg.context + 1),
                    generator=self.generator, device=self.device,
                )
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.precision == "bf16" else nullcontext()
                )
                with context:
                    _, loss = self.model(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
                # Weight a short final microbatch correctly, preserving the exact
                # per-node effective token count for every team size from 1 to 10.
                weight = batch / sequences
                loss_sum.add_(loss.detach(), alpha=weight)
                (loss * weight).backward()
            step_loss = loss_sum.item()
            if not math.isfinite(step_loss):
                raise RuntimeError("Nonfinite training loss")
            loss_total += step_loss
            if step_idx < self.local_steps - 1:
                norm = torch.linalg.vector_norm(self.flat_grad)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("Nonfinite gradients")
                self.flat_grad.mul_((1.0 / (norm + 1e-6)).clamp(max=1.0))
                self.optimizer.step()
                self.steps += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = max(1e-6, time.perf_counter() - started)
        self.last_compute_seconds = elapsed
        self.last_compute_tps = self.local_tokens / elapsed
        self.last_loss = loss_total / self.local_steps
        if participants == 1:
            return None
        flat = self.flat_grad.cpu()
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError("Nonfinite gradients")
        return flat

    def apply(self, averaged):
        if self.optimized:
            if not self._prepared_participants or (averaged is None and self._prepared_participants != 1):
                raise RuntimeError("Prepare gradients and complete peer averaging before updating")
            if averaged is not None:
                self.flat_grad.copy_(averaged)
            norm = torch.linalg.vector_norm(self.flat_grad)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite gradients")
            self.flat_grad.mul_((1.0 / (norm + 1e-6)).clamp(max=1.0))
            self.optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.steps += 1
            self._prepared_participants = 0
            return
        offset = 0
        for p in self.parameters:
            p.grad.copy_(
                averaged[offset : offset + p.numel()].view_as(p).to(self.device)
            )
            offset += p.numel()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.steps += 1
