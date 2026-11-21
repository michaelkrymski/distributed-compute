"""FP16 benchmark with FP32 master weights and pre-collective overflow recovery."""

import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .benchmark import TrainingWorkload


class FP16TrainingWorkload(TrainingWorkload):
    def __init__(self, config, device, seed):
        if device.type != "cuda" or config.get("localSteps", 1) != 1:
            raise ValueError("Optimized FP16 requires a CUDA/ROCm GPU and Local steps = 1")
        super().__init__({**config, "trainingProfile": "optimized-fp32"}, device, seed)
        self.profile = "optimized-fp16"
        self.precision = "fp16"
        self.loss_scale = 1024.0
        digest = hashlib.sha256(self.fingerprint.encode())
        digest.update(Path(__file__).read_text().encode())
        self.fingerprint = digest.hexdigest()

    def _prepare_optimized(self, participants, *, local_tokens=None):
        sequences = (
            local_tokens // self.cfg.context
            if local_tokens is not None
            else math.ceil(self.target_batch / (participants * self.cfg.context))
        )
        self.local_tokens = sequences * self.cfg.context
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # Replay the same batch with a lower scale before entering the
        # collective. Parameters and optimizer state stay untouched on retry.
        rng_state = self.generator.get_state()
        while True:
            loss = self._scaled_backward(sequences)
            self.flat_grad.div_(self.loss_scale)
            if bool(torch.isfinite(torch.linalg.vector_norm(self.flat_grad))):
                break
            if self.loss_scale <= 1:
                raise RuntimeError("FP16 gradient overflow at minimum loss scale; try BF16 or FP32")
            self.loss_scale *= 0.5
            self.generator.set_state(rng_state)
            print(json.dumps({"event": "fp16_overflow_retry", "lossScale": self.loss_scale}), flush=True)
        self.last_compute_seconds = max(1e-6, time.perf_counter() - started)
        self.last_compute_tps = self.local_tokens / self.last_compute_seconds
        self.last_loss = loss
        return None if participants == 1 else self.flat_grad.cpu()

    def _scaled_backward(self, sequences):
        self.flat_grad.zero_()
        loss_sum = torch.zeros((), device=self.device)
        for offset in range(0, sequences, self.microbatch):
            batch = min(self.microbatch, sequences - offset)
            tokens = torch.randint(
                self.cfg.vocab_size, (batch, self.cfg.context + 1),
                generator=self.generator, device=self.device,
            )
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = self.model(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
            weight = batch / sequences
            loss_sum.add_(loss.detach(), alpha=weight)
            (loss * (weight * self.loss_scale)).backward()
        loss = loss_sum.item()
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite FP16 training loss; try BF16 or FP32")
        return loss
