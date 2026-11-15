"""One full model replica per worker, synchronized by hivemind.Optimizer."""

import json
import math
import os
import time
from pathlib import Path
from functools import partial
from contextlib import nullcontext
import hivemind
from hivemind.utils import get_dht_time
from hivemind.compression import Float16Compression
from hivemind.optim.grad_averager import GradientAverager
import numpy as np
import torch
from .benchmark import allocate_proportional_tokens
from .data import TokenBatches
from .model import LanguageModel, ModelConfig
from .network import start_dht, check_contract, workers, Heartbeat, key, calibrate_clock


class ClippedAdamW(torch.optim.AdamW):
    """AdamW that clips global post-averaged gradients right before stepping."""

    def __init__(self, params, max_norm=1.0, **kwargs):
        super().__init__(params, **kwargs)
        self.max_norm = max_norm
        self.last_grad_norm = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if params:
            norm = torch.nn.utils.clip_grad_norm_(
                params, self.max_norm, error_if_nonfinite=True
            )
            self.last_grad_norm = float(norm)
        else:
            self.last_grad_norm = 0.0
        return super().step(closure=closure)


def lr_multiplier(epoch, target_epochs):
    if target_epochs <= 4:
        return 1.0
    warmup_epochs = max(1, int(0.02 * target_epochs))
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = min(
        1.0, max(0.0, (epoch - warmup_epochs) / max(1, target_epochs - warmup_epochs))
    )
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(torch_optimizer, target_epochs):
    return torch.optim.lr_scheduler.LambdaLR(
        torch_optimizer,
        lr_lambda=partial(lr_multiplier, target_epochs=target_epochs),
    )


def reroute_work_and_shards(
    active_workers,
    manifest,
    worker_id,
    primary_shard,
    batches,
    microbatch,
):
    active = dict(active_workers)
    if worker_id not in active:
        active[worker_id] = {"shard": primary_shard}
    data_shards = manifest["training"]["data_shards"]
    claimed = {
        info.get("shard")
        for info in active.values()
        if isinstance(info.get("shard"), int)
    }
    orphans = [s for s in range(data_shards) if s not in claimed]
    peers = sorted(active.items(), key=lambda item: (item[1].get("shard", 0), item[0]))
    my_index = next(i for i, (pid, _) in enumerate(peers) if pid == worker_id)
    assigned_shards = [primary_shard] + [
        s for idx, s in enumerate(orphans) if idx % len(peers) == my_index
    ]
    unique_shards = tuple(sorted(set(assigned_shards)))
    if batches is not None and unique_shards != batches.active_shards:
        batches.set_active_shards(unique_shards)
    speeds = {
        pid: float(info["compute_tps"])
        for pid, info in peers
        if isinstance(info.get("compute_tps"), (int, float))
        and math.isfinite(info["compute_tps"])
        and info["compute_tps"] > 0
    }
    schedule = allocate_proportional_tokens(
        manifest["training"]["target_batch_tokens"],
        manifest["model"]["context"],
        [pid for pid, _ in peers],
        speeds,
        microbatch,
    )
    return schedule[worker_id], len(peers), list(unique_shards)


def save_checkpoint(path, model, optimizer, manifest, shard, local_tokens, batches):
    # Only called after a complete Hivemind epoch, never during accumulation.
    checkpoint = {
        "fingerprint": manifest["fingerprint"],
        "shard": shard,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "local_tokens": local_tokens,
        "rng": batches.rng.bit_generator.state,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, manifest, shard, batches):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["fingerprint"] != manifest["fingerprint"] or state["shard"] != shard:
        raise ValueError("Checkpoint does not belong to this run bundle and data shard")
    with optimizer.tracker.pause_updates():
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer.tracker.update_epoch(new_epoch=optimizer.local_epoch)
        if hasattr(optimizer.tracker, "global_progress") and hasattr(
            optimizer.tracker, "default_refresh_period"
        ):
            optimizer.tracker.global_progress.next_fetch_time = (
                get_dht_time() + optimizer.tracker.default_refresh_period
            )
            if hasattr(optimizer.tracker, "global_state_updated"):
                optimizer.tracker.global_state_updated.set()
    if hasattr(optimizer.state_averager, "_update_scheduler"):
        optimizer.state_averager._update_scheduler()
    optimizer.state_averager.state_sharing_priority = optimizer.local_epoch
    optimizer.grad_averager.state_sharing_priority = optimizer.local_epoch
    batches.rng.bit_generator.state = state["rng"]
    return state["local_tokens"]


@torch.no_grad()
def evaluate(model, batches, device, count=4, precision="fp32"):
    model.eval()
    saved_state = batches.rng.bit_generator.state
    try:
        batches.rng = np.random.default_rng(12345)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bf16" and device.type == "cuda"
            else nullcontext()
        )
        total = 0.0
        for _ in range(count):
            x, y = batches.batch(1, device)
            with autocast:
                _, loss = model(x, y)
            total += loss.item()
        return total / count
    finally:
        batches.rng.bit_generator.state = saved_state
        model.train()


def _run_accum_step(model, batches, microbatch, device, precision, steps):
    accum_loss = 0.0
    accum_tokens = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = batches.batch(microbatch, device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else nullcontext()
        )
        with autocast:
            _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Nonfinite loss; stop this worker and inspect the run"
            )
        (loss / steps).backward()
        accum_loss += loss.item()
        accum_tokens += x.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(1e-6, time.perf_counter() - t0)
    return accum_loss, accum_tokens, elapsed


def train(args, manifest):
    local_steps = int(
        args.local_steps
        if getattr(args, "local_steps", None) is not None
        else manifest["training"].get("local_steps", 1)
    )
    if (
        args.microbatch < 1
        or local_steps < 1
        or args.threads < 1
        or args.checkpoint_every < 1
        or args.min_peers < 1
        or args.max_microsteps is not None
        and args.max_microsteps < 1
    ):
        raise ValueError(
            "Batch size, local steps, threads, checkpoint interval, min peers, and optional microstep limit must be positive"
        )
    cfg = ModelConfig(**manifest["model"])
    settings = manifest["training"]
    if args.microbatch * cfg.context > settings["target_batch_tokens"]:
        raise ValueError(
            "Local microbatch exceeds the global target batch; lower --microbatch"
        )
    batches = TokenBatches(
        Path(args.bundle) / "train.bin",
        cfg.context,
        shard=args.shard,
        shards=settings["data_shards"],
        seed=settings["seed"] + args.shard,
    )
    validation = TokenBatches(
        Path(args.bundle) / "validation.bin", cfg.context, seed=12345
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "latest.pt"
    if checkpoint.exists() and not args.resume:
        raise ValueError(
            "Output contains a checkpoint. Use --resume or choose a different output directory"
        )
    if args.resume and not checkpoint.exists():
        raise ValueError("--resume requires latest.pt in the output directory")
    torch.set_num_threads(args.threads)
    torch.manual_seed(settings["seed"])
    # Start fork-based networking with synchronized clock before initializing CUDA.
    clock = calibrate_clock()
    dht = start_dht(args.host, args.port, args.peer)
    optimizer = None
    try:
        check_contract(dht, manifest, clock=clock)
        active = workers(dht, manifest)
        if any(w.get("shard") == args.shard for w in active.values()):
            raise ValueError("This shard is already active. Choose an unclaimed shard")
        if args.start_new and (active or args.resume):
            raise ValueError("--start-new is only for the first worker of a fresh run")
        if not args.start_new and not args.resume:
            deadline = time.monotonic() + args.join_timeout
            while not active and time.monotonic() < deadline:
                time.sleep(0.5)
                active = workers(dht, manifest)
            if not active:
                raise RuntimeError(
                    "No live training worker. Start the first with --start-new, or recover a saved checkpoint using --resume"
                )
        # Initialization happens on CPU so peers with different CUDA devices start alike.
        model = LanguageModel(cfg)
        device = torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if args.device == "auto"
            else args.device
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but unavailable; install the matching PyTorch CUDA wheel"
            )
        if args.precision == "bf16" and (
            device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError(
                "--precision bf16 requires a BF16-capable CUDA GPU; use fp32 on this device"
            )
        model = model.to(device).train()
        if device.type == "cuda":
            x_w = torch.zeros((1, cfg.context), dtype=torch.long, device=device)
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if args.precision == "bf16"
                else nullcontext()
            ):
                _, w_loss = model(x_w, x_w)
            w_loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            del x_w, w_loss
        target_epochs = math.ceil(
            settings["target_tokens"] / settings["target_batch_tokens"]
        )
        decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": decay_params, "weight_decay": settings["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = hivemind.Optimizer(
            dht=dht,
            run_id=manifest["run_id"] + "_" + manifest["fingerprint"][:12],
            params=param_groups,
            optimizer=partial(
                ClippedAdamW,
                lr=settings["learning_rate"],
                betas=(0.9, 0.95),
                weight_decay=settings["weight_decay"],
                fused=True if device.type == "cuda" else None,
            ),
            scheduler=partial(make_scheduler, target_epochs=target_epochs),
            target_batch_size=settings["target_batch_tokens"],
            batch_size_per_step=args.microbatch * cfg.context,
            offload_optimizer=False,
            reuse_grad_buffers=False,
            use_local_updates=False,
            delay_optimizer_step=False,
            delay_grad_averaging=False,
            delay_state_averaging=False,
            grad_compression=Float16Compression(),
            state_averaging_compression=Float16Compression(),
            grad_averager_factory=partial(
                GradientAverager,
                request_timeout=min(3.0, args.matchmaking_time * 0.75),
                part_size_bytes=2 * 1024 * 1024,
            ),
            averager_opts={
                "request_timeout": min(3.0, args.matchmaking_time * 0.75),
                "part_size_bytes": 2 * 1024 * 1024,
            },
            matchmaking_time=args.matchmaking_time,
            averaging_timeout=args.averaging_timeout,
            next_chunk_timeout=max(10.0, float(args.join_timeout)),
            load_state_timeout=args.join_timeout,
            tracker_opts={
                "min_refresh_period": 0.2,
                "default_refresh_period": 1,
                "max_refresh_period": 5,
            },
            verbose=True,
        )
        local_tokens = (
            load_checkpoint(checkpoint, model, optimizer, manifest, args.shard, batches)
            if args.resume
            else 0
        )
        if active:
            # Hivemind loads model, optimizer, and global epoch together; never join with random weights.
            optimizer.load_state_from_peers()
            latest = workers(dht, manifest)
            if not latest:
                raise RuntimeError(
                    "Live training workers disappeared during state transfer; retry joining or resume a checkpoint"
                )
            advertised_epoch = max(
                w["epoch"] for w in [*active.values(), *latest.values()]
            )
            if optimizer.local_epoch < advertised_epoch:
                raise RuntimeError(
                    "State transfer did not catch up to the advertised epoch; retry joining"
                )
        worker_id = str(dht.peer_id)
        compute_tps = 0.0

        def announce():
            current = workers(dht, manifest)
            if any(
                peer != worker_id and info.get("shard") == args.shard
                for peer, info in current.items()
            ):
                raise RuntimeError(
                    "A duplicate data shard joined the run; coordinate shard numbers with the team"
                )
            dht.store(
                key(manifest, "workers"),
                {
                    "fingerprint": manifest["fingerprint"],
                    "name": args.name,
                    "shard": args.shard,
                    "epoch": optimizer.local_epoch,
                    "local_tokens": local_tokens,
                    "compute_tps": round(compute_tps, 2) if compute_tps > 0 else 0.0,
                    "updated_at": get_dht_time(),
                },
                subkey=worker_id,
                expiration_time=get_dht_time() + 45,
            )

        announce()
        print(
            json.dumps(
                {
                    "event": "ready",
                    "worker": args.name,
                    "peer_id": worker_id,
                    "shard": args.shard,
                    "device": str(device),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "epoch": optimizer.local_epoch,
                    "target_epochs": target_epochs,
                }
            ),
            flush=True,
        )
        with (
            Heartbeat(announce) as heartbeat,
            open(output / "metrics.jsonl", "a", buffering=1) as metrics,
        ):
            deadline = time.monotonic() + args.join_timeout
            while len(workers(dht, manifest)) < args.min_peers:
                heartbeat.check()
                if time.monotonic() > deadline:
                    raise RuntimeError("Timed out waiting for --min-peers")
                time.sleep(0.5)
            microsteps = 0
            local_steps_done = 0
            local_updates = 0
            last_saved_epoch = optimizer.local_epoch
            microbatch_tokens = args.microbatch * cfg.context
            eval_count = 16 if device.type == "cuda" else 4
            while optimizer.local_epoch < target_epochs and (
                args.max_microsteps is None or microsteps < args.max_microsteps
            ):
                heartbeat.check()
                before = optimizer.local_epoch
                current_workers = workers(dht, manifest)
                worker_share_tokens, active_count, assigned_shards = reroute_work_and_shards(
                    current_workers,
                    manifest,
                    worker_id,
                    args.shard,
                    batches,
                    args.microbatch,
                )
                base_opt = optimizer.state_averager.optimizer
                while local_steps_done < local_steps - 1 and (
                    args.max_microsteps is None or microsteps < args.max_microsteps
                ):
                    heartbeat.check()
                    local_accum = max(1, worker_share_tokens // microbatch_tokens)
                    if args.max_microsteps is not None:
                        local_accum = min(local_accum, args.max_microsteps - microsteps)
                    if local_accum <= 0:
                        break
                    optimizer.zero_grad()
                    _, step_tokens, elapsed = _run_accum_step(
                        model, batches, args.microbatch, device, args.precision, local_accum
                    )
                    last_tps = step_tokens / elapsed
                    compute_tps = last_tps if compute_tps <= 0 else (0.7 * compute_tps + 0.3 * last_tps)
                    base_opt.step()
                    optimizer.zero_grad()
                    local_tokens += step_tokens
                    microsteps += local_accum
                    local_steps_done += 1
                    local_updates += 1
                accum_loss = 0.0
                accum_steps = 1
                if args.max_microsteps is None or microsteps < args.max_microsteps:
                    max_chunk = max(1, (worker_share_tokens // microbatch_tokens) // 2)
                    accum_steps = min(8, max_chunk) if device.type == "cuda" else 1
                    if args.max_microsteps is not None:
                        accum_steps = min(accum_steps, args.max_microsteps - microsteps)
                    accum_steps = max(1, accum_steps)

                    optimizer.zero_grad()
                    accum_loss, accum_tokens, elapsed = _run_accum_step(
                        model, batches, args.microbatch, device, args.precision, accum_steps
                    )
                    last_tps = accum_tokens / elapsed
                    compute_tps = last_tps if compute_tps <= 0 else (0.7 * compute_tps + 0.3 * last_tps)
                    optimizer.step(batch_size=accum_tokens)
                    if (
                        optimizer.local_epoch == before
                        and optimizer.grad_averager.local_samples_accumulated
                        >= settings["target_batch_tokens"]
                    ):
                        with optimizer.tracker.pause_updates():
                            optimizer.tracker.global_progress.samples_accumulated = max(
                                optimizer.tracker.global_progress.samples_accumulated,
                                optimizer.grad_averager.local_samples_accumulated,
                            )
                        if optimizer.tracker.ready_to_update_epoch:
                            optimizer._update_global_epoch(None)
                    local_tokens += accum_tokens
                    microsteps += accum_steps
                if optimizer.local_epoch != before:
                    local_steps_done = 0
                    local_updates += 1
                    row = {
                        "event": "epoch",
                        "time": get_dht_time(),
                        "name": args.name,
                        "epoch": optimizer.local_epoch,
                        "local_steps": local_steps,
                        "local_updates": local_updates,
                        "nominal_global_tokens": optimizer.local_epoch
                        * settings["target_batch_tokens"],
                        "local_tokens": local_tokens,
                        "loss": accum_loss / accum_steps,
                        "gradient_norm": float(getattr(base_opt, "last_grad_norm", 0.0)),
                        "lr": float(base_opt.param_groups[0]["lr"]),
                        "assigned_shards": assigned_shards,
                        "visible_workers": active_count,
                    }
                    if (
                        optimizer.local_epoch % args.checkpoint_every == 0
                        or optimizer.local_epoch >= target_epochs
                    ):
                        with optimizer.tracker.performance_ema.pause():
                            row["validation_loss"] = evaluate(
                                model,
                                validation,
                                device,
                                count=eval_count,
                                precision=args.precision,
                            )
                            save_checkpoint(
                                checkpoint,
                                model,
                                optimizer,
                                manifest,
                                args.shard,
                                local_tokens,
                                batches,
                            )
                        optimizer.tracker.report_local_progress(optimizer.local_epoch, 0)
                        last_saved_epoch = optimizer.local_epoch
                    metrics.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    announce()
                # Save only at an epoch boundary; incomplete accumulation is not a resumable state.
                if (
                    args.max_microsteps
                    and microsteps >= args.max_microsteps
                    and optimizer.local_epoch == before
                ):
                    print(
                        "Stopped mid-accumulation; use the last completed-epoch checkpoint, if any.",
                        flush=True,
                    )
            if last_saved_epoch < optimizer.local_epoch:
                # Do not capture partial gradients from later microsteps in optimizer state.
                print(
                    "Latest checkpoint may trail current progress; checkpoints are made only at configured epoch boundaries.",
                    flush=True,
                )
    finally:
        try:
            if optimizer is not None:
                try:
                    dht.store(
                        key(manifest, "workers"),
                        None,
                        subkey=str(dht.peer_id),
                        expiration_time=get_dht_time() + 45,
                    )
                finally:
                    optimizer.shutdown()
        finally:
            dht.shutdown()
