"""Immutable run bundles and fixed data shards; no dataset downloads at runtime."""

import hashlib
import json
import math
import uuid
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .model import ModelConfig

FORMAT_VERSION = 1
FILES = ("tokenizer.json", "train.bin", "validation.bin")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hash(manifest):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "fingerprint"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_training(c, model):
    if set(c) != {
        "seed",
        "learning_rate",
        "weight_decay",
        "target_batch_tokens",
        "target_tokens",
        "data_shards",
    }:
        raise ValueError("Unexpected or missing training configuration keys")
    for key in ("seed", "target_batch_tokens", "target_tokens", "data_shards"):
        if type(c[key]) is not int or c[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    for key in ("learning_rate", "weight_decay"):
        if (
            isinstance(c[key], bool)
            or not isinstance(c[key], (int, float))
            or not math.isfinite(c[key])
            or c[key] < (1e-12 if key == "learning_rate" else 0)
        ):
            raise ValueError(f"Invalid {key}")
    if c["target_batch_tokens"] % model.context:
        raise ValueError("target_batch_tokens must be a multiple of context")


def encode_file(tokenizer, source, target, batch_lines=4096):
    count = 0
    eos = tokenizer.token_to_id("<|endoftext|>")

    def flush(chunk, binary):
        nonlocal count
        if not chunk:
            return
        flat = []
        for enc in tokenizer.encode_batch(chunk):
            flat.extend(enc.ids)
            flat.append(eos)
        arr = np.asarray(flat, dtype="<u2")
        arr.tofile(binary)
        count += int(arr.size)

    with open(source, encoding="utf-8") as text, open(target, "wb") as binary:
        chunk = []
        for line in text:
            if not line.strip():
                continue
            chunk.append(line.rstrip("\r\n"))
            if len(chunk) >= batch_lines:
                flush(chunk, binary)
                chunk.clear()
        flush(chunk, binary)
    return count


def prepare(train_path, validation_path, config_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory")
    if Path(train_path).resolve() == Path(validation_path).resolve() or file_hash(
        train_path
    ) == file_hash(validation_path):
        raise ValueError("Training and validation must be different document splits")
    config = json.loads(Path(config_path).read_text())
    model = ModelConfig(**config["model"])
    validate_training(config["training"], model)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=model.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(train_path)], trainer)
    output.mkdir(parents=True)
    tokenizer.save(str(output / "tokenizer.json"))
    counts = {
        "train": encode_file(tokenizer, train_path, output / "train.bin"),
        "validation": encode_file(
            tokenizer, validation_path, output / "validation.bin"
        ),
    }
    model = ModelConfig(**{**asdict(model), "vocab_size": tokenizer.get_vocab_size()})
    if (
        counts["train"] < config["training"]["data_shards"] * (model.context + 1)
        or counts["validation"] < model.context + 1
    ):
        raise ValueError(
            "Not enough tokens for every data shard and a validation window; use more text or a smaller smoke config"
        )
    manifest = {
        "format_version": FORMAT_VERSION,
        "run_id": "glm0-" + uuid.uuid4().hex[:16],
        "model": asdict(model),
        "training": config["training"],
        "counts": counts,
        "sha256": {name: file_hash(output / name) for name in FILES},
    }
    manifest["fingerprint"] = contract_hash(manifest)
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get(
        "fingerprint"
    ) != contract_hash(manifest):
        raise ValueError(
            "Run manifest changed or has an unsupported format; distribute an unchanged bundle"
        )
    model = ModelConfig(**manifest["model"])
    validate_training(manifest["training"], model)
    if set(manifest["sha256"]) != set(FILES):
        raise ValueError("Invalid bundle file list")
    for name in FILES:
        if file_hash(directory / name) != manifest["sha256"][name]:
            raise ValueError(
                f"Checksum mismatch: {name}; obtain the same run bundle as your teammates"
            )
    for split in ("train", "validation"):
        if (directory / f"{split}.bin").stat().st_size != manifest["counts"][split] * 2:
            raise ValueError(f"Invalid token count for {split}")
        tokens = np.memmap(directory / f"{split}.bin", dtype="<u2", mode="r")
        if int(tokens.max()) >= model.vocab_size:
            raise ValueError(f"Out-of-vocabulary token in {split}")
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if tokenizer.get_vocab_size() != model.vocab_size:
        raise ValueError("Tokenizer vocabulary differs from model vocabulary")
    return manifest


class TokenBatches:
    def __init__(self, path, context, *, shard=0, shards=1, seed=0):
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        if not 0 <= shard < shards:
            raise ValueError("shard must be in [0, data_shards)")
        self.shard = shard
        self.shards = shards
        # Disjoint contiguous token regions. A shifted target never crosses a shard.
        self.start = len(self.tokens) * shard // shards
        self.end = len(self.tokens) * (shard + 1) // shards
        self.context = context
        if self.end - self.start < context + 1:
            raise ValueError("Shard is too short for one context plus its target")
        self.active_shards = (shard,)
        self.active_ranges = ((self.start, self.end),)
        self.rng = np.random.default_rng(seed)

    def set_active_shards(self, shard_indices):
        unique = tuple(sorted({int(s) for s in shard_indices}))
        if not unique or any(not 0 <= s < self.shards for s in unique):
            raise ValueError("Active shards must be valid indices in [0, data_shards)")
        ranges = []
        for s in unique:
            start = len(self.tokens) * s // self.shards
            end = len(self.tokens) * (s + 1) // self.shards
            if end - start < self.context + 1:
                raise ValueError("Shard is too short for one context plus its target")
            ranges.append((start, end))
        self.active_shards = unique
        self.active_ranges = tuple(ranges)

    def batch(self, size, device):
        if len(self.active_ranges) == 1:
            start, end = self.active_ranges[0]
            starts = self.rng.integers(start, end - self.context, size=size)
        else:
            choices = self.rng.integers(0, len(self.active_ranges), size=size)
            starts = np.empty(size, dtype=np.int64)
            for i, idx in enumerate(choices):
                start, end = self.active_ranges[int(idx)]
                starts[i] = self.rng.integers(start, end - self.context)
        data = np.stack([self.tokens[s : s + self.context + 1] for s in starts]).astype(
            np.int64
        )
        return torch.from_numpy(data[:, :-1].copy()).to(device), torch.from_numpy(
            data[:, 1:].copy()
        ).to(device)
