import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from glmtrain.model import ModelConfig, LanguageModel
from glmtrain.data import TokenBatches, prepare, load_bundle
from glmtrain.worker import (
    ClippedAdamW,
    _run_accum_step,
    evaluate,
    lr_multiplier,
    reroute_work_and_shards,
)


def test_parameter_count_matches_dashboard():
    c = ModelConfig()
    model = LanguageModel(c)
    assert sum(p.numel() for p in model.parameters()) == c.parameter_count() == 24388992
    untied = replace(c, tied=False)
    assert (
        sum(p.numel() for p in LanguageModel(untied).parameters())
        == c.parameter_count() + 8192 * 384
    )


def test_causal_attention_and_finite_backward():
    torch.manual_seed(42)
    model = LanguageModel(
        ModelConfig(vocab_size=257, width=16, layers=2, heads=2, context=8)
    )
    x = torch.randint(0, 257, (2, 8))
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 257
    a, loss = model(x, x.roll(-1, 1))
    b, _ = model(changed)
    torch.testing.assert_close(a[:, :4], b[:, :4])
    assert not torch.allclose(a[:, 4:], b[:, 4:])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_disjoint_shards_and_shifted_labels(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(200, dtype="<u2").tofile(path)
    for shard in range(5):
        batches = TokenBatches(path, 8, shard=shard, shards=5, seed=42)
        x, y = batches.batch(20, "cpu")
        assert torch.all(y == x + 1)
        assert x.min() >= shard * 40 and y.max() < (shard + 1) * 40
    with pytest.raises(ValueError):
        TokenBatches(path, 8, shard=5, shards=5)


@pytest.fixture
def bundle(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "validation.txt"
    train.write_text(
        ("A little gator learned a different word each morning.\n" * 100),
        encoding="utf8",
    )
    val.write_text(("The held-out bird flew past a blue lake.\n" * 30), encoding="utf8")
    config = Path(__file__).parents[1] / "configs" / "smoke.json"
    directory = tmp_path / "bundle"
    manifest = prepare(train, val, config, directory)
    return directory, manifest


def test_run_bundle_and_byte_roundtrip(bundle):
    from tokenizers import Tokenizer

    directory, manifest = bundle
    assert load_bundle(directory) == manifest
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    text = " Gator 🐊\t中文\nΩ"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_reject_modified_run_config(bundle):
    directory, manifest = bundle
    manifest["training"]["learning_rate"] = 0.9
    (directory / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed"):
        load_bundle(directory)


def test_reject_modified_tokens(bundle):
    directory, _ = bundle
    with open(directory / "train.bin", "ab") as f:
        f.write(b"\x00\x00")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_bundle(directory)


def test_reject_validation_as_training(tmp_path):
    path = tmp_path / "same.txt"
    path.write_text("same")
    with pytest.raises(ValueError, match="different document splits"):
        prepare(
            path,
            path,
            Path(__file__).parents[1] / "configs" / "smoke.json",
            tmp_path / "bundle",
        )


def test_orphan_shard_rerouting_and_multi_shard_batches(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(200, dtype="<u2").tofile(path)
    batches = TokenBatches(path, 8, shard=0, shards=5, seed=42)
    manifest = {
        "model": {"context": 8},
        "training": {"data_shards": 5, "target_batch_tokens": 64},
    }
    active = {
        "peer-a": {"shard": 0, "compute_tps": 1000.0},
        "peer-b": {"shard": 2, "compute_tps": 500.0},
    }
    tokens_a, count, shards_a = reroute_work_and_shards(
        active, manifest, "peer-a", 0, batches, 1
    )
    tokens_b, _, shards_b = reroute_work_and_shards(
        active, manifest, "peer-b", 2, None, 1
    )
    assert count == 2
    assert shards_a == [0, 1, 4]
    assert shards_b == [2, 3]
    assert sorted(shards_a + shards_b) == [0, 1, 2, 3, 4]
    assert tokens_a > tokens_b
    assert tokens_a + tokens_b == 64
    x, y = batches.batch(16, "cpu")
    assert torch.all(y == x + 1)
    with pytest.raises(ValueError):
        batches.set_active_shards([5])


def test_deterministic_evaluate_clipped_adamw_and_lr_schedule(tmp_path):
    path = tmp_path / "val.bin"
    np.arange(200, dtype="<u2").tofile(path)
    val = TokenBatches(path, 8, seed=12345)
    model = LanguageModel(ModelConfig(vocab_size=257, width=16, layers=1, heads=2, context=8))
    first = evaluate(model, val, torch.device("cpu"), count=4)
    _ = val.batch(3, "cpu")
    second = evaluate(model, val, torch.device("cpu"), count=4)
    assert first == pytest.approx(second, rel=1e-7, abs=1e-7)

    opt = ClippedAdamW(model.parameters(), lr=1e-3, max_norm=0.5)
    x, y = val.batch(2, "cpu")
    _, loss = model(x, y)
    loss.backward()
    opt.step()
    assert opt.last_grad_norm > 0
    loss_sum, tokens, elapsed = _run_accum_step(model, val, 2, torch.device("cpu"), "fp32", 2)
    assert tokens == 2 * 2 * 8
    assert loss_sum > 0 and elapsed > 0
    assert lr_multiplier(0, 100) < lr_multiplier(1, 100) == pytest.approx(1.0)
    assert lr_multiplier(100, 100) == pytest.approx(0.1)
