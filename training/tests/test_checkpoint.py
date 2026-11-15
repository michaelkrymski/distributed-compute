"""Checkpoint file contracts independent of the networked lifecycle smoke test."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from glmtrain.data import TokenBatches
from glmtrain.worker import load_checkpoint, save_checkpoint


class OptimizerStub:
    def __init__(self):
        self.local_epoch = 3
        self.tracker = SimpleNamespace(pause_updates=nullcontext, update_epoch=self.update_epoch)
        self.state_averager = SimpleNamespace(state_sharing_priority=0)
        self.grad_averager = SimpleNamespace(state_sharing_priority=0)

    def update_epoch(self, *, new_epoch):
        self.tracked_epoch = new_epoch

    def state_dict(self):
        return {"local_epoch": self.local_epoch}

    def load_state_dict(self, state):
        self.local_epoch = state["local_epoch"]


def test_checkpoint_roundtrip_and_sampler(tmp_path):
    tokens = tmp_path / "tokens.bin"
    np.arange(200, dtype="<u2").tofile(tokens)
    batches = TokenBatches(tokens, 8, seed=123)
    model = torch.nn.Linear(2, 2)
    optimizer = OptimizerStub()
    path = tmp_path / "latest.pt"
    manifest = {"fingerprint": "run"}
    save_checkpoint(path, model, optimizer, manifest, 0, 24, batches)
    expected_model = {k: v.clone() for k, v in model.state_dict().items()}
    expected_batch = batches.batch(4, "cpu")
    with torch.no_grad():
        model.weight.zero_()
    optimizer.local_epoch = 0
    assert load_checkpoint(path, model, optimizer, manifest, 0, batches) == 24
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
    for actual, expected in zip(batches.batch(4, "cpu"), expected_batch):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert optimizer.local_epoch == optimizer.tracked_epoch == 3
    assert optimizer.state_averager.state_sharing_priority == 3
    assert optimizer.grad_averager.state_sharing_priority == 3

    for bad_manifest, bad_shard in [({"fingerprint": "other"}, 0), (manifest, 1)]:
        with pytest.raises(ValueError, match="run bundle and data shard"):
            load_checkpoint(path, model, optimizer, bad_manifest, bad_shard, batches)


def test_checkpoint_replace_failure_preserves_previous(tmp_path, monkeypatch):
    tokens = tmp_path / "tokens.bin"
    np.arange(100, dtype="<u2").tofile(tokens)
    batches = TokenBatches(tokens, 8)
    path = tmp_path / "latest.pt"
    model, optimizer, manifest = torch.nn.Linear(2, 2), OptimizerStub(), {"fingerprint": "run"}
    save_checkpoint(path, model, optimizer, manifest, 0, 8, batches)
    original = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("glmtrain.worker.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        save_checkpoint(path, model, optimizer, manifest, 0, 16, batches)
    assert path.read_bytes() == original
    assert torch.load(path, weights_only=True)["local_tokens"] == 8
