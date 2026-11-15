"""CPU-safe checks of selection/error handling, not GPU execution tests."""

import pytest
import torch
from smoke_gpu import select_gpu


@pytest.fixture
def gpu_build(monkeypatch):
    def configure(backend, *, available=True, count=1):
        monkeypatch.setattr(torch.version, "hip", "6.4" if backend == "rocm" else None)
        monkeypatch.setattr(torch.version, "cuda", "12.8" if backend == "cuda" else None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    return configure


@pytest.mark.parametrize("backend", ["rocm", "cuda"])
@pytest.mark.parametrize("requested", ["auto", "explicit"])
def test_select_gpu(gpu_build, backend, requested):
    gpu_build(backend, count=2)
    device, detected = select_gpu(backend if requested == "explicit" else "auto", 1)
    assert str(device) == "cuda:1" and detected == backend


@pytest.mark.parametrize("backend", ["rocm", "cuda"])
def test_reject_wrong_build(gpu_build, backend):
    gpu_build(backend)
    other = "cuda" if backend == "rocm" else "rocm"
    with pytest.raises(RuntimeError, match="installed PyTorch uses"):
        select_gpu(other)


def test_reject_cpu_build(gpu_build):
    gpu_build(None, available=False, count=0)
    with pytest.raises(RuntimeError, match="CPU-only"):
        select_gpu()


def test_reject_inaccessible_gpu(gpu_build):
    gpu_build("rocm", available=False)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        select_gpu()


def test_reject_missing_device(gpu_build):
    gpu_build("rocm")
    with pytest.raises(ValueError, match="visible GPUs"):
        select_gpu(index=1)


@pytest.mark.parametrize("backend,index", [("auto", -1), ("cpu", 0)])
def test_reject_invalid_selection(backend, index):
    with pytest.raises(ValueError):
        select_gpu(backend, index)
