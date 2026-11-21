from types import SimpleNamespace

import pytest
import torch
from glmbench.device import select_device, describe


def test_cpu_is_explicit_and_reports_hardware():
    device, dtype = select_device("cpu")
    assert device.type == "cpu" and dtype == torch.float32
    info = describe(device)
    assert info["kind"] == "cpu" and info["name"] and "hostname" in info


def test_invalid_device_selection_fails_before_benchmark():
    with pytest.raises(ValueError):
        select_device("cpu", -1)
    with pytest.raises(ValueError):
        select_device("unknown")
    if not torch.version.hip:
        with pytest.raises(RuntimeError, match="not a ROCm"):
            select_device("rocm")


@pytest.mark.parametrize(
    ("arch", "dtype"), [("gfx1030", torch.float32), ("gfx942:sramecc+", torch.bfloat16)]
)
def test_rocm_precision_uses_requested_gpu_architecture(monkeypatch, arch, dtype):
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)

    def properties(device):
        assert device == torch.device("cuda", 1)
        return SimpleNamespace(gcnArchName=arch)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert select_device("rocm", 1) == (torch.device("cuda", 1), dtype)
