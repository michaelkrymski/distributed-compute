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
