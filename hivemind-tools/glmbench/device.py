"""Explicit NVIDIA/AMD selection and a real, local GPU/CPU smoke test."""

import json
import platform
import torch


def select_device(kind="auto", index=0):
    if index < 0:
        raise ValueError("GPU index must be nonnegative")
    if kind not in ("auto", "cpu", "cuda", "rocm"):
        raise ValueError("Choose auto, cpu, cuda, or rocm")
    hip = bool(torch.version.hip)
    if kind == "rocm" and not hip:
        raise RuntimeError("ROCm requested but this is not a ROCm PyTorch build")
    if kind == "cuda" and hip:
        raise RuntimeError("This is an AMD ROCm build; choose rocm or auto")
    gpu = kind in ("cuda", "rocm") or (kind == "auto" and torch.cuda.is_available())
    if gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU unavailable: check PyTorch CUDA/ROCm build and drivers, or select cpu"
        )
    if gpu and index >= torch.cuda.device_count():
        raise ValueError("GPU index exceeds the number of visible devices")
    device = torch.device("cuda", index) if gpu else torch.device("cpu")
    # PyTorch 2.8 reports BF16 support unconditionally on ROCm, but BF16
    # matmul can crash on RDNA 1/2 (gfx10), e.g. the RX 6800 XT.
    rdna1_or_2 = (
        gpu and hip
        and torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx10")
    )
    use_bf16 = gpu and not rdna1_or_2 and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    return device, dtype


def describe(device):
    return {
        "name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "CPU",
        "kind": f"rocm:{device.index}"
        if torch.version.hip and device.type == "cuda"
        else str(device),
        "hostname": platform.node()[:180],
        "torch": torch.__version__,
        "platform": platform.system(),
    }


def doctor(args):
    device, dtype = select_device(args.device, args.device_index)
    x = torch.randn(1024, 1024, device=device, dtype=dtype, requires_grad=True)
    y = (x @ x).float().square().mean()
    y.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    if not bool(torch.isfinite(y)) or not bool(torch.isfinite(x.grad).all()):
        raise RuntimeError("Device smoke test produced non-finite values")
    print(
        json.dumps(
            {
                **describe(device),
                "hip": torch.version.hip,
                "dtype": str(dtype),
                "forwardBackward": "passed",
                "scope": "Local device smoke test only; not model training or peer connectivity",
            }
        )
    )
