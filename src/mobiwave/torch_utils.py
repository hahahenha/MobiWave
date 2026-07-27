from __future__ import annotations

from functools import lru_cache
import os
import warnings

import torch


def select_torch_device(preferred: str | torch.device | None = None) -> str:
    """Choose a torch device, falling back when an advertised backend is unusable."""

    requested = str(preferred or os.getenv("DISPATCH_DEVICE") or os.getenv("TORCH_DEVICE") or "auto").strip().lower()
    if requested in {"", "default"}:
        requested = "auto"

    if requested == "auto":
        candidates = ["cuda", "mps", "cpu"]
    else:
        candidates = [requested]
        if requested != "cpu":
            candidates.append("cpu")

    explicit_request = requested != "auto"
    failures: list[tuple[str, Exception]] = []
    for candidate in candidates:
        if candidate.startswith("cuda") and not torch.cuda.is_available():
            continue
        if candidate.startswith("mps") and not _mps_available():
            continue
        if candidate.startswith("cuda"):
            configure_cuda_backend()
        ok, error = _device_smoke_test(candidate)
        if ok:
            if candidate == "cpu" and explicit_request and failures:
                _warn_device_fallback(failures[0])
            return candidate
        if error is not None:
            failures.append((candidate, error))
    if explicit_request and failures:
        _warn_device_fallback(failures[0])
    return "cpu"


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _warn_device_fallback(failure: tuple[str, Exception]) -> None:
    failed_device, error = failure
    warnings.warn(
        f"Torch device {failed_device!r} is unavailable for model execution; using CPU. {error}",
        RuntimeWarning,
        stacklevel=3,
    )


def configure_cuda_backend() -> None:
    os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


@lru_cache(maxsize=None)
def _device_smoke_test(device_name: str) -> tuple[bool, Exception | None]:
    if device_name == "cpu":
        return True, None
    try:
        device = torch.device(device_name)
        with torch.no_grad():
            layer = torch.nn.Linear(4, 4).to(device)
            x = torch.ones((2, 4), dtype=torch.float32, device=device)
            y = layer(x)
            _ = float(y.detach().sum().cpu().item())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        if device.type == "cuda":
            network = torch.nn.Sequential(
                torch.nn.Linear(16, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 7),
            ).to(device)
            x = torch.randn((32, 16), dtype=torch.float32, device=device)
            loss = network(x).mean()
            loss.backward()
            torch.cuda.synchronize(device)
        return True, None
    except Exception as exc:
        return False, exc
