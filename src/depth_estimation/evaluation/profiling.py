"""Latency and memory profiling for depth estimation models."""

from __future__ import annotations

import logging
import statistics
import time
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def profile_latency(
    model: Union[str, "BaseDepthModel"],  # noqa: F821
    input_size: int = 518,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: Optional[str] = None,
    half: bool = False,
) -> dict:
    """Measure per-image inference latency and peak GPU memory.

    GPU timings use ``torch.cuda.synchronize()`` before each clock reading
    so that CUDA kernel launches are fully completed before measurement.
    CPU timings use ``time.perf_counter()``.

    Args:
        model:      Model variant ID string (e.g. ``"depth-anything-v2-vitb"``)
                    or an already-loaded :class:`BaseDepthModel` instance.
        input_size: Spatial dimension of the square input tensor. Default 518.
        batch_size: Number of images per forward pass. Default 1.
        num_warmup: Number of forward passes to run before recording.
                    GPU caches are warm after this. Default 10.
        num_runs:   Number of timed forward passes. Default 100.
        device:     Device string (``"cuda"``, ``"cpu"``, ``"mps"``).
                    Auto-detected if ``None``.
        half:       Run in FP16 precision. Default ``False``.

    Returns:
        Dict with keys:

        ============  ========================================================
        Key           Description
        ============  ========================================================
        mean_ms       Mean latency in milliseconds per batch.
        std_ms        Standard deviation of latency (ms).
        min_ms        Minimum observed latency (ms).
        max_ms        Maximum observed latency (ms).
        p50_ms        50th-percentile latency (ms).
        p95_ms        95th-percentile latency (ms).
        p99_ms        99th-percentile latency (ms).
        fps           Throughput: ``batch_size / (mean_ms / 1000)``.
        memory_mb     Peak GPU memory allocated after warmup (MiB).
                      ``None`` on CPU / MPS.
        model_id      Model identifier string (when *model* is a string).
        device        Device string used for the run.
        input_shape   ``(batch_size, 3, input_size, input_size)``.
        ============  ========================================================

    Example::

        from depth_estimation.evaluation import profile_latency

        p = profile_latency("depth-anything-v2-vitb", input_size=518, num_runs=50)
        print(f"{p['mean_ms']:.1f} ms  |  {p['fps']:.1f} FPS")
    """
    from depth_estimation.models.auto.modeling_auto import AutoDepthModel
    from depth_estimation.modeling_utils import _auto_detect_device

    if device is None:
        device = _auto_detect_device()

    model_id: Optional[str] = None
    if isinstance(model, str):
        model_id = model
        logger.info("Loading %s for profiling …", model_id)
        loaded_model = AutoDepthModel.from_pretrained(model_id, device=device)
    else:
        loaded_model = model
        loaded_model = loaded_model.to(device)

    loaded_model.eval()
    if half:
        loaded_model = loaded_model.half()

    dtype = torch.float16 if half else torch.float32
    dummy = torch.randn(batch_size, 3, input_size, input_size, dtype=dtype, device=device)

    is_cuda = device.startswith("cuda")

    def _sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    # Reset peak memory counter
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = loaded_model(dummy)
            _sync()

    memory_mb: Optional[float] = None
    if is_cuda:
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)

    # Timed runs
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            _sync()
            t0 = time.perf_counter()
            _ = loaded_model(dummy)
            _sync()
            latencies.append((time.perf_counter() - t0) * 1000.0)  # → ms

    mean_ms = statistics.mean(latencies)
    latencies_sorted = sorted(latencies)

    def _percentile(data: list[float], p: float) -> float:
        idx = max(0, min(int(len(data) * p / 100), len(data) - 1))
        return data[idx]

    return {
        "mean_ms":    mean_ms,
        "std_ms":     statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms":     latencies_sorted[0],
        "max_ms":     latencies_sorted[-1],
        "p50_ms":     _percentile(latencies_sorted, 50),
        "p95_ms":     _percentile(latencies_sorted, 95),
        "p99_ms":     _percentile(latencies_sorted, 99),
        "fps":        batch_size / (mean_ms / 1000.0),
        "memory_mb":  memory_mb,
        "model_id":   model_id,
        "device":     device,
        "input_shape": (batch_size, 3, input_size, input_size),
    }
