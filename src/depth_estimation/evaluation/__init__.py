"""depth_estimation.evaluation — Evaluation, metrics, and profiling.

Quick start
-----------

**Evaluate one model on a dataset:**

.. code-block:: python

    from depth_estimation.evaluation import evaluate

    results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2", split="test")
    print(f"AbsRel: {results['abs_rel']:.4f}   δ₁: {results['delta1']:.4f}")

**Compare multiple models:**

.. code-block:: python

    from depth_estimation.evaluation import compare

    compare(
        ["depth-anything-v2-vits", "depth-anything-v2-vitb"],
        dataset="nyu_depth_v2",
    )

**Compute metrics for custom predictions:**

.. code-block:: python

    from depth_estimation.evaluation import DepthMetrics

    metrics = DepthMetrics()
    result = metrics(pred_tensor, gt_tensor, valid_mask_tensor)

**Profile latency:**

.. code-block:: python

    from depth_estimation.evaluation import profile_latency

    p = profile_latency("depth-anything-v2-vitb", num_runs=100)
    print(f"{p['mean_ms']:.1f} ms  |  {p['fps']:.1f} FPS")
"""

from .metrics import DepthMetrics, Evaluator, align_least_squares
from .evaluate import evaluate, compare
from .profiling import profile_latency

__all__ = [
    "evaluate",
    "compare",
    "DepthMetrics",
    "Evaluator",
    "align_least_squares",
    "profile_latency",
]
