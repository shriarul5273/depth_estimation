# Evaluation

The `depth_estimation.evaluation` module provides standard depth metrics, a dataset-level accumulator, top-level `evaluate()` / `compare()` functions, and a latency profiler.

## Quick Start

```python
from depth_estimation.evaluation import evaluate

results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2", split="test")
print(results["abs_rel"], results["delta1"])
```

---

## Metrics

Seven standard metrics are computed over valid pixels (pixels where ground-truth depth is within `[min_depth, max_depth]`).

| Metric | Formula | Range | Direction |
|---|---|---|---|
| `abs_rel` | `mean(|d − d̂| / d)` | [0, ∞) | lower ↓ |
| `sq_rel` | `mean((d − d̂)² / d)` | [0, ∞) | lower ↓ |
| `rmse` | `√mean((d − d̂)²)` | [0, ∞) | lower ↓ |
| `rmse_log` | `√mean((log d − log d̂)²)` | [0, ∞) | lower ↓ |
| `delta1` | `% pixels: max(d/d̂, d̂/d) < 1.25` | [0, 1] | higher ↑ |
| `delta2` | same, threshold 1.25² | [0, 1] | higher ↑ |
| `delta3` | same, threshold 1.25³ | [0, 1] | higher ↑ |

All formulas follow Eigen et al. \[1\] and are used in virtually every depth estimation paper. The threshold accuracy metrics (δ < 1.25, 1.25², 1.25³) were further standardised in Eigen & Fergus \[2\].

---

## `evaluate()`

```python
evaluate(
    model: str | BaseDepthModel,
    dataset: str | BaseDepthDataset,
    split: str = "test",
    dataset_root: str = None,
    batch_size: int = 1,
    device: str = None,
    num_workers: int = 4,
    align: bool = True,
    num_samples: int = None,
    **dataset_kwargs,
) -> dict
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| BaseDepthModel` | **required** | Model variant ID or loaded model instance. |
| `dataset` | `str \| BaseDepthDataset` | **required** | Dataset name (`"nyu_depth_v2"`, `"diode"`, `"kitti_eigen"`) or a loaded dataset instance. Aliases `"nyu"` and `"kitti"` are accepted. |
| `split` | `str` | `"test"` | Dataset split. |
| `dataset_root` | `str` | `None` | Root directory (when *dataset* is a string). |
| `batch_size` | `int` | `1` | Images per forward pass. |
| `device` | `str` | `None` | Device. Auto-detected if `None`. |
| `num_workers` | `int` | `4` | DataLoader workers. Use `0` on Windows with h5py. |
| `align` | `bool` | `True` | Apply per-sample least-squares alignment for relative models. |
| `num_samples` | `int` | `None` | Limit the number of samples (for quick checks). |
| `**dataset_kwargs` | | | Forwarded to `load_dataset()` (e.g. `scene_type="indoors"` for DIODE). |

Returns a `dict` with keys `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `delta1`, `delta2`, `delta3`, `n_pixels`, `n_samples`.

Relative models are aligned per-sample via least-squares scale+shift before metrics are computed. Disable with `align=False`.

### Examples

```python
from depth_estimation.evaluation import evaluate

# Full NYU test set
results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2")

# KITTI (path required)
results = evaluate(
    "zoedepth",
    "kitti_eigen",
    dataset_root="/data/kitti",
    align=False,   # metric model — no alignment needed
)
```

---

## `compare()`

```python
compare(
    models: list[str],
    dataset: str | BaseDepthDataset,
    num_samples: int = None,
    ...same remaining args as evaluate()...
) -> dict[str, dict]
```

Evaluates each model in `models` on the same dataset and prints a formatted comparison table. Returns `{model_id: metrics_dict}`.

```python
from depth_estimation.evaluation import compare

results = compare(
    ["depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl"],
    dataset="nyu_depth_v2",
    num_samples=100,   # quick run
)
# results["depth-anything-v2-vitb"]["abs_rel"]  → float
```

Example table output:

```
-----------------------------------------------------------------
Model                        abs_rel   sq_rel     rmse  rmse_log   delta1   delta2   delta3
                                 (↓)      (↓)      (↓)       (↓)      (↑)      (↑)      (↑)
-----------------------------------------------------------------
depth-anything-v2-vits      0.0512*  0.0143*  0.3541*   0.0712*  0.9723*  0.9961*  0.9992*  n=654
depth-anything-v2-vitb      0.0431   0.0124   0.3121    0.0612   0.9824   0.9971   0.9993   n=654
depth-anything-v2-vitl      0.0378   0.0101   0.2874    0.0571   0.9891   0.9981   0.9996   n=654
-----------------------------------------------------------------
* = best in column
```

---

## `DepthMetrics`

Compute metrics for a single prediction:

```python
from depth_estimation.evaluation import DepthMetrics

metrics = DepthMetrics(eps=1e-6)
result  = metrics(pred, target, valid_mask)
# result: {"abs_rel": ..., "sq_rel": ..., "rmse": ..., "rmse_log": ...,
#           "delta1": ..., "delta2": ..., "delta3": ...}
```

| Argument | Type | Description |
|---|---|---|
| `pred` | `Tensor` `(H, W)` or `(B, H, W)` or `(B, 1, H, W)` | Predicted depth, positive values. |
| `target` | same shape | Ground-truth depth in metres. |
| `valid_mask` | same shape, `bool` | Validity mask. Defaults to `target > eps`. |

---

## `Evaluator`

Accumulates metrics across batches for **correct dataset-level RMSE**.

```python
from depth_estimation.evaluation import Evaluator

ev = Evaluator()

for batch in dataloader:
    pred   = model(batch["pixel_values"])
    target = batch["depth_map"]
    mask   = batch["valid_mask"]
    ev.update(pred, target, mask)

results = ev.compute()
# {"abs_rel": ..., "rmse": ..., "delta1": ..., ..., "n_pixels": 12345678}
# Note: RMSE is computed over all valid pixels globally, not as a mean of per-batch RMSEs.

ev.reset()  # clear state before the next evaluation run
```

---

## `align_least_squares`

Compute the scale and shift that maps a relative prediction onto metric ground truth:

```python
from depth_estimation.evaluation import align_least_squares
import numpy as np

scale, shift = align_least_squares(pred_np, target_np, mask_np)
aligned = (scale * pred_np + shift).clip(min=1e-6)
```

Returns `(scale, shift)` as Python floats.

---

## `profile_latency`

Measure per-batch inference latency and peak GPU memory:

```python
from depth_estimation.evaluation import profile_latency

p = profile_latency(
    "depth-anything-v2-vitb",
    input_size=518,
    batch_size=1,
    num_warmup=10,
    num_runs=100,
    device="cuda",
)
```

Returns `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `fps`, `memory_mb`, `device`, `input_shape`.

---

## Evaluation Scripts

Ready-to-run scripts: `examples/eval_nyu.py`, `examples/eval_kitti.py`, `examples/eval_diode.py` — all accept `--model`, `--compare`, `--num-samples`, `--output`.

---

## References

\[1\] **Eigen, D., Puhrsch, C., & Fergus, R.** (2014).
*Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.*
NeurIPS 2014.
[https://arxiv.org/abs/1406.2283](https://arxiv.org/abs/1406.2283)

\[2\] **Eigen, D., & Fergus, R.** (2015).
*Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture.*
ICCV 2015.
[https://arxiv.org/abs/1411.4734](https://arxiv.org/abs/1411.4734)

\[3\] **Silberman, N., Hoiem, D., Kohli, P., & Fergus, R.** (2012).
*Indoor Segmentation and Support Inference from RGBD Images.*
ECCV 2012.
[https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

\[4\] **Geiger, A., Lenz, P., & Urtasun, R.** (2012).
*Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.*
CVPR 2012.
[https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)

\[5\] **Vasiljevic, I., Kolkin, N., Zhang, S., et al.** (2019).
*DIODE: A Dense Indoor and Outdoor DEpth Dataset.*
arXiv 1908.00463.
[https://arxiv.org/abs/1908.00463](https://arxiv.org/abs/1908.00463)

\[6\] **Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V.** (2022).
*Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer.*
IEEE TPAMI.
[https://arxiv.org/abs/1907.01341](https://arxiv.org/abs/1907.01341)
