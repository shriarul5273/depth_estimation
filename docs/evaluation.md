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

All formulas follow Eigen et al. (2014) and are used in virtually every depth estimation paper.

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

### Depth alignment for relative models

Relative models (Depth Anything, MiDaS, Pixel-Perfect Depth) output depth up to an unknown global scale and shift. Before computing metrics against metric ground truth, each sample's prediction is aligned using least-squares:

```
d_aligned = scale * d_pred + shift
```

where `scale` and `shift` minimise `‖scale * d_pred + shift − d_gt‖²` over valid pixels. This is the standard approach used by every paper that evaluates relative models on metric benchmarks (NYU, KITTI, DIODE).

Alignment is applied automatically when `config.is_metric = False`. Disable it with `align=False`.

### Examples

```python
from depth_estimation.evaluation import evaluate

# Full NYU test set
results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2")

# Quick 50-sample check on DIODE indoors
results = evaluate(
    "depth-pro",
    "diode",
    split="val",
    scene_type="indoors",
    num_samples=50,
)

# KITTI (path required)
results = evaluate(
    "zoedepth",
    "kitti_eigen",
    dataset_root="/data/kitti",
    align=False,   # metric model — no alignment needed
)

# Pass an already-loaded model instance to avoid re-loading between calls
from depth_estimation import AutoDepthModel
model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")
results = evaluate(model, "nyu_depth_v2")
```

---

## `compare()`

```python
compare(
    models: list[str],
    dataset: str | BaseDepthDataset,
    split: str = "test",
    dataset_root: str = None,
    batch_size: int = 1,
    device: str = None,
    num_workers: int = 4,
    align: bool = True,
    num_samples: int = None,
    print_table: bool = True,
    **dataset_kwargs,
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

Computing RMSE as the mean of per-batch RMSEs is biased when batches have different numbers of valid pixels (common for KITTI's sparse LiDAR GT). `Evaluator` instead accumulates the sum of squared errors over all valid pixels, then divides and takes the square root at the end.

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

| Argument | Type | Description |
|---|---|---|
| `pred` | `np.ndarray (H, W)` | Relative depth prediction. |
| `target` | `np.ndarray (H, W)` | Metric ground truth in metres. |
| `mask` | `np.ndarray (H, W)` bool | Valid pixels. |

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

| Argument | Default | Description |
|---|---|---|
| `model` | **required** | Model variant ID or instance. |
| `input_size` | `518` | Square input spatial dimension. |
| `batch_size` | `1` | Batch size for each forward pass. |
| `num_warmup` | `10` | Warmup passes (not timed). |
| `num_runs` | `100` | Timed passes. |
| `device` | `None` | Auto-detected. |
| `half` | `False` | FP16 precision. |

Returns a dict:

| Key | Description |
|---|---|
| `mean_ms` | Mean latency per batch (ms). |
| `std_ms` | Standard deviation (ms). |
| `min_ms` | Minimum observed (ms). |
| `max_ms` | Maximum observed (ms). |
| `p50_ms` | 50th-percentile latency (ms). |
| `p95_ms` | 95th-percentile latency (ms). |
| `p99_ms` | 99th-percentile latency (ms). |
| `fps` | Throughput: `batch_size / (mean_ms / 1000)`. |
| `memory_mb` | Peak GPU memory allocated (MiB). `None` on CPU/MPS. |
| `device` | Device string used. |
| `input_shape` | `(B, 3, H, W)` tuple. |

GPU timings use `torch.cuda.synchronize()` before each clock reading; CPU timings use `time.perf_counter()`.

---

## CLI: `depth-estimate evaluate`

The evaluation suite is also available directly from the command line — no Python script required.

```bash
# Single model on NYU Depth V2 (auto-downloads ~2.8 GB)
depth-estimate evaluate --model depth-anything-v2-vitb --dataset nyu_depth_v2

# Quick 50-sample check
depth-estimate evaluate --model depth-pro --dataset nyu --num-samples 50

# KITTI Eigen (manual download required)
depth-estimate evaluate --model zoedepth --dataset kitti --dataset-root /data/kitti --no-align

# DIODE indoors subset
depth-estimate evaluate --model depth-anything-v2-vitl --dataset diode --scene-type indoors

# Compare preset models and save results
depth-estimate evaluate --compare --dataset nyu_depth_v2 --output results/compare.json

# Machine-readable output
depth-estimate evaluate --model depth-pro --dataset nyu --json
```

See [docs/cli.md](cli.md) for the full flag reference.

---

## Evaluation Scripts

Ready-to-run Python scripts are in `examples/`. All scripts share the same interface.

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | varies | Model variant ID. |
| `--compare` | off | Evaluate a preset list of models and print a table. |
| `--num-samples N` | all | Limit to N samples for quick checks. |
| `--batch-size B` | `1` | Forward pass batch size. |
| `--num-workers W` | `4` | DataLoader workers. |
| `--device DEVICE` | auto | `cuda`, `cpu`, or `mps`. |
| `--no-align` | off | Disable alignment for relative models. |
| `--output FILE` | none | Save results to a JSON file. |

### `eval_nyu.py` — NYU Depth V2 (654 test images)

```bash
# Auto-downloads dataset (~2.8 GB) on first run
python examples/eval_nyu.py --model depth-anything-v2-vitb
python examples/eval_nyu.py --compare
python examples/eval_nyu.py --model depth-anything-v2-vitb --num-samples 50
python examples/eval_nyu.py --model depth-pro --output results/depth_pro.json
```

Requires `h5py`: `pip install "depth-estimation[data]"`

### `eval_kitti.py` — KITTI Eigen (697 test images)

```bash
# --dataset-root is required (see docs/data.md for download instructions)
python examples/eval_kitti.py --dataset-root /data/kitti
python examples/eval_kitti.py --model zoedepth --dataset-root /data/kitti
python examples/eval_kitti.py --compare --dataset-root /data/kitti
python examples/eval_kitti.py --dataset-root /data/kitti --split val --num-samples 100
```

### `eval_diode.py` — DIODE val set (771 images)

```bash
# Auto-downloads val set (~2.6 GB) on first run
python examples/eval_diode.py
python examples/eval_diode.py --scene-type indoors --max-depth 10.0
python examples/eval_diode.py --scene-type outdoors
python examples/eval_diode.py --compare
```
