# v0.0.8 — Datasets & Evaluation

# 🎉 depth_estimation v0.0.8

Adds dataset downloading, a full evaluation suite, and ready-to-run benchmark scripts.

## ✨ New: Dataset Loading

`load_dataset()` downloads and loads standard depth benchmarks with a single call.

| Dataset | Name | Auto-download | GT type | Split size |
|---|---|---|---|---|
| NYU Depth V2 | `nyu_depth_v2` | Yes (~2.8 GB) | Dense, metric | 654 test |
| DIODE | `diode` | Yes (~2.6 GB val) | Dense, metric | 771 val |
| KITTI Eigen | `kitti_eigen` | No (registration) | Sparse LiDAR | 697 test |
| Generic folder | `folder` | N/A | Any | N/A |

```python
from depth_estimation import load_dataset

ds = load_dataset("nyu_depth_v2", split="test")   # auto-downloads on first use
ds = load_dataset("diode", split="val", scene_type="indoors")
ds = load_dataset("kitti_eigen", split="test", root="/data/kitti")
ds = load_dataset("folder", image_dir="rgb/", depth_dir="depth/")
```

All datasets return `{"pixel_values", "depth_map", "valid_mask"}` tensors compatible with `DataLoader`. See [docs/data.md](data.md).

Install the `[data]` extra for NYU Depth V2 support: `pip install "depth-estimation[data]"`

## ✨ New: Evaluation Suite

Standard 7-metric evaluation with automatic alignment for relative models.

```python
from depth_estimation.evaluation import evaluate, compare, DepthMetrics, profile_latency

# Full dataset evaluation
results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2")

# Side-by-side comparison table
compare(["depth-anything-v2-vits", "depth-anything-v2-vitb"], dataset="nyu_depth_v2")

# Custom prediction metrics
metrics = DepthMetrics()
result  = metrics(pred_tensor, gt_tensor, valid_mask)

# Latency + GPU memory profiling
p = profile_latency("depth-anything-v2-vitb", num_runs=100)
```

See [docs/evaluation.md](evaluation.md).

## ✨ New: Benchmark Scripts

Three ready-to-run evaluation scripts in `examples/`:

| Script | Dataset | Key feature |
|---|---|---|
| `eval_nyu.py` | NYU Depth V2 | Auto-downloads dataset |
| `eval_kitti.py` | KITTI Eigen | Clear download instructions |
| `eval_diode.py` | DIODE | `--scene-type indoors/outdoors/all` |

All scripts support `--model`, `--compare`, `--num-samples`, `--output`, `--device`.

## ✨ New: CLI Evaluation

Evaluate any model from the command line without writing Python:

```bash
depth-estimate evaluate --model depth-anything-v2-vitb --dataset nyu_depth_v2
depth-estimate evaluate --compare --dataset nyu_depth_v2
depth-estimate evaluate --model depth-pro --dataset diode --scene-type indoors --num-samples 50
```

## ✨ New: `examples/` Folder

All example and benchmark scripts have moved to `examples/` alongside `demo.png`.
