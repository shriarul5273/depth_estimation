# depth_estimation

A **Python library** for monocular depth estimation.

Provides a unified, modular API for inference, evaluation, and dataset loading — supporting **12 model families** with **28 variants** and designed to accommodate new models with minimal friction.

## Installation

```bash
pip install depth-estimation
```

For dataset downloading (NYU Depth V2) install the `data` extra:

```bash
pip install "depth-estimation[data]"   # adds h5py, tqdm
```

For a full list of dependencies see [docs/dependencies.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/dependencies.md).

---

## Quick Start

| | Pipeline API | Auto Classes |
|---|---|---|
| **Setup** | One call, model + processor bundled | Load model and processor separately |
| **Inference** | Pass image path directly | Call `processor()`, `model()`, `postprocess()` manually |
| **Control** | Low — handles everything for you | High — you control each step |
| **Output** | `DepthOutput` with `.depth`, `.colored_depth`, `.metadata` | Raw depth tensor |
| **Best for** | Quick inference, scripts, demos | Custom pipelines, research, fine-grained control |

### Pipeline API (Recommended)

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
result = pipe("image.jpg")

depth_map = result.depth            # np.ndarray, float32, (H, W)
colored   = result.colored_depth    # np.ndarray, uint8, (H, W, 3)
meta      = result.metadata         # dict with model info
```

### Auto Classes

```python
from depth_estimation import AutoDepthModel, AutoProcessor
import torch

model     = AutoDepthModel.from_pretrained("zoedepth")
processor = AutoProcessor.from_pretrained("zoedepth")

inputs = processor("image.jpg")
with torch.no_grad():
    depth = model(inputs["pixel_values"])

result = processor.postprocess(depth, inputs["original_sizes"])
```

### Batch Inference

```python
results = pipe(["img1.jpg", "img2.jpg", "img3.jpg"], batch_size=2)
for r in results:
    print(r.depth.shape)
```

---

## Supported Models

12 model families · 28 variants — see [docs/models.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/models.md) for the full list.

| Family | Variants | Depth type |
|---|---|---|
| Depth Anything v1 | vits / vitb / vitl | Relative |
| Depth Anything v2 | vits / vitb / vitl | Relative |
| Depth Anything v3 | small / base / large / giant + nested + metric + mono | Relative + Metric |
| ZoeDepth | nyu-kitti | Metric |
| MiDaS | dpt-large / dpt-hybrid / beit-large | Relative |
| Apple DepthPro | — | Metric |
| Pixel-Perfect Depth | — | Relative |
| Marigold-DC | — | Relative (depth completion) |
| MoGe | v1 vitl / v2 vitl / v2 vitb / v2 vits (+ normal variants) | Metric |
| OmniVGGT | vitl | Metric |
| VGGT | standard / commercial | Metric |

---

## Datasets

`load_dataset()` downloads and loads standard depth benchmarks with a single call.

```python
from depth_estimation import load_dataset

# NYU Depth V2 — auto-downloads ~2.8 GB on first use
ds = load_dataset("nyu_depth_v2", split="test")

# DIODE val set — auto-downloads ~2.6 GB on first use
ds = load_dataset("diode", split="val", scene_type="indoors")

# KITTI Eigen — path required (see docs/data.md for download instructions)
ds = load_dataset("kitti_eigen", split="test", root="/data/kitti")

# Generic RGB + depth folder
ds = load_dataset("folder", image_dir="rgb/", depth_dir="depth/")
```

Every dataset returns the same schema, compatible with `torch.utils.data.DataLoader`:

```python
sample = ds[0]
sample["pixel_values"]  # (3, H, W) float32, normalised [0, 1]
sample["depth_map"]     # (1, H, W) float32, metres
sample["valid_mask"]    # (1, H, W) bool
```

| Dataset | Auto-download | GT type | Test size |
|---|---|---|---|
| `nyu_depth_v2` | Yes (~2.8 GB) | Dense, metric | 654 images |
| `diode` | Yes (~2.6 GB val) | Dense, metric | 771 images |
| `kitti_eigen` | No (registration required) | Sparse LiDAR | 697 images |
| `folder` | N/A | Any | N/A |

See [docs/data.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/data.md) for full documentation.

---

## Evaluation

Evaluate any model on any supported dataset with a single call. Relative-depth models are aligned per-sample (least-squares scale + shift) before metric computation — detected automatically from `config.is_metric`.

### Evaluate one model

```python
from depth_estimation.evaluation import evaluate

results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2", split="test")
# {"abs_rel": 0.043, "sq_rel": 0.012, "rmse": 0.312,
#  "rmse_log": 0.061, "delta1": 0.982, "delta2": 0.997,
#  "delta3": 0.999, "n_samples": 654}
```

### Compare multiple models

```python
from depth_estimation.evaluation import compare

compare(
    ["depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl"],
    dataset="nyu_depth_v2",
)
```

Prints a formatted table with best values marked (`*`).

### Compute metrics on custom predictions

```python
from depth_estimation.evaluation import DepthMetrics, Evaluator

# Per-prediction
metrics = DepthMetrics()
result  = metrics(pred_tensor, gt_tensor, valid_mask)

# Accumulate correctly across batches (proper RMSE, not mean-of-means)
ev = Evaluator()
for pred, gt, mask in dataloader:
    ev.update(pred, gt, mask)
final = ev.compute()
```

### Profile latency

```python
from depth_estimation.evaluation import profile_latency

p = profile_latency("depth-anything-v2-vitb", num_runs=100)
print(f"{p['mean_ms']:.1f} ms  |  {p['fps']:.1f} FPS  |  {p['memory_mb']:.0f} MiB")
```

Metrics: `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `delta1` / `delta2` / `delta3`.

See [docs/evaluation.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/evaluation.md) for full documentation.

### Evaluation scripts

Ready-to-run scripts are in `examples/`:

```bash
# NYU Depth V2 (auto-downloads dataset)
python examples/eval_nyu.py --model depth-anything-v2-vitb
python examples/eval_nyu.py --compare                        # all models, comparison table

# KITTI Eigen (manual download required)
python examples/eval_kitti.py --model zoedepth --dataset-root /data/kitti

# DIODE (auto-downloads ~2.6 GB val set)
python examples/eval_diode.py --scene-type indoors

# Quick 50-sample sanity check on any script
python examples/eval_nyu.py --model depth-anything-v2-vits --num-samples 50

# Save results to JSON
python examples/eval_nyu.py --model depth-pro --output results/depth_pro_nyu.json
```

---

## CLI

After installing the package, a `depth-estimate` command is available.

```bash
# Single image → saves demo_depth.png
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb

# Batch (directory or glob) → saves to results/
depth-estimate predict "images/*.jpg" --model depth-anything-v2-vitb --output-dir results/

# Video → saves side-by-side RGB | depth as MP4
depth-estimate predict video.mp4 --model depth-anything-v2-vitb --output depth_video.mp4

# Save raw float32 array (.npy) alongside the PNG
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb --format both

# Change colormap
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb --colormap inferno

# List all available models
depth-estimate list-models

# Show config details for a model
depth-estimate info depth-anything-v2-vitb

# Evaluate a model on NYU Depth V2 (auto-downloads ~2.8 GB)
depth-estimate evaluate --model depth-anything-v2-vitb --dataset nyu_depth_v2

# Quick 50-sample check
depth-estimate evaluate --model depth-pro --dataset nyu --num-samples 50

# Compare multiple models and save results
depth-estimate evaluate --compare --dataset nyu_depth_v2 --output results.json
```

**Global flags** (`--device`, `--quiet`, `--verbose`) go before the subcommand:

```bash
depth-estimate --device cpu --quiet predict examples/demo.png --model depth-anything-v2-vitb
```

All subcommands support `--json` for machine-readable output. See [docs/cli.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/cli.md) for full documentation.

---

## Architecture

The library follows the **HuggingFace Transformers** modular design philosophy:

- **Single model, single file** — each model's architecture is self-contained
- **Shared processor** — preprocessing/postprocessing is not duplicated
- **Registry-based auto-loading** — new models self-register, no core changes needed
- **Config inheritance** — configs override only what differs from the base

```
Input → Processor.preprocess() → Model.forward() → Processor.postprocess() → DepthOutput
```

## Adding a New Model

1. Create `src/depth_estimation/models/your_model/`
2. Add `configuration_your_model.py` (inherit `BaseDepthConfig`)
3. Add `modeling_your_model.py` (inherit `BaseDepthModel`, single file)
4. Add `__init__.py` with `MODEL_REGISTRY.register(...)`

That's it — `AutoDepthModel`, `AutoProcessor`, and `pipeline()` will automatically resolve your model. See [docs/adding_a_model.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/adding_a_model.md) for a step-by-step guide.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Acknowledgments

This library builds upon the incredible work of the following research teams:

| Model | Repository |
|---|---|
| **Depth Anything v1** | [github.com/LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything) |
| **Depth Anything v2** | [github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| **Depth Anything v3** | [github.com/DepthAnything/Depth-Anything-V3](https://github.com/DepthAnything/Depth-Anything-V3) |
| **DINOv2** | [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **DepthPro** | [github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) |
| **ZoeDepth** | [github.com/isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth) |
| **MiDaS** | [github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS) |
| **Pixel-Perfect Depth** | [github.com/gangweix/Pixel-Perfect-Depth](https://github.com/gangweix/Pixel-Perfect-Depth) |
| **Marigold-DC** | [github.com/prs-eth/Marigold-DC](https://github.com/prs-eth/Marigold-DC) |
| **MoGe** | [github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) |
| **VGGT** | [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| **OmniVGGT** | [github.com/Livioni/OmniVGGT](https://github.com/Livioni/OmniVGGT) |

## License

MIT
