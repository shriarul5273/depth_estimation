# depth_estimation

<p align="center">
    <a href="https://github.com/shriarul5273/depth_estimation/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/shriarul5273/depth_estimation?color=blue"></a>
    <a href="https://pypi.org/project/depth-estimation/"><img alt="PyPI" src="https://img.shields.io/pypi/v/depth-estimation"></a>
    <a href="https://pypi.org/project/depth-estimation/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/depth-estimation"></a>
    <a href="https://huggingface.co/spaces/shriarul5273/Depth-Estimation-Compare-demo"><img alt="Demo" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue"></a>
</p>

<h3 align="center">A unified Python library for monocular depth estimation</h3>

<h3 align="center">Inference · Fine-Tuning · Evaluation · Dataset Loading</h3>

---

`depth_estimation` is the model-definition framework for depth estimation. It provides a single, consistent API across **12 model families and 28 variants** — so you can swap models, compare them, and fine-tune them without rewriting your pipeline.

It covers the full workflow end-to-end: run inference with one line, evaluate on standard benchmarks, and fine-tune on custom depth data — all with the same library.

## Installation

```bash
pip install depth-estimation
```

See [docs/dependencies.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/dependencies.md) for optional extras (CUDA, MPS, etc.).

---

## Quickstart

The `pipeline` API is the fastest way to get a depth map from any image:

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
result = pipe("image.jpg")

depth_map = result.depth            # np.ndarray, float32, (H, W)
colored   = result.colored_depth    # np.ndarray, uint8,   (H, W, 3)
```

For full control over each step — preprocessing, forward pass, postprocessing — use Auto Classes:

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

Or from the command line:

```bash
depth-estimate predict image.jpg --model depth-anything-v2-vitb
```

---

## Why use depth_estimation?

**1. One API, every model.**
Switch from Depth Anything to DepthPro to MoGe by changing a single string. Preprocessing, postprocessing, and output format are identical across all models.

**2. The full depth workflow in one place.**
Most libraries stop at inference. This one covers training, evaluation on standard benchmarks, and dataset loading — so you don't have to stitch together separate tools.

**3. Modular, single-file model design.**
Each model lives in one self-contained file. No hidden abstractions. If you need to understand or modify a model, there's exactly one place to look. New models self-register — `AutoDepthModel` and `pipeline()` resolve them automatically.

**4. Designed for research.**
Trainable models with backbone freeze schedules, proper batch-level metric accumulation (no mean-of-means), and a `compare()` function that shows a formatted table across models.

---

## Supported Models

12 model families · 28 variants — see [docs/models.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/models.md) for the full list.

All models support inference and CLI. The Trainable column indicates fine-tuning support via `DepthTrainer`.

| Family | Variants | Depth type | Trainable |
|---|---|---|:---:|
| Depth Anything v1 | vits / vitb / vitl | Relative | ✅ |
| Depth Anything v2 | vits / vitb / vitl | Relative | ✅ |
| Depth Anything v3 | small / base / large / giant / mono / metric | Relative + Metric | ✅ |
| Depth Anything v3 Nested | nested-giant-large | Relative | ✅ |
| ZoeDepth | nyu / kitti | Metric | ❌ |
| MiDaS | dpt-large / dpt-hybrid / beit-large | Relative | ✅ |
| Apple DepthPro | — | Metric | ✅ |
| Pixel-Perfect Depth | — | Relative | ❌ |
| Marigold-DC | — | Relative (depth completion) | ❌ |
| MoGe | v1 vitl / v2 vitl / v2 vitb / v2 vits (+ normal variants) | Metric | ❌ |
| OmniVGGT | vitl | Metric | ✅ |
| VGGT | standard / commercial | Metric | ✅ |

---

## What can you do?

<details>
<summary><b>Inference</b> — single image, batch, or video</summary>

```python
# Single image
result = pipe("image.jpg")

# Batch
results = pipe(["img1.jpg", "img2.jpg"], batch_size=2)
```

```bash
# CLI — batch predict
depth-estimate predict "images/*.jpg" --model depth-anything-v2-vitb --output-dir results/
```

</details>

<details>
<summary><b>Evaluation</b> — standard benchmarks, custom predictions</summary>

```python
from depth_estimation.evaluation import evaluate, compare, Evaluator

# Single model on NYU Depth V2
results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2", split="test")

# Compare multiple models — prints table with best values marked (*)
compare(["depth-anything-v2-vits", "depth-anything-v2-vitb"], dataset="nyu_depth_v2")

# Accumulate metrics over your own dataloader
ev = Evaluator()
for pred, gt, mask in dataloader:
    ev.update(pred, gt, mask)
final = ev.compute()    # abs_rel, sq_rel, rmse, rmse_log, delta1/2/3
```

See [docs/evaluation.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/evaluation.md).

</details>

<details>
<summary><b>Fine-Tuning</b> — any trainable model, any depth dataset</summary>

```python
from depth_estimation import DepthAnythingV2Model, DepthTrainer, DepthTrainingArguments, load_dataset
from depth_estimation.data.transforms import get_train_transforms, get_val_transforms

model    = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vits", for_training=True)
train_ds = load_dataset("nyu_depth_v2", split="train", transform=get_train_transforms(518))
val_ds   = load_dataset("nyu_depth_v2", split="test",  transform=get_val_transforms(518))

args = DepthTrainingArguments(output_dir="./checkpoints", num_epochs=25, batch_size=8,
                               freeze_backbone_epochs=5, mixed_precision=True)
DepthTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds).train()
```

Any `torch.utils.data.Dataset` returning `pixel_values / depth_map / valid_mask` works directly — no subclassing needed. See [docs/training.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/training.md).

</details>

<details>
<summary><b>Dataset Loading</b> — standard benchmarks, custom folders</summary>

```python
from depth_estimation import load_dataset

ds = load_dataset("nyu_depth_v2",  split="test")                                    # auto-downloads ~2.8 GB
ds = load_dataset("diode",         split="val", scene_type="indoors")               # auto-downloads ~2.6 GB
ds = load_dataset("kitti_eigen",   split="test", root="/data/kitti")               # local path
ds = load_dataset("folder",        image_dir="rgb/", depth_dir="depth/")           # any folder
```

See [docs/data.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/data.md).

</details>

---

## Adding a New Model

1. Create `src/depth_estimation/models/your_model/`
2. Add `configuration_your_model.py` (inherit `BaseDepthConfig`)
3. Add `modeling_your_model.py` (inherit `BaseDepthModel`, single file)
4. Add `__init__.py` with `MODEL_REGISTRY.register(...)`

`AutoDepthModel`, `AutoProcessor`, and `pipeline()` resolve the new model automatically. See [docs/adding_a_model.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/adding_a_model.md) for a step-by-step guide.

---

## Acknowledgments

This library builds upon the work of 12 research teams — see [docs/models.md#citations](https://github.com/shriarul5273/depth_estimation/blob/main/docs/models.md#citations) for the full list.

## License

MIT
