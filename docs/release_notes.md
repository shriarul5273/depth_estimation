# v0.0.9 — Training & Fine-Tuning

# 🎉 depth_estimation v0.0.9

Adds full training and fine-tuning support for depth estimation models.

## ✨ New: Training API

Fine-tune any model on your own data with three lines:

```python
from depth_estimation import (
    DepthAnythingV2Model, DepthTrainer, DepthTrainingArguments, load_dataset
)
from depth_estimation.data.transforms import get_train_transforms, get_val_transforms

model    = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vits", for_training=True)
model.freeze_backbone()   # decoder-only fine-tuning

train_ds = load_dataset("nyu_depth_v2", split="train", transform=get_train_transforms(518))
val_ds   = load_dataset("nyu_depth_v2", split="test",  transform=get_val_transforms(518))

args = DepthTrainingArguments(output_dir="./checkpoints", num_epochs=25, batch_size=8)
DepthTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds).train()
```

See [docs/training.md](training.md) for the full API reference.

## ✨ New: `BaseDepthModel` training helpers

All models gain training utilities:

| Method | Description |
|---|---|
| `from_pretrained(..., for_training=True)` | Load in train mode (no `.eval()`) |
| `freeze_backbone()` | Freeze backbone, train decoder only |
| `unfreeze_backbone()` | Unfreeze backbone for full fine-tuning |
| `get_parameter_groups(backbone_lr_scale)` | Differential LR groups for AdamW |
| `unfreeze_top_k_backbone_layers(k)` | Gradual unfreezing (DINOv2-based models) |
| `_count_trainable()` | Count trainable parameters |

## ✨ New: Loss Functions (`depth_estimation.losses`)

| Class | Description |
|---|---|
| `ScaleInvariantLoss(lam=0.85)` | SI-log loss — primary loss for relative depth |
| `GradientLoss()` | L1 on log-depth finite differences — preserves edges |
| `BerHuLoss()` | Reverse Huber — robust L1/L2 for metric depth |
| `CombinedDepthLoss(si_weight=1.0, grad_weight=0.5)` | Default weighted combination |

## ✨ New: Paired Data Transforms (`depth_estimation.data.transforms`)

All transforms apply identically to `(pixel_values, depth_map, valid_mask)` tuples:

| Transform | Description |
|---|---|
| `PairedResize(size)` | Resize shorter side to `size` |
| `PairedRandomScale(scale_range)` | Random scale; multiplies depth values |
| `PairedRandomCrop(size)` | Random spatial crop |
| `PairedCenterCrop(size)` | Deterministic center crop |
| `PairedRandomHorizontalFlip(p)` | Random horizontal flip |
| `PairedColorJitter(...)` | Color jitter on image only |
| `PairedNormalize(mean, std)` | ImageNet normalization on image only |
| `Compose(transforms)` | Chain transforms |
| `get_train_transforms(input_size=518)` | Standard training preset |
| `get_val_transforms(input_size=518)` | Standard validation preset |

## ✨ New: `DepthTrainingArguments`

Dataclass with all hyperparameters, serialisable to/from JSON:

```python
args = DepthTrainingArguments(
    output_dir="./checkpoints",
    num_epochs=25,
    learning_rate=5e-5,
    backbone_lr_scale=0.1,
    lr_scheduler="cosine",        # cosine | linear | step | plateau
    freeze_backbone_epochs=5,     # warm up decoder for 5 epochs first
    mixed_precision=True,
    eval_metric="abs_rel",
)
args.to_json("./checkpoints/args.json")
```

## ✨ New: `DepthTrainer`

Full training loop with no external framework dependency:

- AdamW optimiser with differential learning rates (backbone vs. decoder)
- Linear warmup + cosine/linear/step/plateau LR scheduling
- Automatic mixed precision (AMP) support
- Gradient clipping
- Backbone freeze/unfreeze schedule
- Checkpoint save and resume
- Evaluation using `Evaluator` (unbiased dataset-level RMSE)
- Best-model checkpoint tracking

## ✨ New: Training Example Scripts

| Script | Purpose |
|---|---|
| `examples/train_depth_anything.py` | Fine-tune Depth Anything v2-Small on NYU Depth V2 |
| `examples/test_finetuned.py` | Evaluate a fine-tuned checkpoint vs. the base model |

## Trainable models in this release

| Model | `_backbone_module()` | Notes |
|---|---|---|
| `depth-anything-v1-*` | DINOv2 via `self.net.pretrained` | Full fine-tuning supported |
| `depth-anything-v2-*` | DINOv2 via `self.net.pretrained` | Full fine-tuning supported |
| `depth-anything-v3-*` | DinoV2 via `self.net.backbone` | Full fine-tuning supported |
| `midas-*` | HF DPT encoder via `self.dpt.dpt.encoder` | Supported; custom `unfreeze_top_k` |
| `depth-pro` | DepthProEncoder via `self._net.encoder` | Supported; custom `unfreeze_top_k` |
| `vggt` | Aggregator via `self._net.aggregator` | Supported; custom `unfreeze_top_k` for frame/global blocks |
| `omnivggt` | ZeroAggregator via `self._net.aggregator` | Supported; custom `unfreeze_top_k` for frame/global blocks |
| `moge-*` | — | Raises `NotImplementedError` (`forward()` uses non-differentiable focal/shift recovery) |
| `pixel-perfect-depth` | — | Raises `NotImplementedError` (`forward()` runs iterative diffusion sampling) |
| `zoedepth` | — | Raises `NotImplementedError` (wraps HF pipeline) |
| `marigold-dc` | — | Raises `NotImplementedError` (wraps diffusers pipeline) |
