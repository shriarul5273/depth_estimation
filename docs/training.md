# Training & Fine-Tuning

Fine-tune any supported model on your own depth data with a few lines of Python.

---

## Quick Start

```python
from depth_estimation import (
    DepthAnythingV2Model,
    DepthTrainer,
    DepthTrainingArguments,
    load_dataset,
)
from depth_estimation.data.transforms import get_train_transforms, get_val_transforms

# Load model in training mode
model = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vits", for_training=True)
model.freeze_backbone()          # decoder-only fine-tuning to start

# Load datasets with paired augmentation transforms
train_ds = load_dataset("nyu_depth_v2", split="train", transform=get_train_transforms(518))
val_ds   = load_dataset("nyu_depth_v2", split="test",  transform=get_val_transforms(518))

# Configure training
args = DepthTrainingArguments(
    output_dir="./checkpoints",
    num_epochs=25,
    batch_size=8,
    freeze_backbone_epochs=5,    # unfreeze backbone after epoch 5
    mixed_precision=True,
)

# Train
trainer = DepthTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

---

## `BaseDepthModel` Training Helpers

All depth models expose these methods when loaded with `for_training=True`.

```python
model = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vitb", for_training=True)
model.freeze_backbone()                        # freeze backbone; only decoder updates
model.unfreeze_backbone()                      # unfreeze for full fine-tuning
model.unfreeze_top_k_backbone_layers(4)        # unfreeze last 4 transformer blocks only
groups = model.get_parameter_groups(0.1)       # two AdamW groups: decoder (1×), backbone (0.1×)
print(model._count_trainable())                # number of trainable parameters
```

| Method | Description |
|---|---|
| `freeze_backbone()` | Freeze all backbone parameters |
| `unfreeze_backbone()` | Unfreeze for full fine-tuning |
| `unfreeze_top_k_backbone_layers(k)` | Unfreeze last k transformer blocks |
| `get_parameter_groups(backbone_lr_scale)` | Return two AdamW parameter groups |
| `_count_trainable()` | Count trainable parameters |

> **Note:** For MiDaS and DepthPro models, `unfreeze_top_k_backbone_layers()` uses the model's native layer names instead of DINOv2 block names.

---

## `DepthTrainingArguments`

Dataclass holding all hyperparameters. Serialisable to/from JSON.

```python
args.to_json("./checkpoints/args.json")
args = DepthTrainingArguments.from_json("./checkpoints/args.json")
```

### Full argument reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | **required** | Directory for checkpoints and logs. |
| `num_epochs` | `int` | `25` | Total training epochs. |
| `learning_rate` | `float` | `5e-5` | Base LR for the decoder. |
| `backbone_lr_scale` | `float` | `0.1` | Backbone LR multiplier (relative to `learning_rate`). |
| `weight_decay` | `float` | `0.01` | L2 regularisation for AdamW. |
| `batch_size` | `int` | `8` | Samples per GPU per step. |
| `gradient_clip_val` | `float` | `1.0` | Max gradient norm. Set `0.0` to disable. |
| `freeze_backbone_epochs` | `int` | `0` | Epochs to keep backbone frozen at start. `0` = backbone trainable from epoch 0 (full fine-tuning, no warm-up). |
| `lr_scheduler` | `str` | `"cosine"` | `"cosine"` \| `"linear"` \| `"step"` \| `"plateau"` |
| `warmup_epochs` | `int` | `0` | Epochs of linear LR warmup at the start of training. `0` = no warmup. |
| `si_loss_weight` | `float` | `1.0` | Weight for `ScaleInvariantLoss`. |
| `grad_loss_weight` | `float` | `0.5` | Weight for `GradientLoss`. |
| `si_lam` | `float` | `0.85` | Variance weight λ inside SI loss. |
| `save_every_n_epochs` | `int` | `5` | Checkpoint save interval. |
| `eval_every_n_epochs` | `int` | `1` | Validation interval. |
| `log_every_n_steps` | `int` | `50` | Logging interval. |
| `eval_metric` | `str` | `"abs_rel"` | Metric for best-model tracking. One of: `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `delta1`, `delta2`, `delta3`. |
| `lower_is_better` | `bool` | `True` | `True` for error metrics; `False` for accuracy (`delta*`). |
| `mixed_precision` | `bool` | `False` | Enable AMP (`torch.cuda.amp`). CUDA only. |
| `dataloader_num_workers` | `int` | `4` | DataLoader worker processes. |
| `seed` | `int` | `42` | Random seed. |

---

## `DepthTrainer`

Full training loop with no external framework dependency.

```python
from depth_estimation import DepthTrainer

trainer = DepthTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,     # optional; skip to train without evaluation
)
trainer.train()
```

### Constructor arguments

| Argument | Description |
|---|---|
| `model` | A `BaseDepthModel` instance loaded with `for_training=True`. |
| `args` | `DepthTrainingArguments` instance. |
| `train_dataset` | Dataset returning `{"pixel_values", "depth_map", "valid_mask"}`. |
| `eval_dataset` | Optional validation dataset. Pass `None` to skip evaluation. |
| `processor` | Optional processor, used only for logging. |

### `train(resume_from=None)`

Run the full training loop. Pass `resume_from` to continue from a saved checkpoint directory; `load_checkpoint()` is called automatically to restore model weights and training state.

```python
trainer.train()                              # fresh training run
trainer.train(resume_from="./checkpoints/checkpoint_epoch_0009")  # resume
```

---

## Loss Functions (`depth_estimation.losses`)

All loss functions accept `(pred, target, valid_mask=None)` tensors of shape `(B, H, W)`.

| Class | Description |
|---|---|
| `ScaleInvariantLoss(lam=0.85, eps=1e-8)` | SI-log loss: `mean(d²) − λ·mean(d)²` where `d = log(pred) − log(target)`. Scale-invariant. |
| `GradientLoss(eps=1e-8)` | L1 loss on x/y finite differences of log-depth. Preserves depth edges. |
| `BerHuLoss()` | Reverse Huber — L1 for small errors, L2 for large. Robust to metric depth outliers. |
| `CombinedDepthLoss(si_weight, grad_weight, lam)` | Weighted SI-log + gradient. Returns a dict. |

`CombinedDepthLoss` usage:

```python
loss_fn = CombinedDepthLoss(si_weight=1.0, grad_weight=0.5)
pred = model(pixel_values).squeeze(1)          # (B, H, W)
losses = loss_fn(pred, depth_map, valid_mask)
losses["loss"]      # total weighted loss (Tensor)
losses["si_loss"]   # scale-invariant component
losses["grad_loss"] # gradient component
```

---

## Paired Transforms (`depth_estimation.data.transforms`)

All transforms accept and return `(pixel_values, depth_map, valid_mask)` tuples where:
- `pixel_values` — `torch.Tensor` float32 `(3, H, W)`
- `depth_map` — `torch.Tensor` float32 `(1, H, W)`
- `valid_mask` — `torch.Tensor` bool `(1, H, W)`

Spatial transforms (resize, crop, flip, scale) apply identically to all three. Photometric transforms (color jitter, normalize) apply to `pixel_values` only.

### Presets

```python
from depth_estimation.data.transforms import get_train_transforms, get_val_transforms

train_t = get_train_transforms(input_size=518)
val_t   = get_val_transforms(input_size=518)
```

**Training pipeline:** `PairedResize → PairedRandomScale(1.0, 2.0) → PairedRandomCrop → PairedRandomHorizontalFlip → PairedColorJitter → PairedNormalize(ImageNet)`

**Validation pipeline:** `PairedResize → PairedCenterCrop → PairedNormalize(ImageNet)`

### All transforms

| Class | Description |
|---|---|
| `Compose(transforms)` | Chain multiple transforms |
| `PairedResize(size)` | Resize shorter side to `size` |
| `PairedRandomScale(scale_range, scale_depth)` | Random scale; multiplies depth values when `scale_depth=True` |
| `PairedRandomCrop(size)` | Random spatial crop |
| `PairedCenterCrop(size)` | Deterministic center crop |
| `PairedRandomHorizontalFlip(p)` | Random horizontal flip |
| `PairedColorJitter(brightness, contrast, saturation, hue)` | Color jitter on image only |
| `PairedNormalize(mean, std)` | Per-channel normalization on image only |

---

## Custom Datasets

Any `torch.utils.data.Dataset` that returns a dict with three keys works out of the box with `DepthTrainer`. No subclassing of a library class is required.

### Required output schema

| Key | Shape | dtype | Description |
|---|---|---|---|
| `pixel_values` | `(3, H, W)` | `float32` | RGB image, normalised to `[0, 1]` (or ImageNet-normalised after transforms) |
| `depth_map` | `(1, H, W)` | `float32` | Ground-truth depth in metres (or any consistent unit) |
| `valid_mask` | `(1, H, W)` | `bool` | `True` where depth is valid; `False` for sky / missing pixels |

### Loading from files

A typical dataset reads paired RGB and depth images from disk:

```python
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class FolderDepthDataset(Dataset):
    """Loads paired RGB / depth from two directories.

    Directory layout expected:
        rgb/   image_0001.jpg  image_0002.jpg  ...
        depth/ image_0001.png  image_0002.png  ...

    Depth PNGs are assumed to store millimetres as uint16
    (e.g. from a RealSense or similar sensor).
    """

    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_paths   = sorted(Path(rgb_dir).glob("*.jpg"))
        self.depth_paths = sorted(Path(depth_dir).glob("*.png"))
        assert len(self.rgb_paths) == len(self.depth_paths), (
            f"Mismatch: {len(self.rgb_paths)} RGB vs {len(self.depth_paths)} depth files"
        )
        self.transform = transform

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # RGB → float32 tensor (3, H, W) in [0, 1]
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")
        rgb = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Depth → float32 tensor (1, H, W) in metres
        depth_raw = np.array(Image.open(self.depth_paths[idx]), dtype=np.float32)
        depth = torch.from_numpy(depth_raw / 1000.0).unsqueeze(0)  # mm → m

        valid_mask = (depth > 0.1) & (depth < 10.0)  # sensor-specific range

        if self.transform is not None:
            rgb, depth, valid_mask = self.transform(rgb, depth, valid_mask)

        return {"pixel_values": rgb, "depth_map": depth, "valid_mask": valid_mask}
```

### Plugging into DepthTrainer

```python
train_ds = FolderDepthDataset("data/train/rgb", "data/train/depth",
                              transform=get_train_transforms(518))
val_ds   = FolderDepthDataset("data/val/rgb",   "data/val/depth",
                              transform=get_val_transforms(518))
trainer = DepthTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

### Tips

**Depth units** — the trainer and loss functions are unit-agnostic; just be consistent between `depth_map` and `valid_mask`. SI-log loss is scale-invariant, so absolute scale only matters for metric evaluation.

**No validation data** — pass `eval_dataset=None` to skip validation entirely. Checkpoints are still saved on the `save_every_n_epochs` schedule, and a `final/` checkpoint is always written.

**Large datasets** — set `dataloader_num_workers=4` (or higher) and ensure your `__getitem__` is stateless so it can be called from multiple worker processes safely.

**Variable-size images** — the built-in `PairedResize` + `PairedRandomCrop` / `PairedCenterCrop` transforms handle arbitrary input sizes and produce fixed-size tensors. Always apply at least a resize + crop before passing to the trainer so all samples in a batch share the same spatial dimensions.

---

## Trainable Models

| Model ID prefix | Backbone | `_backbone_module()` | Notes |
|---|---|---|---|
| `depth-anything-v1-*` | DINOv2 | `self.net.pretrained` | Full fine-tuning supported |
| `depth-anything-v2-*` | DINOv2 | `self.net.pretrained` | Full fine-tuning supported |
| `depth-anything-v3-*` | DINOv2 | `self.net.backbone` | Full fine-tuning supported |
| `midas-*` | HF DPT encoder | `self.dpt.dpt.encoder` | Supported; custom `unfreeze_top_k` using `encoder.layer` |
| `depth-pro` | DepthProEncoder | `self._net.encoder` | Supported; custom `unfreeze_top_k` for dual-ViT encoder |
| `vggt` | Aggregator (frame + global blocks) | `self._net.aggregator` | Supported; custom `unfreeze_top_k` for `frame_blocks`/`global_blocks` |
| `omnivggt` | ZeroAggregator (frame + global blocks) | `self._net.aggregator` | Supported; custom `unfreeze_top_k` for `frame_blocks`/`global_blocks` |
| `moge-*` | — | raises `NotImplementedError` | `forward()` uses non-differentiable focal/shift recovery |
| `pixel-perfect-depth` | — | raises `NotImplementedError` | `forward()` runs iterative diffusion sampling — not differentiable |
| `zoedepth` | — | raises `NotImplementedError` | Wraps HF pipeline; no `nn.Module` parameters |
| `marigold-dc` | — | raises `NotImplementedError` | Wraps diffusers pipeline; not directly trainable |

---

## Training Modes

| Mode | How to activate | What happens |
|---|---|---|
| **Full fine-tuning** | `freeze_backbone_epochs=0` (default) | Backbone trainable from epoch 0; all parameters update immediately |
| **Full fine-tuning with warm-up** | `freeze_backbone_epochs=N` (e.g. `5`) | Backbone frozen for first N epochs (decoder warms up), then unfrozen |
| **Decoder-only** | `model.freeze_backbone()` + `freeze_backbone_epochs=0` | Backbone stays frozen for the entire run |

`freeze_backbone_epochs=0` is the default — it never calls `freeze_backbone()`, so all parameters are trainable from step 1.

---

## Recipes

### Decoder-only fine-tuning (fastest)

Freeze the backbone and train only the DPT decoder. Good starting point for domain adaptation.

```python
model = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vitb", for_training=True)
model.freeze_backbone()
print(f"Trainable: {model._count_trainable():,}")   # ~5-10M params

args = DepthTrainingArguments(
    output_dir="./checkpoints",
    num_epochs=15,
    learning_rate=1e-4,
    batch_size=8,
)
```

### Full fine-tuning with warm-up period

Freeze backbone for first 5 epochs (warm up decoder), then unfreeze for full fine-tuning.

```python
args = DepthTrainingArguments(
    output_dir="./checkpoints",
    num_epochs=30,
    learning_rate=5e-5,
    backbone_lr_scale=0.1,    # backbone LR = 5e-6
    freeze_backbone_epochs=5,
    lr_scheduler="cosine",
    mixed_precision=True,
)
```

### Resume & load

To resume training, pass `resume_from` to `train()`. To load a fine-tuned checkpoint for inference, use `torch.load` directly:

```python
# Resume training
trainer.train(resume_from="./checkpoints/checkpoint_epoch_0009")

# Load fine-tuned model for inference
import torch
model = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vitb")
model.load_state_dict(torch.load("./checkpoints/best_model/model.pt", weights_only=True))
model.eval()
result = model.predict(pixel_values)
```

---

## Example Scripts

See `examples/` for ready-to-run scripts:

| Script | Purpose |
|---|---|
| `examples/train_depth_anything.py` | Fine-tune Depth Anything v2-Small on NYU Depth V2 |
| `examples/test_finetuned.py` | Evaluate a fine-tuned checkpoint vs. the base model |
