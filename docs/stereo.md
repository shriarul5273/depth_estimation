# Stereo Models (S2M2)

Stereo models take a rectified **left/right image pair** and predict the left
view's **disparity**, the horizontal pixel shift of each point between the
two views. Disparity is inverse depth: with the rectified focal length `f`
(pixels) and baseline `B`, `depth = B * f / disparity`.

The library's stereo model is [S2M2](models.md#s2m2-stereo) (`s2m2-s`, `-m`,
`-l`, `-xl`). Everything that works for monocular models also works for it,
with a pair wherever a single image would go:

| Feature | Stereo usage |
|---|---|
| [Inference](#inference) | `pipe(("left.png", "right.png"))`, `model.infer(left, right, ...)` |
| [Video](#video) | `pipe.stream((left_src, right_src))` or one side-by-side source |
| [Evaluation](#evaluation) | `evaluate("s2m2-l", "middlebury_2014")`: EPE / bad-N / D1 |
| [Fine-tuning](#fine-tuning) | `DepthTrainer` with a stereo dataset and `DisparityLoss` |
| [ONNX export](#onnx-export) | Two inputs, `left` and `right` |
| [Quantization & profiling](#quantization-and-profiling) | `quantize_model`, `profile_latency` as usual |

---

## Inference

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="s2m2-l")

result = pipe(("left.png", "right.png"))           # one pair
results = pipe([("l1.png", "r1.png"), ("l2.png", "r2.png")], batch_size=2)

# Native resolution, metric depth, occlusion and confidence
out = pipe.model.infer("left.png", "right.png",
                       focal_length_px=1000.0, baseline=0.12, doffs=0.0)
out["disparity"], out["depth"], out["occlusion"], out["confidence"]
```

`pipe(...)` returns disparity as relative depth (min-max normalised, larger =
closer). Passing a single image to a stereo model, or a pair to a monocular
model, raises a clear error.

```bash
depth-estimate predict left.png --right right.png -m s2m2-l
depth-estimate predict pair_sbs.jpg --stereo-layout sbs -m s2m2-l   # packed pair
```

---

## Video

A stereo video is either two synchronized sources (video files, frame globs
or webcam indices) passed as a tuple, or one source with both views packed
into each frame: `stereo_layout="sbs"` (left | right) or `"tab"` (left above
right), as stereo cameras such as the ZED record.

```python
for result in pipe.stream(("left.mp4", "right.mp4"), temporal_smoothing=0.5):
    ...
for result in pipe.stream("zed_capture.mp4", stereo_layout="sbs"):
    ...
pipe.process_video(("left.mp4", "right.mp4"), "depth.mp4")   # left view | depth
```

```bash
depth-estimate predict left.mp4 --right right.mp4 -m s2m2-l
depth-estimate predict zed.mp4 --stereo-layout sbs -m s2m2-l
```

The lower-level `StereoVideoStream` yields `((left, right), metadata)` and
`split_stereo_frame(frame, layout)` unpacks a single frame
(`depth_estimation.video`).

---

## Datasets

Stereo datasets return `pixel_values` `(6, H, W)` in `[0, 1]` (left RGB then
right RGB), `disparity` `(1, H, W)` in pixels (0 where invalid) and
`valid_mask` `(1, H, W)`.

| `load_dataset` name | Auto-download | Contents |
|---|---|---|
| `middlebury_2014` | Yes (Q: 44 MB) | MiddEval3 training scenes, the 15 with public ground truth. `resolution="Q"` / `"H"` / `"F"`, `occlusion="all"` / `"noc"`. `calibration(i)` gives focal length, baseline and `doffs`. |
| `kitti_2015` | No (registration) | 200 training pairs. Extract `data_scene_flow.zip` so `root/training/image_2`, `image_3` and `disp_occ_0` exist. `occlusion="all"` / `"noc"`. |
| `stereo_folder` | N/A | `left_dir`, `right_dir`, `disparity_dir` matched by file stem. Disparity as `.pfm`, `.npy` or 16-bit `.png` (÷ `disparity_scale`, default 256). |

Only the training scenes of Middlebury and KITTI have public ground truth, so
every `split` returns those.

```python
from depth_estimation import load_dataset

mb = load_dataset("middlebury_2014", resolution="Q", occlusion="noc")
mine = load_dataset("stereo_folder", left_dir="L", right_dir="R", disparity_dir="D")
```

---

## Evaluation

`evaluate()` and `compare()` detect stereo models and stereo datasets. Each
pair runs at its native resolution through `model.infer()`, and disparity is
scored directly in pixels (no alignment):

| Metric | Definition (over valid pixels) |
|---|---|
| `epe` | mean absolute disparity error (px) |
| `bad1` / `bad2` / `bad3` | fraction of pixels with error > 1 / 2 / 3 px |
| `d1` | fraction with error > 3 px **and** > 5 % of the true disparity (KITTI D1) |

Fractions are in `[0, 1]`. All are lower-is-better.

```python
from depth_estimation.evaluation import evaluate, compare

evaluate("s2m2-l", "middlebury_2014")
compare(["s2m2-s", "s2m2-l"], dataset="kitti_2015", dataset_root="/data/kitti2015")
```

```bash
depth-estimate evaluate -m s2m2-l -d middlebury --resolution Q --occlusion noc
depth-estimate evaluate -d kitti_2015 --dataset-root /data/kitti2015 --compare
```

Reference result: `s2m2-l` on Middlebury 2014 Q (all pixels, 15 scenes) gives
EPE 0.42 px, bad-2 2.5 %, D1 1.7 %.

---

## Fine-tuning

`DepthTrainer` switches to stereo mode when the model is stereo. It reads
`disparity` from the batch, trains with
[`DisparityLoss`](training.md#loss-functions-depth_estimationlosses)
(smooth-L1 by default, `disparity_loss="l1"` for plain L1), validates with
the metrics above, and selects checkpoints by `epe` (or `bad1`/`bad2`/`bad3`/`d1`
via `eval_metric`).

```python
from depth_estimation import (AutoDepthModel, DepthTrainer, DepthTrainingArguments,
                              load_dataset)
from depth_estimation.data.stereo import (get_stereo_train_transforms,
                                          get_stereo_val_transforms)

model = AutoDepthModel.from_pretrained("s2m2-s", for_training=True)
train = load_dataset("stereo_folder", left_dir=..., right_dir=..., disparity_dir=...,
                     transform=get_stereo_train_transforms(crop_size=(320, 640)))
val = load_dataset("middlebury_2014", transform=get_stereo_val_transforms())

args = DepthTrainingArguments(output_dir="ckpt", num_epochs=10, batch_size=2,
                              learning_rate=1e-5, mixed_precision=True)
DepthTrainer(model, args, train_dataset=train, eval_dataset=val).train()
```

- Stereo augmentations keep the views consistent: random scale (disparity
  rescales with the width), random crop, colour jitter (shared by both views,
  or independent with probability 0.2), and normalisation to S2M2's `[-1, 1]`.
  There is no horizontal flip, because it would swap the roles of the cameras.
  Use crop sides that are multiples of 32.
- Validation sets with mixed image sizes (Middlebury) need `batch_size=1`,
  or pass `crop_size` to `get_stereo_val_transforms`.
- For `freeze_backbone()` / `get_parameter_groups()`, the backbone is
  S2M2's feature-matching network (CNN encoder, U-Net feature pyramid,
  multi-resolution transformer). The refiners and upsampling heads are the
  decoder. `unfreeze_top_k_backbone_layers(k)` unfreezes the last `k` stages.
- In training mode S2M2 follows the trainer's `mixed_precision` setting
  instead of its inference default of float16 on CUDA.
- S2M2-S at 256×512 crops, batch 2 with mixed precision, peaks at ~3.3 GB.

---

## ONNX export

```python
model = AutoDepthModel.from_pretrained("s2m2-l")
model.export_onnx("s2m2_l_512.onnx", input_size=512, verify=True)
```

The graph has two inputs, `left` and `right`, each `(B, 3, H, W)` normalised
to `[-1, 1]`, and one output `depth` = left-view disparity `(B, H, W)`. The
batch axis is dynamic. `dynamic_spatial` is rejected because padding,
positional encodings and the cost volume are sized at trace time, so export
one graph per resolution (as upstream does). Export traces in float32 even
when the model runs in float16 on CUDA, and restores that afterwards.

---

## Quantization and profiling

```python
from depth_estimation import quantize_model
from depth_estimation.evaluation import profile_latency

quantize_model(model, "float16")       # or "bfloat16", "int8" (CPU)
profile_latency("s2m2-l", input_size=512)   # times (1, 6, 512, 512) pairs
```

Mean disparity error against float32 on a 1100×800 pair (S2M2-S): float16
0.11 px, bfloat16 0.5 px, int8 0.69 px.
