# v0.2.0 — 🍎 Depth Pro 2, Marigold V2 & Dependency Fixes

This release adds two model families — Apple Depth Pro 2 and Marigold V2 —
and fixes the package's dependency floors so that a plain install can run every
model. The library now covers **14 model families and 36 variants**.

## ✨ New models

- **Apple Depth Pro 2** (`depth-pro-2-s`, `-b`, `-l`, `-h`): sharp, temporally
  consistent metric depth from a DINOv3 multi-view transformer, with Apple's
  focal-length estimator (`apple/DepthPro2-focal`) recovering metric scale when
  the focal length is unknown. The network is ported into a single
  self-contained file, with outputs identical to Apple's reference
  implementation for all four sizes. `model.infer()` accepts a known focal
  length and multi-frame clips whose frames attend to each other. Trainable with
  `DepthTrainer`. Weights are research-only under Apple's AMLR license.
- **Marigold V2** (`marigold-v2`, `-log-layered`, `-uniform`, `-disparity`):
  single-step depth from a Qwen-Image-Edit DiT with LoRA adapters. The default
  variant uses the paper's `depth/Log-stage2` checkpoint. It runs in 4-bit on
  CUDA with the new `[4bit]` extra (~19 GB GPU memory) or in bfloat16 otherwise
  (~46 GB).

## ⚠️ Dependency changes

The minimum versions now reflect what the models actually need:

- `torch>=2.6` and `torchvision>=0.21` (previously `>=2.0` / `>=0.15`). Current
  `transformers` and `diffusers` reject older PyTorch, which broke ZoeDepth,
  MiDaS, Marigold-DC and Marigold V2 on the old floor. **If you are on
  PyTorch < 2.6, upgrade before installing this release.**
- `timm>=1.0.20` for the DINOv3 backbones used by Depth Pro 2.
- `diffusers>=0.38`, `peft` and `safetensors` are now core dependencies.
- New optional extra: `pip install "depth-estimation[4bit]"` installs
  `bitsandbytes` for 4-bit Marigold V2 on CUDA.

## 🐛 Fixes

- `DepthProcessor.postprocess()` and `evaluate()` now accept bfloat16 and
  float16 model outputs instead of failing in NumPy conversion.

## 🧪 Testing

- Every variant except the gated `vggt-commercial` was run with real weights
  on an A100 80GB GPU, and the full fast and slow test suites pass.
- CI's minimum-PyTorch job now tests PyTorch 2.6.0 on Python 3.10–3.12.

---

# v0.1.3 — 📚 Documentation Refresh

This patch release refreshes the package documentation so installation,
model selection, and deployment workflows are easier to discover and follow.
There are no runtime API changes in this release.

## ✨ Improvements

- Expanded the README to cover the complete workflow, including video,
  training, evaluation, ONNX export, pruning, and quantization.
- Added a documentation index linking directly to the model, data, training,
  evaluation, CLI, and deployment guides.
- Split installation instructions into core, ONNX export, and verified
  export/quantization paths.
- Corrected the ONNX dependency guidance: export requires the `export` extra,
  while verification and ONNX quantization also require `onnxruntime` or
  `onnxruntime-gpu`.
- Linked the end-to-end optimization example and clarified export compatibility
  and reliability guarantees.

---

# v0.1.2 — 📦 Export, Optimization & Reliability

This release adds a complete model-optimization path—from pruning through ONNX
export and post-export quantization—and hardens inference, dataset handling, and
CI across the supported Python and PyTorch versions.

## ✨ Highlights

- **ONNX export:** export supported models through `export_onnx()`,
  `BaseDepthModel.export_onnx()`, or `depth-estimate export`, with optional
  ONNX Runtime verification and dynamic-batch support.
- **Model pruning:** use `prune_model()`, `compute_sparsity()`, and
  `make_pruning_permanent()` for unstructured PyTorch pruning workflows.
- **Quantization:** cast models to `float16`/`bfloat16`, dynamically quantize
  linear layers to int8, or quantize exported ONNX weights with ONNX Runtime.
- **Optimization example:** `examples/optimize.py` demonstrates a verified
  prune → export → quantize workflow.

## 🐛 Fixes and hardening

- Fixed ONNX export for MoGe, froze inference-only Marigold components, and
  added clear early failures for model families that cannot produce usable graphs.
- Made ONNX verification reliable on CUDA by disabling TF32 only during the
  comparison and restoring the caller's setting afterward.
- Fixed float16 export inputs, CUDA int8 quantization, non-square MiDaS input
  handling, and GPU-index selection for DepthPro and ZoeDepth.
- Enabled verification by default for ONNX quantization so materially incorrect
  quantized outputs are rejected instead of silently shipped.
- Protected dataset archive extraction against path traversal and loaded
  externally downloaded checkpoints with `weights_only=True` where applicable.
- Made gated `vggt-commercial` weights skip cleanly when Hugging Face access has
  not been granted, without masking failures from other model variants.

## 🧪 Tooling and compatibility

- Added CI coverage across Python 3.10–3.12 and multiple PyTorch versions,
  together with Ruff, build checks, CodeQL, scheduled slow tests, and Dependabot.
- Package `__version__` now comes from installed package metadata, preventing
  source and PyPI version drift.
- Expanded fast and pretrained-model regression coverage for export, pruning,
  quantization, CLI behavior, devices, real images, and non-square inputs.

See [export.md](export.md), [pruning.md](pruning.md), and
[quantization.md](quantization.md) for supported models and current limitations.

---

# v0.1.1 — 🎨 Visualization Toolkit

## ✨ New: `depth_estimation.viz`

A dedicated visualization sub-package with six functions:

| Function | What it does |
|---|---|
| `show_depth(result, colormap, title, save)` | 🖼️ Display a single depth map |
| `compare_depths(results, labels, colormap, save)` | 🔲 Side-by-side subplot grid |
| `overlay_depth(image, depth, alpha, colormap)` | 🎭 Blend depth colormap over RGB |
| `create_anaglyph(image, depth, baseline)` | 🔴🔵 Red-cyan stereoscopic anaglyph |
| `animate_3d(image, depth, output_path, frames, elevation, fps)` | 🌀 Rotating 3D surface → GIF/MP4 |
| `plot_error_map(pred, gt, metric, save)` | 🗺️ Per-pixel error heatmap |

```python
from depth_estimation.viz import show_depth, compare_depths, overlay_depth
show_depth(result, save="depth.png")
compare_depths([r1, r2], labels=["DA V2", "MiDaS"], save="compare.png")
overlay = overlay_depth(image, result.depth, alpha=0.5)
```

See [docs/viz.md](viz.md) for the full API reference.

---

# v0.1.0 — 🎬 Video & Streaming Inference

## ✨ New: `VideoStream` and `pipe.stream()`

Stream depth from video files, webcams, or frame globs:

```python
pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")

for result in pipe.stream("video.mp4", temporal_smoothing=0.5):
    depth = result.depth
    print(result.metadata["frame_index"])

pipe.process_video("input.mp4", "output.mp4", side_by_side=True)
```

### 🆕 What's new

- 📹 `VideoStream` class — iterable over video files (`cv2.VideoCapture`), webcam device index, or frame glob patterns
- 🔁 `DepthPipeline.stream()` — yields `DepthOutput` per frame with frame metadata
- 💾 `DepthPipeline.process_video()` — reads video, runs inference, writes side-by-side or depth-only MP4
- 📈 EMA temporal smoothing via `temporal_smoothing` parameter (0.0 = disabled)
- ⏳ `tqdm` progress bar for `process_video()`

See [docs/video.md](video.md) for the full API reference.

## 🐛 Bug Fix: PyTorch version pin

Downgraded the required `torch` version from `2.10.0` to `2.9.0` to fix a compatibility issue.
