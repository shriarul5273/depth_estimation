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
