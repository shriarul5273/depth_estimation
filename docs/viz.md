# Visualization Toolkit

`depth_estimation.viz` provides six visualization functions for inspecting and presenting depth estimation results. All functions accept either a `DepthOutput` or a raw `np.ndarray`, and all use `matplotlib` and `opencv-python` — both are already core dependencies.

## Quick Start

```python
from depth_estimation import pipeline
from depth_estimation.viz import show_depth, compare_depths, overlay_depth

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
result = pipe("image.jpg")

# Interactive display
show_depth(result)

# Save to file
show_depth(result, colormap="inferno", title="Depth Anything V2", save="depth.png")
```

---

## `show_depth()`

```python
show_depth(
    result: DepthOutput | np.ndarray,
    colormap: str = "Spectral_r",
    title: str | None = None,
    save: str | None = None,
)
```

Display a single depth map.

| Argument | Description |
|---|---|
| `result` | `DepthOutput` (uses `.colored_depth` if present) or `(H, W)` float32 array |
| `colormap` | Matplotlib colormap. Only used when `result` has no `colored_depth`. |
| `title` | Optional figure title |
| `save` | Path to save the figure. If `None`, calls `plt.show()`. |

```python
show_depth(result)
show_depth(result.depth, colormap="magma", save="out.png")
```

---

## `compare_depths()`

```python
compare_depths(
    results: list[DepthOutput | np.ndarray],
    labels: list[str] | None = None,
    colormap: str = "Spectral_r",
    save: str | None = None,
)
```

Display multiple depth maps side by side in a 1-row subplot grid.

| Argument | Description |
|---|---|
| `results` | List of `DepthOutput` or `(H, W)` float32 arrays |
| `labels` | Optional subplot titles. Must match `len(results)` if provided. |
| `colormap` | Matplotlib colormap |
| `save` | Path to save the figure. If `None`, calls `plt.show()`. |

```python
pipe_v2 = pipeline("depth-estimation", model="depth-anything-v2-vitb")
pipe_pro = pipeline("depth-estimation", model="depth-pro")

r1 = pipe_v2("photo.jpg")
r2 = pipe_pro("photo.jpg")

compare_depths([r1, r2], labels=["Depth Anything V2", "DepthPro"], save="compare.png")
```

---

## `overlay_depth()`

```python
overlay_depth(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "inferno",
) -> np.ndarray
```

Blend a depth colormap over an RGB image. Returns `(H, W, 3)` uint8.

| Argument | Description |
|---|---|
| `image` | `(H, W, 3)` uint8 RGB image |
| `depth` | `(H, W)` float32 depth map in [0, 1] |
| `alpha` | Blending weight: `0.0` = image only, `1.0` = depth only |
| `colormap` | Matplotlib colormap |

```python
import cv2
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
overlay = overlay_depth(image, result.depth, alpha=0.5)
cv2.imwrite("overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

---

## `create_anaglyph()`

```python
create_anaglyph(
    image: np.ndarray,
    depth: np.ndarray,
    baseline: float = 0.065,
) -> np.ndarray
```

Create a red-cyan anaglyph stereo image from a monocular RGB image and depth map. Returns `(H, W, 3)` uint8. View with standard red-cyan 3D glasses.

| Argument | Description |
|---|---|
| `image` | `(H, W, 3)` uint8 RGB image |
| `depth` | `(H, W)` float32 depth map in [0, 1] |
| `baseline` | Maximum horizontal disparity as a fraction of image width. Default `0.065` ≈ 6.5 %. |

Near objects (small depth values) are shifted more than far objects, creating a 3D parallax effect.

```python
anaglyph = create_anaglyph(image, result.depth, baseline=0.08)
cv2.imwrite("anaglyph.jpg", cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR))
```

---

## `animate_3d()`

```python
animate_3d(
    image: np.ndarray,
    depth: np.ndarray,
    output_path: str,
    frames: int = 60,
    elevation: float = 20.0,
    fps: int = 15,
)
```

Produce a rotating 3D surface animation of the depth map, textured with the RGB image colors. Output format is inferred from the file extension.

| Argument | Description |
|---|---|
| `image` | `(H, W, 3)` uint8 RGB image |
| `depth` | `(H, W)` float32 depth map in [0, 1] |
| `output_path` | Destination path: `.gif` (no system deps) or `.mp4` (requires `ffmpeg`) |
| `frames` | Number of rotation frames; azimuth sweeps 0 → 360° |
| `elevation` | Fixed elevation angle in degrees |
| `fps` | Output frames per second |

The depth map is automatically downsampled to a ~128×128 grid for performance.

```python
# GIF — no extra dependencies
animate_3d(image, result.depth, "rotation.gif", frames=60, fps=15)

# MP4 — requires ffmpeg (conda install ffmpeg / apt install ffmpeg)
animate_3d(image, result.depth, "rotation.mp4", frames=120, fps=30)
```

---

## `plot_error_map()`

```python
plot_error_map(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    metric: str = "abs_rel",
    save: str | None = None,
)
```

Render a per-pixel error heatmap between a predicted depth map and ground truth. Invalid ground-truth pixels (`gt == 0`) are shown as blank.

| Argument | Description |
|---|---|
| `pred_depth` | `(H, W)` float32 predicted depth |
| `gt_depth` | `(H, W)` float32 ground-truth depth. Zeros are treated as invalid. |
| `metric` | Error formula: `"abs_rel"`, `"sq_rel"`, `"log10"`, or `"rmse"` |
| `save` | Path to save. If `None`, calls `plt.show()`. |

| Metric | Formula |
|---|---|
| `abs_rel` | `\|pred − gt\| / gt` |
| `sq_rel` | `(pred − gt)² / gt` |
| `log10` | `\|log10(pred) − log10(gt)\|` |
| `rmse` | `(pred − gt)²` (squared error per pixel) |

```python
from depth_estimation.evaluation import DepthMetrics
from depth_estimation.viz import plot_error_map

plot_error_map(pred_depth, gt_depth, metric="abs_rel", save="error_map.png")
```

---

## Input Flexibility

All functions accept either a `DepthOutput` or a raw `np.ndarray`:

```python
# Both are valid:
show_depth(result)              # DepthOutput
show_depth(result.depth)        # (H, W) float32 ndarray

compare_depths([result, result.depth])   # mixed list also works
```

When a `DepthOutput` has a pre-computed `colored_depth` (the default from `pipeline()`), the existing colorized image is used directly — the colormap argument is ignored. Pass a raw `ndarray` or set `colorize=False` when calling the pipeline if you want to control the colormap yourself.

---

## Dependencies

No new dependencies are needed — `matplotlib` and `opencv-python` are already core requirements. The `animate_3d()` function requires `ffmpeg` as a system binary only for `.mp4` output; `.gif` output uses Pillow which is already installed.
