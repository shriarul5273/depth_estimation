# Video & Streaming Inference

The `depth_estimation` library can process video files, webcam streams, and sequences of image frames. All video functionality lives in `depth_estimation.video` and is exposed as methods on `DepthPipeline`.

## Quick Start

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")

# Stream a video file — yields one DepthOutput per frame
for result in pipe.stream("video.mp4"):
    depth   = result.depth           # (H, W) float32
    colored = result.colored_depth   # (H, W, 3) uint8
    idx     = result.metadata["frame_index"]
    ts      = result.metadata["timestamp_seconds"]

# Write depth video directly to disk
pipe.process_video("input.mp4", "output_depth.mp4", side_by_side=True)
```

---

## `DepthPipeline.stream()`

```python
pipe.stream(
    source: str | int,
    batch_size: int = 1,
    colormap: str = "inferno",
    temporal_smoothing: float = 0.0,
) -> Generator[DepthOutput, None, None]
```

Yields a `DepthOutput` for every frame in `source`.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `source` | `str` or `int` | **required** | Video file path, webcam index, or glob pattern — see [Source Types](#source-types). |
| `batch_size` | `int` | `1` | Frames per forward pass. |
| `colormap` | `str` | `"inferno"` | Matplotlib colormap name for `colored_depth`. |
| `temporal_smoothing` | `float` | `0.0` | EMA coefficient. `0.0` = disabled, `0.9` = heavy smoothing. |

### `DepthOutput.metadata` — extra keys added by `stream()`

| Key | Type | Description |
|---|---|---|
| `frame_index` | `int` | 0-based frame counter |
| `timestamp_seconds` | `float` | Frame position in seconds |
| `fps` | `float` | Source frames per second |

Standard keys (`model_type`, `backbone`, `device`, `latency_seconds`, `input_resolution`) are also present.

### Source Types

| Source | Example | Description |
|---|---|---|
| Video file | `"video.mp4"` | Any format OpenCV can read: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v` |
| Webcam | `0` | Integer device index (0 = default camera) |
| Frame glob | `"frames/*.png"` | Must contain `*` or `?`; files are sorted alphabetically |

### Temporal Smoothing

`temporal_smoothing` applies an exponential moving average (EMA) on the raw depth output across consecutive frames:

```
depth_t = alpha * depth_{t-1} + (1 - alpha) * depth_t
```

Higher values produce smoother, less flickery depth maps at the cost of temporal lag. Use `0.3–0.5` for mild smoothing, `0.7–0.9` for heavy.

### Examples

```python
# Video file with smoothing
for result in pipe.stream("drive.mp4", temporal_smoothing=0.5):
    cv2.imshow("depth", cv2.cvtColor(result.colored_depth, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Webcam (press Ctrl+C to stop)
for result in pipe.stream(0, colormap="plasma"):
    ...

# Image sequence from a directory
for result in pipe.stream("dataset/frames/*.jpg"):
    np.save(f"depth/{result.metadata['frame_index']:06d}.npy", result.depth)
```

---

## `DepthPipeline.process_video()`

```python
pipe.process_video(
    input_path: str,
    output_path: str,
    colormap: str = "inferno",
    side_by_side: bool = True,
    fps: float | None = None,
    temporal_smoothing: float = 0.0,
    batch_size: int = 1,
)
```

Reads `input_path`, runs depth estimation on every frame, and writes the result to `output_path` using the `mp4v` codec.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `input_path` | `str` | **required** | Path to the input video file. |
| `output_path` | `str` | **required** | Destination path (`.mp4` recommended). |
| `colormap` | `str` | `"inferno"` | Matplotlib colormap for depth visualization. |
| `side_by_side` | `bool` | `True` | If `True`, output is `[RGB | colored depth]` concatenated horizontally. |
| `fps` | `float \| None` | `None` | Output FPS. `None` matches the source FPS. |
| `temporal_smoothing` | `float` | `0.0` | EMA smoothing coefficient. |
| `batch_size` | `int` | `1` | Frames per forward pass. |

### Examples

```python
# Side-by-side RGB | depth with smoothing
pipe.process_video(
    "dashcam.mp4",
    "dashcam_depth.mp4",
    colormap="inferno",
    side_by_side=True,
    temporal_smoothing=0.5,
)

# Depth-only output
pipe.process_video("input.mp4", "depth_only.mp4", side_by_side=False)

# Slower model, larger batch
pipe.process_video("input.mp4", "out.mp4", batch_size=4)
```

---

## `VideoStream` class

For lower-level control, use `VideoStream` directly:

```python
from depth_estimation.video import VideoStream

with VideoStream("video.mp4") as stream:
    print(f"FPS: {stream.fps}, Total frames: {stream.total_frames}")
    for frame_rgb, metadata in stream:
        # frame_rgb: (H, W, 3) uint8 RGB
        # metadata: dict with frame_index, timestamp_seconds, fps
        pass
```

### Constructor

```python
VideoStream(
    source: str | int,
    batch_size: int = 1,
    temporal_smoothing: float = 0.0,
)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `fps` | `float` | Frames per second (30.0 for glob sources) |
| `total_frames` | `int \| None` | Frame count. `None` for live webcam. |

### Methods

| Method | Description |
|---|---|
| `__iter__()` | Yields `(frame_rgb, metadata)` tuples |
| `_temporal_filter(depth, alpha)` | Apply EMA smoothing on a depth array |
| `close()` | Release the `VideoCapture` handle |
| `__enter__` / `__exit__` | Context manager — calls `close()` on exit |

---

## CLI

The `predict` subcommand detects video sources automatically:

```bash
# Video file — output is a side-by-side MP4
depth-estimate predict video.mp4 \
    --model depth-anything-v2-vitb \
    --output depth_video.mp4

# Change colormap
depth-estimate predict video.mp4 \
    --model depth-anything-v2-vitb \
    --colormap plasma \
    --output depth_plasma.mp4
```

---

## Performance Tips

- Use `depth-anything-v2-vits` (ViT-S backbone) for real-time processing on a GPU.
- Set `batch_size > 1` when GPU memory allows — processes multiple frames per forward pass.
- Use `temporal_smoothing=0.3–0.5` to reduce flickering without significant lag.
- For offline processing, `process_video()` is more efficient than manually iterating `stream()` because it avoids redundant colorization calls.
