# Pipeline API

The `pipeline()` function is the highest-level abstraction for depth estimation. It chains `AutoDepthModel` and `AutoProcessor` into a single callable — you pass an image in, a `DepthOutput` comes out.

## Quick Example

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
result = pipe("image.jpg")

result.depth          # np.ndarray, float32, shape (H, W)
result.colored_depth  # np.ndarray, uint8,   shape (H, W, 3)
result.metadata       # dict — model info, device, latency, resolution
```

## `pipeline()` Factory

```python
pipeline(
    task: str = "depth-estimation",
    model: str = None,
    device: str = None,
    **kwargs
) -> DepthPipeline
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | `"depth-estimation"` | Task name. Only `"depth-estimation"` is supported. |
| `model` | `str` | **required** | Model variant ID (e.g. `"depth-anything-v2-vitb"`). See [models.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/models.md). |
| `device` | `str` | `None` | Device string (`"cuda"`, `"cpu"`, `"mps"`). Auto-detected if `None`. |
| `**kwargs` | | | Extra arguments forwarded to `AutoDepthModel.from_pretrained()`. |

## `DepthPipeline.__call__()`

```python
pipe(
    images: str | PIL.Image | np.ndarray | list,
    batch_size: int = 1,
    colorize: bool = True,
    colormap: str = "Spectral_r",
) -> DepthOutput | list[DepthOutput]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `images` | `str`, `PIL.Image`, `np.ndarray`, or `list` | **required** | Single image or list. Accepts file paths, URLs, PIL images, or NumPy arrays. |
| `batch_size` | `int` | `1` | Number of images processed per forward pass. |
| `colorize` | `bool` | `True` | Whether to generate a colormapped RGB visualization. |
| `colormap` | `str` | `"Spectral_r"` | Any Matplotlib colormap name (e.g. `"inferno"`, `"viridis"`). |

Returns a single `DepthOutput` for a single image, or a `list[DepthOutput]` for a batch.

## `DepthOutput`

| Field | Type | Description |
|---|---|---|
| `depth` | `np.ndarray` float32 `(H, W)` | Raw depth map. Normalized to `[0, 1]` for relative models; in **metres** for metric models. |
| `colored_depth` | `np.ndarray` uint8 `(H, W, 3)` | Colormapped RGB visualization. `None` if `colorize=False`. |
| `metadata` | `dict` | Model type, backbone, device, latency (seconds), input resolution, and any model-specific fields. |

### `metadata` keys

| Key | Description |
|---|---|
| `model_type` | Model family (e.g. `"depth_anything_v2"`) |
| `backbone` | Backbone variant (e.g. `"vitb"`) |
| `device` | Device used for inference (e.g. `"cuda:0"`) |
| `latency_seconds` | Per-image inference time |
| `input_resolution` | `(H, W)` tuple of the preprocessed input fed to the model |

## Examples

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")

# Single image — accepts path, PIL.Image, or np.ndarray
result = pipe("photo.jpg")
print(result.depth.shape)          # (720, 1280)

# Batch
results = pipe(["img1.jpg", "img2.jpg"], batch_size=2)

# Custom colormap / disable colorization
result = pipe("photo.jpg", colormap="inferno")
result = pipe("photo.jpg", colorize=False)   # colored_depth is None

# Explicit device
pipe = pipeline("depth-estimation", model="depth-pro", device="cuda:0")
```

## Internal Flow

```
images
  └─ processor.preprocess()      # resize, normalize → pixel_values tensor
       └─ model.forward()        # forward pass on device (no_grad)
            └─ processor.postprocess()  # resize back to original, colorize
                 └─ DepthOutput  # depth, colored_depth, metadata
```

## Video & Streaming

`DepthPipeline` has two additional methods for video input. See [video.md](video.md) for the full reference.

### `.stream()`

```python
pipe.stream(
    source: str | int,
    batch_size: int = 1,
    colormap: str = "inferno",
    temporal_smoothing: float = 0.0,
) -> Generator[DepthOutput, ...]
```

Yields a `DepthOutput` for each frame. `metadata` includes `frame_index`, `timestamp_seconds`, and `fps` in addition to the standard keys.

```python
for result in pipe.stream("video.mp4", temporal_smoothing=0.5):
    print(result.metadata["frame_index"], result.depth.shape)
```

`temporal_smoothing` applies an exponential moving average (EMA) across frames:
`depth_t = alpha * depth_{t-1} + (1 - alpha) * depth_t`. Set to `0.0` to disable.

### `.process_video()`

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

Reads a video file, runs depth estimation on every frame, and writes the result to `output_path`. When `side_by_side=True` (default), the output frame is `[RGB | colored depth]` concatenated horizontally.

---

## When to Use Auto Classes Instead

Use `AutoDepthModel` + `AutoProcessor` directly when you need to:
- Modify preprocessing (custom resize, normalization)
- Run the model inside a training loop
- Access raw model outputs before postprocessing
- Integrate with a custom batching strategy

See the [README](https://github.com/shriarul5273/depth_estimation#auto-classes) for an example.
