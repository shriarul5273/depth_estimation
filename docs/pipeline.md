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

### Single image

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
result = pipe("photo.jpg")

print(result.depth.shape)      # (720, 1280)
print(result.metadata)
```

### Batch inference

```python
results = pipe(["img1.jpg", "img2.jpg", "img3.jpg"], batch_size=2)
for r in results:
    print(r.depth.shape, r.metadata["latency_seconds"])
```

### Custom colormap

```python
result = pipe("photo.jpg", colormap="inferno")
```

### Disable colorization

```python
result = pipe("photo.jpg", colorize=False)
# result.colored_depth is None
```

### Explicit device

```python
pipe = pipeline("depth-estimation", model="depth-pro", device="cuda:0")
```

### PIL image or NumPy array input

```python
from PIL import Image
import numpy as np

img_pil = Image.open("photo.jpg")
result = pipe(img_pil)

img_np = np.array(img_pil)
result = pipe(img_np)
```

## Internal Flow

```
images
  └─ processor.preprocess()      # resize, normalize → pixel_values tensor
       └─ model.forward()        # forward pass on device (no_grad)
            └─ processor.postprocess()  # resize back to original, colorize
                 └─ DepthOutput  # depth, colored_depth, metadata
```

## When to Use Auto Classes Instead

Use `AutoDepthModel` + `AutoProcessor` directly when you need to:
- Modify preprocessing (custom resize, normalization)
- Run the model inside a training loop
- Access raw model outputs before postprocessing
- Integrate with a custom batching strategy

See the [README](https://github.com/shriarul5273/depth_estimation#auto-classes) for an example.
