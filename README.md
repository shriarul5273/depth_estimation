# depth_estimation

A **Python library** for monocular depth estimation.

Provides a unified, modular API for running inference, comparing, and integrating depth estimation models — supporting **12 model families** with **28 variants** and designed to accommodate new models with minimal friction.

## Installation

```bash
pip install depth-estimation
```

For a full list of core and optional dependencies, see [docs/dependencies.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/dependencies.md).

## Quick Start

| | Pipeline API | Auto Classes |
|---|---|---|
| **Setup** | One call, model + processor bundled | Load model and processor separately |
| **Inference** | Pass image path directly | Call `processor()`, `model()`, `postprocess()` manually |
| **Control** | Low — handles everything for you | High — you control each step |
| **Output** | `DepthOutput` with `.depth`, `.colored_depth`, `.metadata` | Raw depth tensor |
| **Best for** | Quick inference, scripts, demos | Custom pipelines, research, fine-grained control |

### Pipeline API (Recommended)

```python
from depth_estimation import pipeline

pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
result = pipe("image.jpg")

depth_map = result.depth            # np.ndarray, float32, (H, W)
colored   = result.colored_depth    # np.ndarray, uint8, (H, W, 3)
meta      = result.metadata         # dict with model info
```

### Auto Classes

```python
from depth_estimation import AutoDepthModel, AutoProcessor

# Works with any of the 25 supported variants
model     = AutoDepthModel.from_pretrained("zoedepth")
processor = AutoProcessor.from_pretrained("zoedepth")

inputs = processor("image.jpg")
with torch.no_grad():
    depth = model(inputs["pixel_values"])

result = processor.postprocess(depth, inputs["original_sizes"])
```

### Batch Inference

```python
results = pipe(["img1.jpg", "img2.jpg", "img3.jpg"])
for r in results:
    print(r.depth.shape)
```

## Supported Models

12 model families · 28 variants — see [docs/models.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/models.md) for the full list.

## Architecture

The library follows the **HuggingFace Transformers** modular design philosophy:

- **Single model, single file** — each model's architecture is self-contained
- **Shared processor** — preprocessing/postprocessing is not duplicated
- **Registry-based auto-loading** — new models self-register, no core changes needed
- **Config inheritance** — configs override only what differs from the base

```
Input → Processor.preprocess() → Model.forward() → Processor.postprocess() → DepthOutput
```

## Adding a New Model

1. Create `src/depth_estimation/models/your_model/`
2. Add `configuration_your_model.py` (inherit `BaseDepthConfig`)
3. Add `modeling_your_model.py` (inherit `BaseDepthModel`, single file)
4. Add `__init__.py` with `MODEL_REGISTRY.register(...)`

That's it — `AutoDepthModel`, `AutoProcessor`, and `pipeline()` will automatically resolve your model.

## CLI

After installing the package, a `depth-estimate` command is available.

```bash
# Single image → saves demo10_depth.png
depth-estimate predict demo10.png --model depth-anything-v2-vitb

# Batch (directory or glob) → saves to results/
depth-estimate predict "images/*.jpg" --model depth-anything-v2-vitb --output-dir results/

# Video → saves side-by-side RGB | depth as MP4
depth-estimate predict video.mp4 --model depth-anything-v2-vitb --output depth_video.mp4

# Save raw float32 array (.npy) alongside the PNG
depth-estimate predict demo10.png --model depth-anything-v2-vitb --format both

# Change colormap
depth-estimate predict demo10.png --model depth-anything-v2-vitb --colormap inferno

# List all available models
depth-estimate list-models

# Show config details for a model
depth-estimate info depth-anything-v2-vitb
```

**Global flags** (`--device`, `--quiet`, `--verbose`) go before the subcommand:

```bash
depth-estimate --device cpu --quiet predict demo10.png --model depth-anything-v2-vitb
```

All subcommands support `--json` for machine-readable output.

For full documentation see [docs/cli.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/cli.md).

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Acknowledgments

This library builds upon the incredible work of the following research teams:

| Model | Repository |
|---|---|
| **Depth Anything v1** | [github.com/LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything) |
| **Depth Anything v2** | [github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| **Depth Anything v3** | [github.com/DepthAnything/Depth-Anything-V3](https://github.com/DepthAnything/Depth-Anything-V3) |
| **DINOv2** | [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| **DepthPro** | [github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) |
| **ZoeDepth** | [github.com/isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth) |
| **MiDaS** | [github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS) |
| **Pixel-Perfect Depth** | [github.com/gangweix/Pixel-Perfect-Depth](https://github.com/gangweix/Pixel-Perfect-Depth) |
| **Marigold-DC** | [github.com/prs-eth/Marigold-DC](https://github.com/prs-eth/Marigold-DC) |

## License

MIT
