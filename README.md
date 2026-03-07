# depth_estimation

A **Python library** for monocular depth estimation.

Provides a unified, modular API for running inference, comparing, and integrating depth estimation models — supporting **8 model families** with **20 variants** and designed to accommodate new models with minimal friction.

## Installation

```bash
pip install depth-estimation

# With dev dependencies
pip install "depth-estimation[dev]"

# With all optional model dependencies
pip install "depth-estimation[all]"
```

For a full list of core and optional dependencies, see [docs/dependencies.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/dependencies.md).

## Quick Start

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

# Works with any of the 20 supported variants
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

8 model families · 20 variants — see [docs/models.md](https://github.com/shriarul5273/depth_estimation/blob/main/docs/models.md) for the full list.

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
