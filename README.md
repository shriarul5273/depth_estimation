# depth_estimation

A **Transformers-style Python library** for monocular depth estimation.

Provides a unified, modular API for running, comparing, and integrating depth estimation models — supporting **7 model families** with **19 variants** and designed to accommodate new models with minimal friction.

## Installation

```bash
git clone https://github.com/shriarul5273/depth_estimation.git
cd depth_estimation
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# With all optional model dependencies
pip install -e ".[all]"
```

### Core Dependencies

| Package | Min Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.0 |
| torchvision | 0.15 |
| transformers | 4.30 |
| Pillow | 9.0 |
| NumPy | 1.24 |
| matplotlib | 3.6 |
| opencv-python | 4.8 |
| huggingface-hub | 0.16 |

### Optional Dependencies

| Group | Package | Models |
|---|---|---|
| `da3` | `depth-anything-3` | Depth Anything v3 |
| `depth-pro` | `depth-pro` | Apple DepthPro |
| `ppd` | `ppd`, `moge` | Pixel-Perfect Depth |

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

# Works with any of the 19 supported variants
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

### Depth Anything v1

| Variant ID | Backbone | Source |
|---|---|---|
| `depth-anything-v1-vits` | ViT-S | `LiheYoung/depth-anything-small` |
| `depth-anything-v1-vitb` | ViT-B | `LiheYoung/depth-anything-base` |
| `depth-anything-v1-vitl` | ViT-L | `LiheYoung/depth-anything-large` |

### Depth Anything v2

| Variant ID | Backbone | Source |
|---|---|---|
| `depth-anything-v2-vits` | ViT-S | `depth-anything/Depth-Anything-V2-Small` |
| `depth-anything-v2-vitb` | ViT-B | `depth-anything/Depth-Anything-V2-Base` |
| `depth-anything-v2-vitl` | ViT-L | `depth-anything/Depth-Anything-V2-Large` |

### Depth Anything v3

| Variant ID | Source |
|---|---|
| `depth-anything-v3-small` | `depth-anything/DA3-SMALL` |
| `depth-anything-v3-base` | `depth-anything/DA3-BASE` |
| `depth-anything-v3-large` | `depth-anything/DA3-LARGE` |
| `depth-anything-v3-giant` | `depth-anything/DA3-GIANT` |
| `depth-anything-v3-nested-giant-large` | `depth-anything/DA3NESTED-GIANT-LARGE` |
| `depth-anything-v3-metric-large` | `depth-anything/DA3METRIC-LARGE` |
| `depth-anything-v3-mono-large` | `depth-anything/DA3MONO-LARGE` |

### ZoeDepth (Metric)

| Variant ID | Source |
|---|---|
| `zoedepth` | `Intel/zoedepth-nyu-kitti` |

### MiDaS

| Variant ID | Source |
|---|---|
| `midas-dpt-large` | `Intel/dpt-large` |
| `midas-dpt-hybrid` | `Intel/dpt-hybrid-midas` |
| `midas-beit-large` | `Intel/dpt-beit-large-512` |

### Apple DepthPro (Metric)

| Variant ID | Source |
|---|---|
| `depth-pro` | `apple/DepthPro` |

### Pixel-Perfect Depth (Metric)

| Variant ID | Source |
|---|---|
| `pixel-perfect-depth` | `gangweix/Pixel-Perfect-Depth` + `Ruicheng/moge-2-vitl-normal` |

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

See `docs/adding_a_model.md` for the full guide.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Acknowledgments

This library builds upon the incredible work of the following research teams. We are deeply grateful for their open-source contributions:

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
| **MoGe** | [github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) |

Thank you to all the researchers and engineers who made their models and code publicly available. 🙏

## License

MIT
