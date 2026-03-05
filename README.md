# depth_estimation

A **Transformers-style Python library** for monocular depth estimation.

Provides a unified, modular API for running, comparing, and integrating depth estimation models — starting with the **Depth Anything** family (v1 & v2) and designed to accommodate new model families with minimal friction.

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url>
cd depth_estimation
pip install -e .

# With dev dependencies (pytest)
pip install -e ".[dev]"
```

### Dependencies

| Package | Min Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.0 |
| torchvision | 0.15 |
| Pillow | 9.0 |
| NumPy | 1.24 |
| matplotlib | 3.6 |
| opencv-python | 4.8 |
| huggingface-hub | 0.16 |

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

### Advanced / Component API

```python
from depth_estimation import AutoDepthModel, AutoProcessor

model     = AutoDepthModel.from_pretrained("depth-anything-v1-vitl")
processor = AutoProcessor.from_pretrained("depth-anything-v1-vitl")

inputs = processor("image.jpg")
with torch.no_grad():
    depth = model(inputs["pixel_values"].to("cuda"))

result = processor.postprocess(depth, inputs["original_sizes"])
```

### Batch Inference

```python
results = pipe(["img1.jpg", "img2.jpg", "img3.jpg"], batch_size=4)
for r in results:
    print(r.depth.shape)
```

## Supported Models

### Depth Anything v1

| Variant ID | Backbone | Weights Source |
|---|---|---|
| `depth-anything-v1-vits` | ViT-S | `LiheYoung/depth-anything-small` |
| `depth-anything-v1-vitb` | ViT-B | `LiheYoung/depth-anything-base` |
| `depth-anything-v1-vitl` | ViT-L | `LiheYoung/depth-anything-large` |

### Depth Anything v2

| Variant ID | Backbone | Weights Source |
|---|---|---|
| `depth-anything-v2-vits` | ViT-S | `depth-anything/Depth-Anything-V2-Small` |
| `depth-anything-v2-vitb` | ViT-B | `depth-anything/Depth-Anything-V2-Base` |
| `depth-anything-v2-vitl` | ViT-L | `depth-anything/Depth-Anything-V2-Large` |

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

## License

Apache 2.0
