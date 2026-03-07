# 🎉 depth_estimation v0.0.3

A Python library for monocular depth estimation — **8 model families, 20 variants**, unified API.

## ✨ New Models

| Model | Variants | Integration |
|---|---|---|
| **Marigold-DC** | `marigold-dc` (metric, sparse-guided) | Wrapper over `diffusers` `MarigoldDepthPipeline` |

## 📦 Existing Models (from v0.0.2)

| Model | Variants |
|---|---|
| **Depth Anything v1** | `vits`, `vitb`, `vitl` |
| **Depth Anything v2** | `vits`, `vitb`, `vitl` |
| **Depth Anything v3** | `small`, `base`, `large`, `giant`, `nested-giant-large`, `metric-large`, `mono-large` |
| **Intel ZoeDepth** | `zoedepth` |
| **MiDaS** | `dpt-large`, `dpt-hybrid`, `beit-large` |
| **Apple DepthPro** | `depth-pro` |
| **Pixel-Perfect Depth** | `pixel-perfect-depth` |

## 🚀 Usage

```python
from depth_estimation import AutoDepthModel

# Standard depth estimation
model = AutoDepthModel.from_pretrained("marigold-dc")
depth = model(pixel_values)  # (B, H, W) tensor

# Depth completion with sparse guidance
depth = model.predict_with_guidance(
    image_pil,
    sparse_depth_np,        # (H, W) array, zeros where missing
    num_inference_steps=50,
    ensemble_size=1,
    seed=2024,
)
```

## 📋 Requirements

```
pip install depth_estimation[diffusers]
# or
pip install diffusers>=0.25
```

> **Note:** Marigold-DC requires `diffusers>=0.25`. The model lazy-loads the pipeline on first use and automatically selects `bfloat16` on CUDA.
