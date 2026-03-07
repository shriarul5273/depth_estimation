# Dependencies


## Core Dependencies

These are installed automatically with `pip install depth-estimation`

| Package | Version Constraint | Notes |
|---|---|---|
| Python | `>=3.9` | 3.9 – 3.12 supported |
| torch | `==2.10.0` | Pinned release |
| torchvision | `>=0.15` | |
| Pillow | `>=9.0` | |
| NumPy | `>=1.24` | |
| matplotlib | `>=3.6` | |
| opencv-python | `>=4.8` | |
| huggingface-hub | `>=0.16` | |
| transformers | (latest) | |
| diffusers | (latest) | |
| accelerate | (latest) | Required by diffusers |
| timm | `>=0.9.1` | |
| einops | `>=0.6` | |
| addict | (any) | |

## Optional Dependencies

Install with the extras shown below.

| Extra | Install command | Package pinned | Required for |
|---|---|---|---|
| `transformers` | `pip install "depth-estimation[transformers]"` | `transformers>=4.30` | MiDaS, ZoeDepth |
| `diffusers` | `pip install "depth-estimation[diffusers]"` | `diffusers>=0.25` | Marigold-DC |
| `all` | `pip install "depth-estimation[all]"` | both above | All models |

## Development Dependencies

| Package | Version Constraint |
|---|---|
| pytest | `>=7.0` |
| pytest-cov | (latest) |

Install with:

```bash
pip install "depth-estimation[dev]"
```
