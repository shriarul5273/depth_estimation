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

## Extras

| Extra | Install command | Required for |
|---|---|---|
| `data` | `pip install "depth-estimation[data]"` | NYU Depth V2 (h5py) |
| `dev` | `pip install "depth-estimation[dev]"` | Running tests |

## Development Dependencies

| Package | Version Constraint |
|---|---|
| pytest | `>=7.0` |
| pytest-cov | (latest) |

Install with:

```bash
pip install "depth-estimation[dev]"
```
