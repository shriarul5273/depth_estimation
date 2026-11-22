# Dependencies

## Core Dependencies

These are installed automatically with `pip install depth-estimation`

| Package | Version Constraint | Notes |
|---|---|---|
| Python | `>=3.10` | 3.10 – 3.12 supported |
| torch | `>=2.6` | Not hard-pinned — installs alongside your existing torch (GPU or CPU) instead of forcing a specific version. See [issue #1](https://github.com/shriarul5273/depth_estimation/issues/1). 2.6 is the oldest torch that current `transformers` and `diffusers` accept, which ZoeDepth, MiDaS and both Marigold models need. |
| torchvision | `>=0.21` | Matches torch 2.6 |
| Pillow | `>=9.0` | |
| NumPy | `>=1.24` | |
| matplotlib | `>=3.6` | |
| opencv-python | `>=4.8` | |
| huggingface-hub | `>=0.16` | |
| transformers | (latest) | |
| diffusers | `>=0.38` | Marigold-DC; Marigold V2 needs 0.38+ for the Qwen-Image pipeline classes |
| accelerate | (latest) | Required by diffusers |
| peft | (any) | LoRA adapters for Marigold V2 |
| safetensors | (any) | Checkpoint format for Depth Anything v3, VGGT, Marigold V2, Depth Pro 2 |
| timm | `>=1.0.20` | Depth Pro 2 needs the DINOv3 models first released in timm 1.0.20 |
| einops | `>=0.6` | |
| addict | (any) | |
| h5py | (any) | Required for NYU Depth V2 — already a core dependency, no extra needed |
| tqdm | (any) | Download progress bars |

## Extras

| Extra | Install command | Required for |
|---|---|---|
| `dev` | `pip install "depth-estimation[dev]"` | Running tests |
| `export` | `pip install "depth-estimation[export]"` | [ONNX export](export.md) — `onnx` is a genuine requirement for `torch.onnx.export`, not bundled with torch |
| `4bit` | `pip install "depth-estimation[4bit]"` | Optional 4-bit (NF4) loading of Marigold V2 on CUDA via `bitsandbytes` (needs torch>=2.4). Without it Marigold V2 runs in bfloat16, which needs much more GPU memory. |

`xformers` is never required: models that can use it (Depth Anything v3, MoGe) fall back to plain PyTorch attention when it isn't installed.

## Development Dependencies

| Package | Version Constraint |
|---|---|
| pytest | `>=7.0` |
| pytest-cov | (latest) |
| onnx | `>=1.14` |
| onnxruntime | (latest) — needed for `export_onnx(..., verify=True)` |

Install with:

```bash
pip install "depth-estimation[dev]"
```
