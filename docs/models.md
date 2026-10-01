# Supported Models

13 model families · 32 variants

**Column key:**
- **Inference** — `model.predict()` / `pipeline()` / CLI predict
- **CLI** — available via `depth-estimate` commands
- **Trainable** — supports `DepthTrainer` / `freeze_backbone()` / `get_parameter_groups()`

---

## Depth Anything v1

| Variant ID | Backbone | Source | Inference | CLI | Trainable |
|---|---|---|:---:|:---:|:---:|
| `depth-anything-v1-vits` | ViT-S | `LiheYoung/depth-anything-small` | ✅ | ✅ | ✅ |
| `depth-anything-v1-vitb` | ViT-B | `LiheYoung/depth-anything-base` | ✅ | ✅ | ✅ |
| `depth-anything-v1-vitl` | ViT-L | `LiheYoung/depth-anything-large` | ✅ | ✅ | ✅ |

## Depth Anything v2

| Variant ID | Backbone | Source | Inference | CLI | Trainable |
|---|---|---|:---:|:---:|:---:|
| `depth-anything-v2-vits` | ViT-S | `depth-anything/Depth-Anything-V2-Small` | ✅ | ✅ | ✅ |
| `depth-anything-v2-vitb` | ViT-B | `depth-anything/Depth-Anything-V2-Base` | ✅ | ✅ | ✅ |
| `depth-anything-v2-vitl` | ViT-L | `depth-anything/Depth-Anything-V2-Large` | ✅ | ✅ | ✅ |

## Depth Anything v3

| Variant ID | Source | Inference | CLI | Trainable |
|---|---|:---:|:---:|:---:|
| `depth-anything-v3-small` | `depth-anything/DA3-SMALL` | ✅ | ✅ | ✅ |
| `depth-anything-v3-base` | `depth-anything/DA3-BASE` | ✅ | ✅ | ✅ |
| `depth-anything-v3-large` | `depth-anything/DA3-LARGE` | ✅ | ✅ | ✅ |
| `depth-anything-v3-giant` | `depth-anything/DA3-GIANT` | ✅ | ✅ | ✅ |
| `depth-anything-v3-nested-giant-large` | `depth-anything/DA3NESTED-GIANT-LARGE` | ✅ | ✅ | ✅ |
| `depth-anything-v3-metric-large` | `depth-anything/DA3METRIC-LARGE` | ✅ | ✅ | ✅ |
| `depth-anything-v3-mono-large` | `depth-anything/DA3MONO-LARGE` | ✅ | ✅ | ✅ |

## ZoeDepth (Metric)

| Variant ID | Source | Inference | CLI | Trainable |
|---|---|:---:|:---:|:---:|
| `zoedepth` | `Intel/zoedepth-nyu-kitti` | ✅ | ✅ | ❌¹ |

¹ Wraps a HuggingFace pipeline — no directly accessible `nn.Module` parameters.

## MiDaS

| Variant ID | Source | Inference | CLI | Trainable |
|---|---|:---:|:---:|:---:|
| `midas-dpt-large` | `Intel/dpt-large` | ✅ | ✅ | ✅ |
| `midas-dpt-hybrid` | `Intel/dpt-hybrid-midas` | ✅ | ✅ | ✅ |
| `midas-beit-large` | `Intel/dpt-beit-large-512` | ✅ | ✅ | ✅ |

## Apple DepthPro (Metric)

| Variant ID | Source | Inference | CLI | Trainable |
|---|---|:---:|:---:|:---:|
| `depth-pro` | `apple/DepthPro` | ✅ | ✅ | ✅ |

## Pixel-Perfect Depth

| Variant ID | Source | Inference | CLI | Trainable |
|---|---|:---:|:---:|:---:|
| `pixel-perfect-depth` | `gangweix/Pixel-Perfect-Depth` | ✅ | ✅ | ❌² |

² `forward()` runs iterative diffusion sampling — not differentiable for supervised training.

## Marigold V2

Single-step diffusion transformer depth estimation using a Qwen-Image-Edit-2509
backbone with LoRA adapters. Outputs affine-invariant depth (log-space by default).
Requires `diffusers>=0.38`, `peft`, `safetensors`, and optionally `bitsandbytes`
for 4-bit quantization. The Qwen base model (~17 GB) must be accessible as
`Qwen/Qwen-Image-Edit-2509` on HuggingFace Hub or via a local path set in config.

| Variant ID | Checkpoint | Output | Inference | CLI | Trainable |
|---|---|---|:---:|:---:|:---:|
| `marigold-v2` | `depth/Log-base` | affine-invariant log depth | ✅ | ✅ | ❌⁵ |
| `marigold-v2-log-layered` | `depth/Log-layered` | see-through log depth (glass) | ✅ | ✅ | ❌⁵ |
| `marigold-v2-uniform` | `depth/Uniform-base` | affine-invariant linear depth | ✅ | ✅ | ❌⁵ |
| `marigold-v2-disparity` | `depth/Disparity-base` | affine-invariant disparity | ✅ | ✅ | ❌⁵ |

⁵ Wraps a Qwen-Image-Edit diffusion pipeline — fine-tuning requires the full
`marigoldv2` training framework. Access `model._transformer` (a PeftModel) for
direct LoRA parameter manipulation.

## Marigold-DC (Depth Completion)

| Variant ID | Source | Inference | CLI | Trainable |
|---|---|:---:|:---:|:---:|
| `marigold-dc` | `prs-eth/marigold-depth-v1-0` | ✅ | ✅ | ❌¹ |

¹ Wraps a diffusers pipeline — not directly trainable.

## MoGe (Metric)

| Variant ID | Backbone | Source | Inference | CLI | Trainable |
|---|---|---|:---:|:---:|:---:|
| `moge-v1` | ViT-L | `Ruicheng/moge-vitl` | ✅ | ✅ | ❌³ |
| `moge-v2-vitl` | ViT-L | `Ruicheng/moge-2-vitl` | ✅ | ✅ | ❌³ |
| `moge-v2-vitl-normal` | ViT-L | `Ruicheng/moge-2-vitl-normal` | ✅ | ✅ | ❌³ |
| `moge-v2-vitb-normal` | ViT-B | `Ruicheng/moge-2-vitb-normal` | ✅ | ✅ | ❌³ |
| `moge-v2-vits-normal` | ViT-S | `Ruicheng/moge-2-vits-normal` | ✅ | ✅ | ❌³ |

³ `forward()` calls `infer()` which includes non-differentiable Gauss-Newton focal/shift recovery.

## OmniVGGT (Metric)

| Variant ID | Backbone | License | Source | Inference | CLI | Trainable |
|---|---|---|---|:---:|:---:|:---:|
| `omnivggt` | ViT-L | MIT | `Livioni/OmniVGGT` | ✅ | ✅ | ✅ |

## VGGT (Metric)

| Variant ID | Backbone | Source | Inference | CLI | Trainable |
|---|---|---|:---:|:---:|:---:|
| `vggt` | ViT-L | `facebook/VGGT-1B` | ✅ | ✅ | ✅ |
| `vggt-commercial` | ViT-L | `facebook/VGGT-1B-Commercial` | ✅⁴ | ✅⁴ | ✅⁴ |

⁴ Gated on Hugging Face — downloading requires a logged-in `huggingface-hub` session that has accepted the repo's license at its Hugging Face page (set `HF_TOKEN`, or run `huggingface-cli login`). Without this, loading raises `GatedRepoError` (401 Unauthorized). Confirmed: this is the only variant of the 28 that fails in this package's own scheduled slow-test CI run, precisely because that runner has no such login configured.

---

## Citations

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
| **Marigold V2** | [github.com/huawei-bayerlab/marigold-v2](https://github.com/huawei-bayerlab/marigold-v2) |
| **MoGe** | [github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) |
| **VGGT** | [github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| **OmniVGGT** | [github.com/Livioni/OmniVGGT](https://github.com/Livioni/OmniVGGT) |
