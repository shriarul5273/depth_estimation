# ONNX Export

Export a depth estimation model to ONNX for deployment outside PyTorch (ONNX Runtime, TensorRT, mobile, etc.).

## Installation

`onnx` is a genuine requirement for `torch.onnx.export` to work at all — it's not bundled with `torch`. It's an optional dependency of this package, not installed by default:

```bash
pip install "depth-estimation[export]"
```

Verifying an export (`verify=True`) additionally needs `onnxruntime`, which isn't part of the `export` extra (it's a dev/test dependency here). Install it directly if you want to use `verify=True`:

```bash
pip install onnxruntime
```

## Quick Example

```python
from depth_estimation import AutoDepthModel, export_onnx

model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")
export_onnx(model, "depth_anything_v2_vitb.onnx", input_size=518, verify=True)
```

Or via `BaseDepthModel.export_onnx()`:

```python
model.export_onnx("model.onnx", verify=True)
```

Or from the command line — see [docs/cli.md](cli.md#export).

## `export_onnx()`

```python
export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: int = 518,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_spatial: bool = False,
    verify: bool = False,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Path
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | **required** | A model with `forward(pixel_values) -> (B, H, W)` depth, e.g. any `BaseDepthModel` subclass. |
| `output_path` | `str` or `Path` | **required** | Destination `.onnx` file path. |
| `input_size` | `int` | `518` | Spatial size (H=W) of the dummy input used to trace the graph. Must be a size the model actually supports — e.g. a multiple of `config.patch_size`. `config.input_size` is a safe default for most families. |
| `opset_version` | `int` | `17` | ONNX opset version. |
| `dynamic_batch` | `bool` | `True` | Whether the exported graph supports a variable batch size at inference time. |
| `dynamic_spatial` | `bool` | `False` | Whether the exported graph supports variable H/W. Many families have constraints on input size (multiple-of-patch-size, or an internally fixed resolution like DepthPro) that this doesn't validate — leave `False` unless you've confirmed the target model tolerates arbitrary sizes. |
| `verify` | `bool` | `False` | Run the same input through both PyTorch and the exported ONNX graph (via `onnxruntime`) and assert the outputs match within `atol`/`rtol`. Recommended — see [Known Limitations](#known-limitations) below for why. |
| `atol`, `rtol` | `float` | `1e-3` | Tolerances for `verify`. |

Returns `output_path` as a `Path`.

## Supported Models

Verified numerically — output compared against the PyTorch model, not just "export didn't crash". All of the following was confirmed by running the full prune → export → ONNX inference → quantize pipeline against real pretrained weights and a real (non-square) photo, on both CPU and a real GPU — not just synthetic square test images:

| Family | Status |
|---|---|
| `depth-anything-v1`, `depth-anything-v2` | ✅ Verified (all backbones) |
| `depth-anything-v3` | ✅ Verified (most variants; `-giant`/`-nested-giant-large` need more VRAM than an 8GB GPU has alongside a live PyTorch model — not an export bug, see [GPU ONNX Inference](#gpu-onnx-inference-onnxruntime-gpu)); requires torch with `scaled_dot_product_attention` ONNX support — see below |
| `depth-pro` | ✅ Verified |
| `moge-v1`, `moge-v2-*` | ✅ Verified — but see [MoGe's `onnx_compatible_mode`](#moges-onnx_compatible_mode) below for a real caveat about its focal/shift recovery step |
| `pixel-perfect-depth` | ⚠️ Exports without error but output does **not** match PyTorch — see below |
| `marigold-dc`, `zoedepth` | ❌ **Not exportable at all** — `export_onnx()` raises `NotImplementedError` immediately rather than attempting a doomed trace. See [Opaque-pipeline models](#opaque-pipeline-models-are-not-exportable) below. |
| `midas-dpt-large`, `midas-dpt-hybrid`, `midas-beit-large` | Inference (non-export) confirmed working on real non-square images. Export itself hasn't been independently verified. |
| `vggt`, `omnivggt` | Not tested — need real pretrained weights to construct at all (no offline random-weight path), and (for `vggt`) more VRAM than fits alongside a live PyTorch model on an 8GB GPU. Same `scaled_dot_product_attention` requirement as `moge`. |

## Known Limitations

### Diffusion-based models sample random noise inside `forward()`

`pixel-perfect-depth` calls `torch.randn(...)` inside its diffusion sampling loop to generate the initial noise. `torch.onnx.export` traces the model by running it once and recording the operations — a random call executed during that one trace gets **frozen as a constant** in the exported graph. The result: the ONNX model always reuses that one fixed noise sample instead of drawing fresh randomness on every call, so its output is structurally different from the PyTorch model (which samples new noise each time).

This isn't a bug you can work around with export options — it would require changing `PixelPerfectDepthModel.forward()`'s public signature to accept noise as an explicit input, which is a real API change outside the scope of `export_onnx()` itself.

**Always use `verify=True` for this family** — it will raise on the mismatch rather than silently handing you a broken export.

### Opaque-pipeline models are not exportable

`zoedepth` and `marigold-dc` both wrap an opaque external pipeline (`transformers.pipeline()` / `diffusers.MarigoldDepthPipeline`) internally and round-trip the input tensor through PIL/numpy inside `forward()`, rather than staying in pure differentiable PyTorch ops. Confirmed neither is meaningfully traceable:

- **`zoedepth`** crashes *during* tracing with `AttributeError: 'Tensor' object has no attribute 'astype'` — a real bug inside `transformers`' `ZoeDepthImageProcessor` that only manifests under `torch.onnx.export`'s tracing context (normal eager inference is unaffected; `np.round()` on a traced value returns a `Tensor` instead of a plain scalar, and the processor calls `.astype()` on it).
- **`marigold-dc`**'s trace "succeeds" with no error, but the resulting graph has **zero declared inputs** — the pixel tensor gets converted to a PIL image via `.numpy()` before any traceable op uses it, so the tracer never connects any op back to the actual input. The exported `.onnx` file would just replay one memorized output regardless of what image it's given — not merely inaccurate (like `pixel-perfect-depth` above), completely useless.

`BaseDepthModel._onnx_exportable = False` is set on both classes; `export_onnx()` checks this immediately and raises `NotImplementedError` before attempting any trace, rather than surfacing a confusing third-party crash or wasting time on `marigold-dc`'s (real) diffusion sampling only to discover the result is useless. If you hit the zero-input pattern on some *other* model in the future, `export_onnx()` also detects it generically after tracing and raises a clear `RuntimeError` rather than the confusing `IndexError` onnxruntime gives you otherwise.

### MoGe's `onnx_compatible_mode`

`moge-v1`/`moge-v2-*` use several ops with no ONNX symbolic mapping at opset 17: antialiased bicubic/bilinear resizing (`aten::_upsample_{bicubic,bilinear}2d_aa`), an in-place bitwise AND (`aten::__iand_`), and `autocast`-induced mixed fp16/fp32 dtypes baked into the graph. `export_onnx()` automatically sets `model.onnx_compatible_mode = True` before tracing for any model that defines this flag (currently just `moge`), which swaps these out for exportable equivalents.

Separately, both MoGe versions recover an optimal focal length/depth shift via a **non-differentiable NumPy Gauss-Newton solve** inside `infer()`. That gets frozen as a constant at trace time — the same category of issue as `pixel-perfect-depth`'s frozen noise, except deterministic, so `verify=True` (which reuses the same input for both sides of the comparison) **cannot catch it**: the exported model will not recompute this correction for a genuinely different image at deployment time, even though `verify` passes cleanly. There's no current fix for this beyond being aware of it — it would require reimplementing the Gauss-Newton solve in pure differentiable PyTorch.

### `dynamic_spatial=True` is unsafe for DINOv2-backed models

It doesn't error at export time, but silently produces a broken graph. DINOv2's position-embedding interpolation is skipped via a Python conditional — `if npatch == N and w == h: return pos_embed` — that the tracer resolves once, against the square dummy input, baking in "skip interpolation" as a constant regardless of `dynamic_axes`. The exported graph then only actually works at the exact square shape it was traced with; feeding it any other shape raises a broadcast shape error deep in the graph:

```
Non-zero status code returned while running Add node ... Attempting to
broadcast an axis by a dimension other than 1. 1370 by 1407
```

not a clear "dynamic_spatial unsupported" message — and the PyTorch model itself handles the same non-square input just fine, so this is purely an export limitation, not a real model constraint. Affects `depth-anything-v1`/`v2`, `moge`, and likely other DINOv2-backed families. Leave `dynamic_spatial=False` (the default) for these.

### TF32 can make `verify=True` misleading on GPU

cuDNN's TF32 mode (default-on for Ampere+ GPUs, including anything from the last several NVIDIA generations) trades precision for speed on conv/matmul ops. Confirmed: on a real pretrained+pruned model, TF32 alone pushed the max output difference between PyTorch and the exported ONNX graph to **~0.19** — dwarfing `verify`'s default `1e-3` tolerance — versus **~8e-5** with TF32 off. onnxruntime's CPU execution provider always computes in full FP32, so a GPU-resident PyTorch model compared against it isn't a fair comparison unless TF32 is accounted for.

`export_onnx()` handles this for you: `verify=True`'s internal comparison forward pass runs with `torch.backends.cudnn.allow_tf32` temporarily disabled, then restores your original setting afterward. You don't need to do anything — this is just why `verify=True` on GPU won't spuriously fail (or pass with far looser real agreement than the tolerance implies) the way it used to.

### `scaled_dot_product_attention` requires a newer torch

Models whose attention blocks call `F.scaled_dot_product_attention` (`depth-anything-v3-*`, `pixel-perfect-depth`, `moge`, `vggt`, `omnivggt`) need a torch version whose ONNX exporter has a symbolic mapping for that op. That support was added in a torch release later than this package's declared floor (`torch>=2.0`) — exporting with `torch==2.0.1` raises:

```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator
'aten::scaled_dot_product_attention' to ONNX opset version 17 is not supported.
```

regardless of `opset_version`. Upgrade torch if you hit this. `depth-anything-v2` and `depth-pro` are unaffected — they use an older manual-attention implementation that doesn't call `scaled_dot_product_attention`.

## GPU ONNX Inference (`onnxruntime-gpu`)

Everything above uses `onnxruntime`'s CPU execution provider (`verify=True` always does, by design, for a stable comparison baseline). For actual GPU inference on the exported `.onnx` file, install `onnxruntime-gpu` instead of (not alongside) plain `onnxruntime` — they share the same importable `onnxruntime` module and installing one after the other silently leaves whichever was installed last active:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print(sess.get_providers())  # confirm CUDAExecutionProvider is actually active
```

Confirmed working end-to-end on a real GPU across multiple model families (`depth-anything-v1/v2/v3`, `moge`) — both the unquantized and `uint8`-quantized graphs ran correctly under `CUDAExecutionProvider`. `CPUExecutionProvider` is listed as a fallback: onnxruntime silently falls back per-node to CPU for any op the CUDA provider doesn't implement, so keeping it in the list avoids a hard failure for edge cases rather than causing one.

## Design Notes

- **`dynamo` pinning**: `torch.onnx.export`'s `dynamo` kwarg didn't exist before roughly torch 2.5, and its *default* value later flipped from `False` (legacy TorchScript-based tracer) to `True` (newer `torch.export`-based exporter). `export_onnx()` pins it to `False` whenever the kwarg is available, so export behavior is consistent across the whole supported torch range instead of silently depending on whichever exporter a given torch release happens to default to.
- **Warm-up forward pass**: some families (`depth-pro`, `pixel-perfect-depth`) lazily build their network on the *first* `forward()` call rather than in `__init__`. If that first call happens during tracing, the module's parameter set changes mid-trace and `torch.onnx.export` raises `"state_dict changed after running the tracer"`. `export_onnx()` always runs one eager forward pass before tracing to force that lazy init to already be done — harmless for every family, not just the lazy ones.
- **`_onnx_exportable` opt-out**: any `BaseDepthModel` subclass can set the class attribute `_onnx_exportable = False` to have `export_onnx()` reject it immediately with a clear message, instead of a confusing failure partway through tracing. Currently set on `ZoeDepthModel` and `MarigoldDCModel` — see [Opaque-pipeline models](#opaque-pipeline-models-are-not-exportable) above.
- **`onnx_compatible_mode` opt-in**: any model can expose a settable `onnx_compatible_mode` property; `export_onnx()` sets it to `True` before tracing if present (a no-op for every model that doesn't define it). Use this to route around ops with no ONNX symbolic mapping — see [MoGe's `onnx_compatible_mode`](#moges-onnx_compatible_mode) above for the reference implementation.
