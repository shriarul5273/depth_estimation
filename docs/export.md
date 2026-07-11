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

Verified numerically — output compared against the PyTorch model, not just "export didn't crash":

| Family | Status |
|---|---|
| `depth-anything-v2` | ✅ Verified (all backbones) |
| `depth-anything-v3` | ✅ Verified (`depth-anything-v3-small`; requires torch with `scaled_dot_product_attention` ONNX support — see below) |
| `depth-pro` | ✅ Verified |
| `pixel-perfect-depth` | ⚠️ Exports without error but output does **not** match PyTorch — see below |
| `marigold-dc` | ⚠️ Likely the same limitation as `pixel-perfect-depth` (not independently tested — needs a real download to exercise) |
| `moge`, `vggt`, `omnivggt`, `midas`, `zoedepth` | Not tested — `moge`/`vggt`/`omnivggt` need real pretrained weights to construct at all (no offline random-weight path); `midas`/`zoedepth` wrap other libraries. Same `scaled_dot_product_attention` torch-version requirement applies to `moge`/`vggt`/`omnivggt`. |

## Known Limitations

### Diffusion-based models sample random noise inside `forward()`

`pixel-perfect-depth` calls `torch.randn(...)` inside its diffusion sampling loop to generate the initial noise. `torch.onnx.export` traces the model by running it once and recording the operations — a random call executed during that one trace gets **frozen as a constant** in the exported graph. The result: the ONNX model always reuses that one fixed noise sample instead of drawing fresh randomness on every call, so its output is structurally different from the PyTorch model (which samples new noise each time).

This isn't a bug you can work around with export options — it would require changing `PixelPerfectDepthModel.forward()`'s public signature to accept noise as an explicit input, which is a real API change outside the scope of `export_onnx()` itself. `marigold-dc` likely has the same issue (also diffusion-based) but hasn't been independently confirmed.

**Always use `verify=True` for these families** — it will raise on the mismatch rather than silently handing you a broken export.

### `scaled_dot_product_attention` requires a newer torch

Models whose attention blocks call `F.scaled_dot_product_attention` (`depth-anything-v3-*`, `pixel-perfect-depth`, `moge`, `vggt`, `omnivggt`) need a torch version whose ONNX exporter has a symbolic mapping for that op. That support was added in a torch release later than this package's declared floor (`torch>=2.0`) — exporting with `torch==2.0.1` raises:

```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator
'aten::scaled_dot_product_attention' to ONNX opset version 17 is not supported.
```

regardless of `opset_version`. Upgrade torch if you hit this. `depth-anything-v2` and `depth-pro` are unaffected — they use an older manual-attention implementation that doesn't call `scaled_dot_product_attention`.

## Design Notes

- **`dynamo` pinning**: `torch.onnx.export`'s `dynamo` kwarg didn't exist before roughly torch 2.5, and its *default* value later flipped from `False` (legacy TorchScript-based tracer) to `True` (newer `torch.export`-based exporter). `export_onnx()` pins it to `False` whenever the kwarg is available, so export behavior is consistent across the whole supported torch range instead of silently depending on whichever exporter a given torch release happens to default to.
- **Warm-up forward pass**: some families (`depth-pro`, `pixel-perfect-depth`) lazily build their network on the *first* `forward()` call rather than in `__init__`. If that first call happens during tracing, the module's parameter set changes mid-trace and `torch.onnx.export` raises `"state_dict changed after running the tracer"`. `export_onnx()` always runs one eager forward pass before tracing to force that lazy init to already be done — harmless for every family, not just the lazy ones.
