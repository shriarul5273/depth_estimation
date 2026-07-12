# Quantization

Reduce a model's numeric precision — two independent paths depending on what you're targeting.

## Quick Example

```python
from depth_estimation import AutoDepthModel, quantize_model, quantize_onnx

model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")

# In-process PyTorch — GPU inference speedup, halves memory
model.quantize(dtype="float16")

# ...or int8 dynamic quantization (CPU inference, nn.Linear only)
qmodel = quantize_model(model, dtype="int8")  # returns a NEW model, see below

# Post-export ONNX Runtime quantization — broader op coverage (covers Conv2d too)
model.export_onnx("model.onnx")
quantize_onnx("model.onnx", "model_uint8.onnx", verify=True)  # default weight_type="uint8"
```

## Which path should I use?

| | `quantize_model()` | `quantize_onnx()` |
|---|---|---|
| Operates on | A PyTorch model, in-process | An already-exported `.onnx` file |
| `float16`/`bfloat16` | ✅ Simple dtype cast, every layer | Not applicable — export first at your target precision instead |
| `int8`/`uint8` | ✅ `nn.Linear` only | `uint8` verified working end-to-end. `int8` on a model with `Conv2d` layers (every model in this package has one, in its patch embedding) requires `onnxruntime>=1.26.0` — older CPU builds (e.g. `1.23.2`, the newest available for Python 3.10 at time of writing) don't implement the `ConvInteger` op it produces. See [Known Limitations](#known-limitations). |
| `int16`/`uint16` | ❌ Not supported by PyTorch's quantization at all | ⚠️ Accepted by the API but the resulting model commonly **fails to load** — see [Known Limitations](#known-limitations) |
| Calibration data needed | No | No (uses dynamic quantization) |

## `quantize_model()`

```python
quantize_model(model: nn.Module, dtype: str = "float16") -> nn.Module
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | **required** | Any model. |
| `dtype` | `str` | `"float16"` | `"float16"` or `"bfloat16"` — a simple precision cast, every parameter/buffer affected, structure unchanged. Intended for GPU inference (float16 on CPU is slow/partially unsupported for many ops in vanilla PyTorch). Or `"int8"` — dynamic quantization of `nn.Linear` layers only, via `torch.quantization.quantize_dynamic`. **Always runs on CPU**, regardless of what device the model is on — see below. |

**Return value depends on `dtype`** — this is the one thing to watch:
- `"float16"`/`"bfloat16"`: returns `model`, mutated in-place, on whatever device it was already on.
- `"int8"`: returns a **new** model, always on CPU (verified: `result is model` → `False`, though `type(result) is type(model)` holds). The original `model` argument is left completely unmodified, including its device even if it was on GPU. Don't chain further calls assuming it's the same object or the same device.

**`"int8"` always runs on CPU, even if you call it on a GPU-resident model** — confirmed the hard way: `torch.quantization.quantize_dynamic`'s output only has CPU kernels for the quantized linear op it produces. Calling it on a CUDA model raises `NotImplementedError: Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend`, and — this part is easy to miss — simply moving the *already-quantized* result to CPU afterward doesn't fix it either (`apply_dynamic is not implemented for this packed parameter type`). `quantize_model()`/`model.quantize()` handle this for you: they deep-copy the model, move the copy to CPU, then quantize — so you never hit this, and your original GPU-resident model is untouched.

Also available as `model.quantize(dtype=...)` (`BaseDepthModel.quantize()`), with the same return-value caveat.

Raises `ValueError` for `"int16"`/`"uint16"` (not supported by PyTorch's native quantization at all) or any other unrecognized `dtype`.

## `quantize_onnx()`

```python
quantize_onnx(
    onnx_path: str | Path,
    output_path: str | Path,
    weight_type: str = "uint8",
    verify: bool = False,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> Path
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `onnx_path` | `str` or `Path` | **required** | Path to an existing `.onnx` file — e.g. from [`export_onnx()`](export.md). |
| `output_path` | `str` or `Path` | **required** | Destination for the quantized `.onnx` file. |
| `weight_type` | `str` | `"uint8"` | `"uint8"` (default — verified working end-to-end, and ONNX Runtime's own recommended default for CPU execution) or `"int8"` (needs `onnxruntime>=1.26.0` for models with `Conv2d` layers — see [Known Limitations](#known-limitations)). `"int16"`/`"uint16"` accepted but see [Known Limitations](#known-limitations). |
| `verify` | `bool` | `False` | Load the quantized model in an `onnxruntime.InferenceSession` and run one forward pass, raising if it fails to load/run or if the output diverges from the original beyond `atol`/`rtol`. **Recommended** — this is what actually caught the limitations below, rather than silently shipping a broken file. |
| `atol`, `rtol` | `float` | `5e-2` | Tolerances for `verify`'s output comparison. Looser than `export_onnx()`'s `verify` (`1e-3`) since quantization is lossy by design. |

Confirmed on `depth-anything-v2-vits`: `int8`/`uint8` quantization reduced the exported ONNX file size from 94.2 MB to 25.6 MB (3.7×).

## Known Limitations

### `quantize_model(dtype="int8")` uses a deprecated PyTorch API

`torch.quantization.quantize_dynamic` raises a `DeprecationWarning` on recent torch releases, in favor of migrating to `torchao`. This package doesn't migrate to `torchao` yet — that's a separate, new optional dependency with its own version-compat surface across our supported torch range (`torch>=2.0`) that hasn't been verified. If a future torch release actually removes the deprecated API, `quantize_model(dtype="int8")` will start raising an error. `quantize_onnx()`'s int8 path is unaffected — it's a different library (ONNX Runtime) entirely.

### `int16`/`uint16` don't actually work

- `quantize_model()` rejects `int16`/`uint16` outright with a clear error — PyTorch's native quantization stack only has `torch.qint8`/`quint8`/`qint32`, no 16-bit integer quantized dtype at all.
- `quantize_onnx()` *accepts* `weight_type="int16"`/`"uint16"` (ONNX Runtime's `QuantType` enum has `QInt16`/`QUInt16`) and the quantization step itself succeeds — but the resulting file then **fails to load** in ONNX Runtime's CPU execution provider:
  ```
  INVALID_GRAPH: Load model from quant_int16.onnx failed:
  This is an invalid model. Type Error: Type 'tensor(int16)' ...
  ```
  Confirmed by testing against `depth-anything-v2`, not assumed from the API surface. **Always pass `verify=True`** when experimenting with `int16`/`uint16` — it will raise this error immediately instead of handing you a quantized file that silently doesn't work.

### `weight_type="int8"` needs a recent-enough `onnxruntime` for `Conv2d` layers

Every model in this package has at least one `Conv2d` layer (its patch embedding). Quantizing one to `int8` produces a `ConvInteger` op, which older ONNX Runtime CPU builds don't implement at all:

```
NOT_IMPLEMENTED: Could not find an implementation for ConvInteger(10) node
with name '/net/patch_embed/proj/Conv_quant'
```

Confirmed: fails on `onnxruntime==1.23.2` (the newest release available for Python 3.10 at time of writing — caught by this package's own CI matrix, not assumed), works on `onnxruntime==1.26.0+`. `weight_type="uint8"` (the default) doesn't hit this — it produces different ops that are more broadly supported, and is what this package defaults to for exactly that reason. Pass `verify=True` if you do use `int8` — it will surface this immediately rather than shipping a quantized file that fails at load time for your users.
