"""Precision reduction / quantization for depth estimation models.

Two independent paths, depending on what you're targeting — **do not mix
them**: ``quantize_model(dtype="int8")``'s output cannot be exported to
ONNX at all (see below), so if you want a quantized ONNX file, always
export first, then quantize with :func:`quantize_onnx`.

- :func:`quantize_model` — in-process PyTorch precision casting/quantization.
  ``"float16"``/``"bfloat16"`` are simple dtype casts (GPU inference
  speedup, halves memory, no accuracy-recovery step needed). ``"int8"``
  uses torch's dynamic quantization of ``nn.Linear`` layers only (CPU
  inference speedup, no calibration data needed) — for **direct PyTorch
  inference only**. Confirmed: ``export_onnx()`` on an int8-quantized
  model raises ``UnsupportedOperatorError: Exporting the operator
  'quantized::linear_dynamic' to ONNX opset version 17 is not
  supported`` — torch's ONNX exporter has no symbolic mapping for that
  op at all, regardless of opset.
- :func:`quantize_onnx` — post-export quantization via ONNX Runtime,
  operating on an already-exported ``.onnx`` file (see
  :mod:`depth_estimation.export`). Broader op coverage than PyTorch's own
  dynamic quantization (covers ``Conv2d``, not just ``Linear``). **This
  is the correct path if you want a quantized ONNX file** — export the
  unquantized model first, then quantize the resulting ``.onnx``.

Known limitations (verified, not guessed):
    - ``quantize_onnx()``'s ``uint8`` accuracy is **model-dependent, not
      reliably safe across the board**. Testing across the 28 registered
      variants (not just the one model this was originally checked
      against) found 7 of them produce badly wrong quantized output —
      e.g. ``depth-anything-v1-vitb``: 100% of output elements outside
      the default 5% tolerance, magnitude nearly doubled. Not a pruning
      interaction (reproduces on an unpruned model too) and not a code
      bug — naive dynamic quantization (no calibration data) is simply a
      poor fit for some real weight distributions. This is why
      ``verify`` now defaults to ``True``: always check accuracy for your
      specific model rather than assuming ``uint8`` is safe.
    - ``quantize_model(dtype="int8")`` uses ``torch.quantization.quantize_dynamic``,
      which is **deprecated** as of recent torch releases in favor of
      ``torchao``. This package doesn't migrate to ``torchao`` yet, since
      that's a separate new dependency with its own version-compat surface
      to verify — if a future torch release actually removes the
      deprecated API, this specific path will start raising an error.
      ``quantize_onnx()``'s int8 path is unaffected (different library).
    - ``int16``/``uint16`` are **not supported by PyTorch's native
      quantization at all** (``torch.qint8``/``quint8``/``qint32`` are
      the only quantized dtypes it has) — ``quantize_model()`` raises a
      clear error for these.
    - ``quantize_onnx()``'s ``int16``/``uint16`` weight types are accepted
      by ONNX Runtime's quantization *API*, but the resulting model
      commonly **fails to load** in ONNX Runtime's CPU execution
      provider — confirmed by testing (``depth-anything-v2``):
      ``INVALID_GRAPH: Type 'tensor(int16)' ...``. Pass ``verify=True``
      to catch this rather than silently shipping a broken quantized
      model — it's exactly what caught this in the first place.
    - ``quantize_onnx()``'s ``weight_type="int8"`` produces a
      ``ConvInteger`` op for any ``Conv2d`` layer (every model in this
      package has one, in its patch embedding). Older ONNX Runtime CPU
      builds don't implement it at all — confirmed:
      ``onnxruntime==1.23.2`` (the newest available for Python 3.10 at
      time of writing) raises ``NOT_IMPLEMENTED: ... ConvInteger(10)``;
      verified working on ``onnxruntime==1.26.0+``. ``weight_type="uint8"``
      (the default here) is unaffected — it produces different, more
      broadly-supported ops, and is ONNX Runtime's own recommended
      default for CPU execution regardless.
"""

import copy
import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CAST_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# ONNX Runtime quantization weight types. int8/uint8 are verified working;
# int16/uint16 are accepted by the quantization API but commonly fail to
# load afterward — see the module docstring.
_ONNX_WEIGHT_TYPES = {"int8", "uint8", "int16", "uint16"}


def quantize_model(model: nn.Module, dtype: str = "float16") -> nn.Module:
    """Reduce a PyTorch model's numeric precision, in-process.

    Args:
        model: Any ``nn.Module``.
        dtype: ``"float16"`` or ``"bfloat16"`` — a simple precision cast;
            every parameter/buffer is affected, structure is unchanged.
            Intended for GPU inference (float16 on CPU is slow/partially
            unsupported in vanilla PyTorch for many ops). Or ``"int8"`` —
            dynamic quantization of ``nn.Linear`` layers only, via
            ``torch.quantization.quantize_dynamic``. **Always runs on
            CPU**, regardless of what device ``model`` is on — torch's
            dynamic quantization only has CPU kernels for the quantized
            op it produces (confirmed: quantizing a CUDA-resident model
            directly raises ``NotImplementedError: Could not run
            'quantized::linear_dynamic' with arguments from the 'CUDA'
            backend``, and moving the *already-quantized* result to CPU
            afterward doesn't recover it either). No calibration data
            needed.

    Returns:
        For ``"float16"``/``"bfloat16"``: ``model``, mutated in-place and
        returned for chaining.
        For ``"int8"``: a **new** model, always on CPU — the original
        ``model`` argument is left completely unmodified (including its
        device, even if it was on GPU), since this deep-copies before
        moving to CPU and quantizing rather than mutating ``model``
        in-place. **Not exportable to ONNX** — see this module's
        docstring; use :func:`quantize_onnx` instead if you need a
        quantized ``.onnx`` file.

    Raises:
        ValueError: For ``"int16"``/``"uint16"`` (not supported by
            PyTorch's native quantization at all — use
            :func:`quantize_onnx` instead) or any other unrecognized
            ``dtype``.
    """
    if dtype in _CAST_DTYPES:
        return model.to(_CAST_DTYPES[dtype])

    if dtype == "int8":
        # torch's dynamic quantization only has CPU kernels for the
        # quantized linear op it produces (confirmed: quantizing a
        # CUDA-resident model raises "Could not run 'quantized::
        # linear_dynamic' with arguments from the 'CUDA' backend", and
        # simply moving the *quantized* model to CPU afterward doesn't
        # recover it either — "apply_dynamic is not implemented for this
        # packed parameter type"). The model must be on CPU *before*
        # quantizing. Deep-copy first rather than model.to("cpu") directly
        # — nn.Module.to() mutates and returns self, which would silently
        # move the caller's original (possibly GPU-resident) model to CPU
        # as a side effect, contradicting "the original model argument is
        # left unmodified" below.
        cpu_model = copy.deepcopy(model).to("cpu")
        return torch.quantization.quantize_dynamic(
            cpu_model, {nn.Linear}, dtype=torch.qint8
        )

    if dtype in ("int16", "uint16"):
        raise ValueError(
            f"{dtype!r} is not supported by PyTorch's native quantization — "
            "torch.qint8/quint8/qint32 are the only quantized dtypes it has. "
            "Use quantize_onnx() instead, which wraps ONNX Runtime's "
            "quantization (though see that function's docstring: int16/"
            "uint16 there commonly fail to load afterward too — only "
            "int8/uint8 are verified working end-to-end)."
        )

    raise ValueError(
        f"Unknown dtype {dtype!r}. Available: {list(_CAST_DTYPES)} + ['int8'] "
        "(see quantize_onnx() for int16/uint16, with caveats)."
    )


def quantize_onnx(
    onnx_path: Union[str, Path],
    output_path: Union[str, Path],
    weight_type: str = "uint8",
    verify: bool = True,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> Path:
    """Quantize an already-exported ONNX model via ONNX Runtime's dynamic
    quantization (no calibration data needed).

    Args:
        onnx_path: Path to an existing ``.onnx`` file, e.g. from
            :func:`depth_estimation.export.export_onnx`.
        output_path: Destination for the quantized ``.onnx`` file.
        weight_type: ``"uint8"`` (default — ONNX Runtime's own recommended
            default for CPU execution, and the only weight type broadly
            supported at all) or ``"int8"`` — for a model with ``Conv2d``
            layers (every model in this package has one, in its patch
            embedding), ``"int8"`` produces a ``ConvInteger`` op that
            requires ``onnxruntime>=1.26.0``; older CPU builds (confirmed:
            ``1.23.2``, the newest available for Python 3.10 at time of
            writing) raise ``NOT_IMPLEMENTED``. ``"int16"``/``"uint16"``
            are accepted but commonly fail to *load* afterward in ONNX
            Runtime's CPU execution provider regardless of version — see
            this module's docstring. **Quantization accuracy is
            model-dependent, not just weight-type-dependent**: testing
            across the 28 registered variants found ``uint8`` produces
            badly wrong output (100% of elements outside tolerance, up to
            ~2x magnitude error) for several real pretrained checkpoints,
            not just an edge case — this is why ``verify`` defaults to
            True.
        verify: If True (the default — changed from False after the
            finding above), load the quantized model in an
            ``onnxruntime.InferenceSession`` and run one forward pass
            with random input, raising if it fails to load/run, or if
            the output doesn't loosely match the original ``onnx_path``
            model's output (quantization is lossy by design, so the
            tolerance here is much looser than
            :func:`~depth_estimation.export.export_onnx`'s ``verify``).
            Set False only if you've already confirmed accuracy for your
            specific model and want to skip the extra forward pass.
        atol, rtol: Tolerances for ``verify``'s output comparison.

    Returns:
        ``output_path``, as a ``Path``.

    Raises:
        ValueError: For an unrecognized ``weight_type``.
        ImportError: If the optional ``onnxruntime`` package isn't
            installed.
        AssertionError: If ``verify=True`` and the quantized output
            diverges beyond ``atol``/``rtol`` — this is a real,
            model-dependent risk (see ``weight_type`` above), not a rare
            edge case.
    """
    if weight_type not in _ONNX_WEIGHT_TYPES:
        raise ValueError(
            f"Unknown weight_type {weight_type!r}. Available: "
            f"{sorted(_ONNX_WEIGHT_TYPES)}"
        )

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise ImportError(
            "quantize_onnx() requires the optional 'onnxruntime' package: "
            "pip install onnxruntime"
        ) from e

    type_map = {
        "int8": QuantType.QInt8,
        "uint8": QuantType.QUInt8,
        "int16": QuantType.QInt16,
        "uint16": QuantType.QUInt16,
    }

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(str(onnx_path), str(output_path), weight_type=type_map[weight_type])
    logger.info("Quantized ONNX model (%s) written to %s", weight_type, output_path)

    if verify:
        _verify_onnx_quantization(onnx_path, output_path, atol=atol, rtol=rtol)

    return output_path


def _verify_onnx_quantization(
    original_path: Path, quantized_path: Path, atol: float, rtol: float
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "verify=True requires the optional 'onnxruntime' package: "
            "pip install onnxruntime"
        ) from e

    orig_sess = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    input_meta = orig_sess.get_inputs()[0]
    input_name = input_meta.name
    shape = [d if isinstance(d, int) else 1 for d in input_meta.shape]
    dummy = np.random.randn(*shape).astype(np.float32)

    orig_out = orig_sess.run(None, {input_name: dummy})[0]

    quant_sess = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    quant_out = quant_sess.run(None, {input_name: dummy})[0]

    np.testing.assert_allclose(
        orig_out,
        quant_out,
        atol=atol,
        rtol=rtol,
        err_msg=(
            "Quantized ONNX model's output diverges from the original beyond "
            "tolerance. For int16/uint16, this may instead show up as the "
            "quantized model failing to even load — see this module's "
            "docstring for the known limitation."
        ),
    )
    logger.info("ONNX quantization verified: matches original within atol=%s rtol=%s", atol, rtol)
