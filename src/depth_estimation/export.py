"""ONNX export for depth estimation models.

Usage::

    from depth_estimation import AutoDepthModel
    from depth_estimation.export import export_onnx

    model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")
    export_onnx(model, "depth_anything_v2_vitb.onnx", input_size=518, verify=True)

Known limitations:
    - Diffusion-based models (``pixel-perfect-depth``, and likely
      ``marigold-dc``) sample internal random noise inside ``forward()``.
      Tracing freezes that call's result as a constant in the exported
      graph, so the ONNX model always reuses the same noise instead of
      sampling fresh randomness per call — the export "succeeds" but
      produces materially different output than the PyTorch model. Do
      not export these families for production use; ``verify=True``
      will catch this (it'll raise on the numerical mismatch).
"""

import inspect
import logging
from pathlib import Path
from typing import Union

import torch

logger = logging.getLogger(__name__)

# torch.onnx.export's `dynamo` kwarg didn't exist before ~torch 2.5, and its
# *default* value flipped from False to True in later releases. Pin it to
# False (the legacy TorchScript-based tracer) whenever available, so export
# behavior is consistent across the whole torch>=2.0 support range rather
# than silently depending on whichever exporter a given torch version
# happens to default to.
_ONNX_EXPORT_SUPPORTS_DYNAMO = (
    "dynamo" in inspect.signature(torch.onnx.export).parameters
)


def export_onnx(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    input_size: int = 518,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_spatial: bool = False,
    verify: bool = False,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Path:
    """Export a depth estimation model to ONNX.

    Args:
        model: A model with ``forward(pixel_values) -> (B, H, W)`` depth,
            e.g. any ``BaseDepthModel`` subclass. Must already be on its
            target device.
        output_path: Destination ``.onnx`` file path.
        input_size: Spatial size (H=W) of the dummy input used to trace the
            graph. Must be a size the model actually supports (e.g. a
            multiple of its ``config.patch_size``) — the model's own
            ``config.input_size`` is a safe default for most families.
        opset_version: ONNX opset version.
        dynamic_batch: Whether the exported graph supports a variable batch
            size at inference time.
        dynamic_spatial: Whether the exported graph supports variable H/W.
            Many of these architectures have constraints on spatial size
            (multiple-of-patch-size, or an internally fixed resolution like
            DepthPro) — leave this False unless you've confirmed the target
            model tolerates arbitrary input sizes.
        verify: If True, run the same input through both the PyTorch model
            and the exported ONNX graph (via onnxruntime, an optional
            dependency) and assert the outputs match within ``atol``/
            ``rtol``. Recommended — this is the only way to catch cases
            like the stochastic-model limitation described in this
            module's docstring.
        atol: Absolute tolerance for ``verify``.
        rtol: Relative tolerance for ``verify``.

    Returns:
        ``output_path``, as a ``Path``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    # Prefer BaseDepthModel.device: some families (DepthPro,
    # PixelPerfectDepth) have no parameters at all until their lazily-built
    # network exists, so next(model.parameters()) raises StopIteration
    # before the warm-up forward pass below has had a chance to run.
    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Warm-up: some model families (DepthPro, PixelPerfectDepth) lazily
    # build their network on the *first* forward() call rather than in
    # __init__. If that first call happens during tracing, the module's
    # parameter set changes mid-trace and torch.onnx.export raises
    # "state_dict changed after running the tracer". Running forward once
    # eagerly beforehand forces that lazy init to already be done, which is
    # harmless (idempotent) for every model family, not just the lazy ones.
    with torch.no_grad():
        model(dummy_input)

    dynamic_axes = {}
    if dynamic_batch or dynamic_spatial:
        in_axes = {}
        out_axes = {}
        if dynamic_batch:
            in_axes[0] = "batch"
            out_axes[0] = "batch"
        if dynamic_spatial:
            in_axes[2] = "height"
            in_axes[3] = "width"
            out_axes[1] = "height"
            out_axes[2] = "width"
        dynamic_axes = {"pixel_values": in_axes, "depth": out_axes}

    export_kwargs = {}
    if _ONNX_EXPORT_SUPPORTS_DYNAMO:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["depth"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes or None,
            **export_kwargs,
        )

    logger.info("Exported ONNX model to %s", output_path)

    if verify:
        _verify_onnx_export(
            model, output_path, dummy_input=dummy_input, atol=atol, rtol=rtol
        )

    return output_path


def _verify_onnx_export(
    model: torch.nn.Module,
    onnx_path: Union[str, Path],
    dummy_input: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Run ``dummy_input`` through both PyTorch and the exported ONNX graph
    and assert the outputs match. Raises ImportError if onnxruntime (an
    optional dependency, not required for export itself) isn't installed.
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "verify=True requires the optional 'onnxruntime' package: "
            "pip install onnxruntime"
        ) from e

    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: dummy_input.cpu().numpy()})[0]

    np.testing.assert_allclose(
        torch_out,
        onnx_out,
        atol=atol,
        rtol=rtol,
        err_msg=(
            "ONNX export output does not match PyTorch output. If this "
            "model samples random noise inside forward() (e.g. a "
            "diffusion-based family like pixel-perfect-depth or "
            "marigold-dc), this is expected — tracing freezes that "
            "noise as a constant. See depth_estimation.export's module "
            "docstring."
        ),
    )
    logger.info("ONNX export verified: matches PyTorch within atol=%s rtol=%s", atol, rtol)
