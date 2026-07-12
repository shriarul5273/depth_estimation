"""
Optimize a model for deployment: prune, quantize, and export to ONNX.

Demonstrates the three model-optimization utilities together — see
docs/pruning.md, docs/quantization.md, and docs/export.md for the full
reference on each.

Usage
-----
# Export only (no pruning/quantization):
python examples/optimize.py --model depth-anything-v2-vits --output model.onnx

# Prune 30% of weights, then export:
python examples/optimize.py --model depth-anything-v2-vits --prune 0.3 --output pruned.onnx

# Prune, then quantize the exported ONNX graph to uint8:
python examples/optimize.py --model depth-anything-v2-vits --prune 0.3 \\
    --quantize-onnx uint8 --output optimized.onnx

# float16 cast before export (GPU-oriented — see docs/quantization.md):
python examples/optimize.py --model depth-anything-v2-vits --quantize-dtype float16 \\
    --output model_fp16.onnx

Requires the optional `onnx` package: pip install "depth-estimation[export]"
(and `onnxruntime` for --verify / --quantize-onnx).
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prune, quantize, and export a depth model to ONNX."
    )
    p.add_argument(
        "--model",
        default="depth-anything-v2-vits",
        help="Model variant to optimize (default: depth-anything-v2-vits)",
    )
    p.add_argument(
        "--output",
        default="model.onnx",
        help="Destination .onnx file path (default: model.onnx)",
    )
    p.add_argument(
        "--prune",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="Prune this fraction of weights (e.g. 0.3) before quantizing/exporting. "
        "Skipped by default.",
    )
    p.add_argument(
        "--quantize-dtype",
        choices=["float16", "bfloat16", "int8"],
        default=None,
        help="In-process PyTorch precision cast/quantization, applied before export. "
        "Skipped by default. NOTE: 'int8' here is NOT usable in this script — "
        "its output can't be exported to ONNX at all (no symbolic mapping for "
        "the op it produces); use --quantize-onnx for ONNX-compatible int8/"
        "uint8 instead. See docs/quantization.md.",
    )
    p.add_argument(
        "--quantize-onnx",
        choices=["uint8", "int8", "int16", "uint16"],
        default=None,
        metavar="WEIGHT_TYPE",
        help="Post-export ONNX Runtime quantization (broader op coverage than "
        "--quantize-dtype, e.g. covers Conv2d). Skipped by default. See "
        "docs/quantization.md for which weight types are actually verified "
        "working (int16/uint16 are not).",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Spatial size (H=W) for the ONNX export trace (default: 518).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Verify the export (and ONNX quantization, if requested) actually "
        "matches the source model's output. Requires onnxruntime. Recommended.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device to load the model on: cuda, cpu, mps. Auto-detected when omitted.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.quantize_dtype == "int8":
        # quantize_model(dtype="int8")'s output cannot be exported to ONNX
        # at all — torch.onnx.export has no symbolic mapping for the
        # quantized::linear_dynamic op it produces, at any opset (confirmed:
        # raises UnsupportedOperatorError). Since this script always exports,
        # that combination can never work here. See docs/quantization.md.
        sys.exit(
            "--quantize-dtype int8 cannot be combined with ONNX export (this "
            "script always exports) — torch.onnx.export has no symbolic "
            "mapping for the quantized op it produces. Use --quantize-onnx "
            "instead (export first, then quantize the ONNX graph), or drop "
            "--quantize-dtype int8 if you only need a plain PyTorch model. "
            "See docs/quantization.md."
        )

    from depth_estimation import AutoDepthModel
    from depth_estimation.export import export_onnx
    from depth_estimation.pruning import prune_model, compute_sparsity

    logger.info(f"Loading model: {args.model}")
    model = AutoDepthModel.from_pretrained(args.model, device=args.device)

    if args.prune is not None:
        logger.info(f"Pruning {args.prune:.0%} of weights...")
        prune_model(model, amount=args.prune)
        sparsity = compute_sparsity(model)["overall"]
        logger.info(f"Achieved overall sparsity: {sparsity:.3f}")

    export_path = args.output
    # export_onnx()'s default verify tolerance (atol=rtol=1e-3) is tuned for
    # float32 round-tripping through the tracer, not for a model already
    # cast to half precision — confirmed against a real pretrained
    # checkpoint (depth-anything-v2-vits): a float16 cast produces ~0.005%
    # of output elements differing by up to ~0.004, comfortably outside
    # 1e-3 but expected rounding noise for float16 (~3 decimal digits of
    # precision), not an export bug. Loosen verify's tolerance to match
    # whenever the model has been cast to half precision beforehand.
    export_verify_kwargs = {}
    if args.quantize_dtype in ("float16", "bfloat16"):
        export_verify_kwargs = {"atol": 1e-2, "rtol": 1e-2}

    if args.quantize_dtype is not None:
        from depth_estimation.quantization import quantize_model

        logger.info(f"Quantizing (in-process, dtype={args.quantize_dtype})...")
        model = quantize_model(model, dtype=args.quantize_dtype)

    logger.info(f"Exporting to ONNX (input_size={args.input_size})...")
    if args.quantize_onnx is not None:
        # Export to a temp path first, then quantize into the requested output.
        base_path = export_path + ".pre_quant.onnx"
        export_onnx(
            model,
            base_path,
            input_size=args.input_size,
            verify=args.verify,
            **export_verify_kwargs,
        )

        from depth_estimation.quantization import quantize_onnx

        logger.info(f"Quantizing exported ONNX graph (weight_type={args.quantize_onnx})...")
        quantize_onnx(
            base_path,
            export_path,
            weight_type=args.quantize_onnx,
            verify=args.verify,
        )
        os.remove(base_path)
    else:
        export_onnx(
            model,
            export_path,
            input_size=args.input_size,
            verify=args.verify,
            **export_verify_kwargs,
        )

    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    logger.info(f"Done. Saved to {export_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
