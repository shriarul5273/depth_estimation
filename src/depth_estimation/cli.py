"""
Command-line interface for depth_estimation.

Entry point: depth-estimate
Subcommands: predict, list-models, info, benchmark
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _save_result(result, output_path: Path, fmt: str, source_path: Optional[Path] = None) -> None:
    """Save a DepthOutput to disk in the requested format."""
    import cv2
    import numpy as np

    if fmt in ("png", "both"):
        if result.colored_depth is not None:
            png_path = output_path.with_suffix(".png")
            bgr = cv2.cvtColor(result.colored_depth, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(png_path), bgr)

    if fmt in ("npy", "both"):
        npy_path = output_path.with_suffix(".npy")
        np.save(str(npy_path), result.depth)


def _resolve_output_path(source: Path, output_arg: Optional[str], output_dir: Optional[str]) -> Path:
    """Work out where to write the output file."""
    if output_arg:
        return Path(output_arg)
    base = output_dir or source.parent
    return Path(base) / (source.stem + "_depth")


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a plain-text table with aligned columns."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def _collect_images(source: str) -> List[Path]:
    """Expand source (file, directory, or glob) to a list of image paths."""
    import glob as _glob

    # Glob pattern
    if any(c in source for c in ("*", "?", "[")):
        paths = [Path(p) for p in _glob.glob(source, recursive=True)]
        return sorted(p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS)

    p = Path(source)

    if p.is_dir():
        return sorted(
            f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )

    if p.is_file():
        return [p]

    _die(f"Source not found: {source}")


def _run_predict(args: argparse.Namespace) -> None:
    source = args.source
    model_id = args.model
    colormap = args.colormap
    fmt = args.format
    batch_size = args.batch_size
    device = args.device
    quiet = args.quiet
    output_arg = args.output
    output_dir = args.output_dir

    # ---- Video branch ----
    src_path = Path(source)
    if src_path.is_file() and src_path.suffix.lower() in VIDEO_EXTENSIONS:
        _run_predict_video(src_path, model_id, output_arg, colormap, fmt, device, quiet)
        return

    # ---- Image branch ----
    image_paths = _collect_images(source)
    if not image_paths:
        _die(f"No images found at: {source}")

    if not quiet:
        print(f"Loading model: {model_id}")

    from depth_estimation import pipeline as _pipeline
    pipe = _pipeline("depth-estimation", model=model_id, device=device)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Process in batches
    total = len(image_paths)
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_inputs = [str(p) for p in batch_paths]

        results = pipe(batch_inputs, batch_size=batch_size, colorize=(fmt in ("png", "both")), colormap=colormap)

        if not isinstance(results, list):
            results = [results]

        for path, result in zip(batch_paths, results):
            out_path = _resolve_output_path(path, output_arg if total == 1 else None, output_dir)
            _save_result(result, out_path, fmt, source_path=path)
            if not quiet:
                saved = out_path.with_suffix(".png") if fmt in ("png", "both") else out_path.with_suffix(".npy")
                print(f"  [{i + batch_paths.index(path) + 1}/{total}] {path.name} -> {saved}")


def _run_predict_video(
    src_path: Path,
    model_id: str,
    output_arg: Optional[str],
    colormap: str,
    fmt: str,
    device: Optional[str],
    quiet: bool,
) -> None:
    import cv2
    import numpy as np

    out_video_path = Path(output_arg) if output_arg else src_path.parent / (src_path.stem + "_depth.mp4")

    if not quiet:
        print(f"Loading model: {model_id}")

    from depth_estimation import pipeline as _pipeline
    pipe = _pipeline("depth-estimation", model=model_id, device=device)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        _die(f"Cannot open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Side-by-side doubles width
    out_w = width * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (out_w, height))

    if not quiet:
        print(f"Processing video: {src_path.name}  ({total_frames} frames)")

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = pipe(frame_rgb, colorize=True, colormap=colormap)

        if result.colored_depth is not None:
            depth_bgr = cv2.cvtColor(result.colored_depth, cv2.COLOR_RGB2BGR)
            depth_resized = cv2.resize(depth_bgr, (width, height))
        else:
            depth_arr = (result.depth * 255).astype(np.uint8)
            depth_bgr = cv2.applyColorMap(depth_arr, cv2.COLORMAP_INFERNO)
            depth_resized = cv2.resize(depth_bgr, (width, height))

        combined = np.concatenate([frame_bgr, depth_resized], axis=1)
        writer.write(combined)

        frame_idx += 1
        if not quiet and frame_idx % 10 == 0:
            pct = int(frame_idx / max(total_frames, 1) * 100)
            print(f"  {frame_idx}/{total_frames} frames  ({pct}%)", end="\r")

    cap.release()
    writer.release()
    if not quiet:
        print(f"\nSaved: {out_video_path}")


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------

def _run_list_models(args: argparse.Namespace) -> None:
    # Import triggers all model self-registrations
    from depth_estimation import MODEL_REGISTRY
    import depth_estimation  # noqa: F401

    variants = MODEL_REGISTRY.list_variants()

    if args.json:
        rows = []
        for v in sorted(variants):
            model_type = MODEL_REGISTRY.resolve_model_type(v)
            config_cls = MODEL_REGISTRY.get_config_cls(v)
            cfg = config_cls() if not hasattr(config_cls, "from_variant") else config_cls.from_variant(v)
            rows.append({
                "variant": v,
                "model_type": model_type,
                "is_metric": getattr(cfg, "is_metric", False),
                "backbone": getattr(cfg, "backbone", "—"),
            })
        print(json.dumps(rows, indent=2))
        return

    headers = ["Variant", "Model Type", "Type", "Backbone"]
    rows = []
    for v in sorted(variants):
        model_type = MODEL_REGISTRY.resolve_model_type(v)
        config_cls = MODEL_REGISTRY.get_config_cls(v)
        try:
            cfg = config_cls.from_variant(v) if hasattr(config_cls, "from_variant") else config_cls()
        except Exception:
            cfg = config_cls()
        depth_type = "metric" if getattr(cfg, "is_metric", False) else "relative"
        backbone = getattr(cfg, "backbone", "—")
        rows.append([v, model_type, depth_type, backbone])

    _print_table(headers, rows)
    print(f"\n{len(variants)} variants across {len(MODEL_REGISTRY.list_model_types())} model families.")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def _run_info(args: argparse.Namespace) -> None:
    from depth_estimation import MODEL_REGISTRY
    import depth_estimation  # noqa: F401

    model_id = args.model_id

    try:
        model_type = MODEL_REGISTRY.resolve_model_type(model_id)
    except ValueError as e:
        _die(str(e))

    config_cls = MODEL_REGISTRY.get_config_cls(model_id)
    try:
        cfg = config_cls.from_variant(model_id) if hasattr(config_cls, "from_variant") else config_cls()
    except Exception:
        cfg = config_cls()

    info = {
        "variant": model_id,
        "model_type": model_type,
        "backbone": getattr(cfg, "backbone", "—"),
        "depth_type": "metric" if getattr(cfg, "is_metric", False) else "relative",
        "input_size": getattr(cfg, "input_size", "—"),
        "patch_size": getattr(cfg, "patch_size", "—"),
        "embed_dim": getattr(cfg, "embed_dim", "—"),
        "num_heads": getattr(cfg, "num_heads", "—"),
        "num_layers": getattr(cfg, "num_layers", "—"),
        "max_depth": getattr(cfg, "max_depth", None),
        "min_depth": getattr(cfg, "min_depth", None),
    }

    if args.json:
        print(json.dumps(info, indent=2))
        return

    print(f"\nModel: {model_id}")
    print("-" * 40)
    for k, v in info.items():
        if k == "variant":
            continue
        label = k.replace("_", " ").title()
        if v is not None:
            print(f"  {label:<16} {v}")
    print()


# ---------------------------------------------------------------------------
# benchmark (stub)
# ---------------------------------------------------------------------------

def _run_benchmark(args: argparse.Namespace) -> None:
    print("benchmark: not yet implemented (depends on the evaluation suite).")
    print("Coming in v0.1.1.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depth-estimate",
        description="depth_estimation CLI — run depth estimation models from the command line.",
    )
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, mps (auto-detected if omitted).")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")

    sub = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    sub.required = True

    # ---- predict ----
    p_predict = sub.add_parser(
        "predict",
        help="Run depth estimation on an image, directory, glob, or video.",
    )
    p_predict.add_argument("source", help="Image path, directory, glob pattern (quoted), or video file.")
    p_predict.add_argument("--model", "-m", required=True, help="Model variant ID (e.g. depth-anything-v2-vitb).")
    p_predict.add_argument("--output", "-o", default=None, help="Output file path (single image / video).")
    p_predict.add_argument("--output-dir", default=None, help="Output directory for batch predictions.")
    p_predict.add_argument(
        "--colormap", default="Spectral_r", help="Matplotlib colormap name (default: Spectral_r)."
    )
    p_predict.add_argument(
        "--format",
        choices=["png", "npy", "both"],
        default="png",
        help="Output format: png (colored), npy (raw float32), or both (default: png).",
    )
    p_predict.add_argument("--batch-size", type=int, default=1, help="Batch size for image processing (default: 1).")

    # ---- list-models ----
    p_list = sub.add_parser("list-models", help="List all available model variants.")
    p_list.add_argument("--json", action="store_true", help="Output as JSON.")

    # ---- info ----
    p_info = sub.add_parser("info", help="Show configuration details for a model variant.")
    p_info.add_argument("model_id", help="Model variant ID (e.g. depth-anything-v2-vitb).")
    p_info.add_argument("--json", action="store_true", help="Output as JSON.")

    # ---- benchmark ----
    sub.add_parser("benchmark", help="(Coming soon) Evaluate a model on a standard dataset.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    dispatch = {
        "predict": _run_predict,
        "list-models": _run_list_models,
        "info": _run_info,
        "benchmark": _run_benchmark,
    }

    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
