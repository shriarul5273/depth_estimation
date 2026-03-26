"""
Evaluate depth estimation models on the NYU Depth V2 test set (654 images).

Usage
-----
# Evaluate one model (default)
python examples/eval_nyu.py

# Evaluate a specific model
python examples/eval_nyu.py --model depth-anything-v2-vitb

# Compare multiple models
python examples/eval_nyu.py --compare

# Quick sanity check on 50 samples
python examples/eval_nyu.py --model depth-anything-v2-vitb --num-samples 50

# Use a pre-downloaded dataset instead of auto-downloading
python examples/eval_nyu.py --model depth-anything-v2-vitb --dataset-root /data/nyu

# Save results to JSON
python examples/eval_nyu.py --model depth-anything-v2-vitb --output results.json

Notes
-----
- The NYU Depth V2 labeled set is auto-downloaded on first use (~2.8 GB).
  Stored at: ~/.cache/depth_estimation/datasets/nyu_depth_v2/
- Requires h5py: pip install "depth-estimation[data]"
- Relative-depth models (Depth Anything v1/v2, MiDaS, etc.) are automatically
  aligned to ground-truth scale per sample using least-squares before metrics
  are computed.
- Metric-depth models (ZoeDepth, DepthPro, Depth Anything v3 metric) are
  evaluated without alignment.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Models to compare when --compare is passed ──────────────────────────────
COMPARE_MODELS = [
    "depth-anything-v2-vits",
    # "depth-anything-v2-vitb",
    "depth-anything-v2-vitl",
    # "zoedepth",
    # "depth-pro",
    "midas-dpt-large",
]

# ── Default single model ─────────────────────────────────────────────────────
DEFAULT_MODEL = "depth-anything-v2-vits"


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate depth models on the NYU Depth V2 test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model variant ID to evaluate (default: {DEFAULT_MODEL}). "
             "Ignored when --compare is used.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate all models in COMPARE_MODELS and print a comparison table.",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        metavar="DIR",
        help="Path to NYU Depth V2 root directory (containing .mat files). "
             "Auto-downloads to ~/.cache/... if not provided.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N test samples (for quick checks).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="B",
        help="Images per forward pass (default: 1).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="W",
        help="DataLoader worker processes (default: 4). Use 0 on Windows.",
    )
    p.add_argument(
        "--device",
        default=None,
        help='Device: "cuda", "cpu", or "mps". Auto-detected if not set.',
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="Disable per-sample least-squares alignment for relative models.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Save results to a JSON file (e.g. results.json).",
    )
    return p.parse_args()


def _fmt_metrics(metrics: dict) -> str:
    """One-line summary of the key metrics."""
    return (
        f"AbsRel={metrics['abs_rel']:.4f}  "
        f"RMSE={metrics['rmse']:.4f}  "
        f"δ₁={metrics['delta1']:.4f}  "
        f"δ₂={metrics['delta2']:.4f}  "
        f"n={metrics.get('n_samples', '?')}"
    )


def evaluate_single(args) -> dict:
    """Run evaluation for a single model and print results."""
    from depth_estimation.evaluation import evaluate

    model_id = args.model
    print(f"\n{'=' * 65}")
    print(f"  NYU Depth V2 Evaluation")
    print(f"  Model:   {model_id}")
    print(f"  Samples: {args.num_samples or 'all 654'}")
    print(f"  Align:   {not args.no_align}")
    print(f"{'=' * 65}\n")

    t0 = time.time()
    results = evaluate(
        model=model_id,
        dataset="nyu_depth_v2",
        split="test",
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        align=not args.no_align,
        num_samples=args.num_samples,
    )
    elapsed = time.time() - t0

    print(f"\n{'─' * 65}")
    print(f"  Results for {model_id}")
    print(f"{'─' * 65}")
    print(f"  {'Metric':<14}  {'Value':>10}  {'Direction'}")
    print(f"  {'──────':<14}  {'─────':>10}  {'─────────'}")
    for metric, direction in [
        ("abs_rel",  "lower ↓"),
        ("sq_rel",   "lower ↓"),
        ("rmse",     "lower ↓"),
        ("rmse_log", "lower ↓"),
        ("delta1",   "higher ↑"),
        ("delta2",   "higher ↑"),
        ("delta3",   "higher ↑"),
    ]:
        print(f"  {metric:<14}  {results[metric]:>10.4f}  {direction}")
    print(f"{'─' * 65}")
    print(f"  Samples evaluated : {results.get('n_samples', '?')}")
    print(f"  Total wall time   : {elapsed:.1f}s")
    print(f"{'─' * 65}\n")

    results["model_id"] = model_id
    results["elapsed_s"] = round(elapsed, 2)
    return results


def evaluate_compare(args) -> dict:
    """Compare multiple models and print a table."""
    from depth_estimation.evaluation import compare

    print(f"\n{'=' * 65}")
    print(f"  NYU Depth V2 — Model Comparison")
    print(f"  Models:  {len(COMPARE_MODELS)}")
    print(f"  Samples: {args.num_samples or 'all 654'}")
    print(f"  Align:   {not args.no_align}")
    print(f"{'=' * 65}\n")

    t0 = time.time()
    results = compare(
        models=COMPARE_MODELS,
        dataset="nyu_depth_v2",
        split="test",
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        align=not args.no_align,
        num_samples=args.num_samples,
        print_table=True,
    )
    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed:.1f}s")
    return results


def save_results(results: dict, path: str) -> None:
    """Write results dict to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")


def main():
    args = parse_args()

    try:
        import depth_estimation  # noqa: F401
    except ImportError:
        print("ERROR: depth_estimation package not found.")
        print("Install it with:  pip install -e .")
        sys.exit(1)

    try:
        import h5py  # noqa: F401
    except ImportError:
        print("ERROR: h5py is required to read the NYU Depth V2 .mat file.")
        print("Install it with:  pip install h5py")
        print("  or:             pip install 'depth-estimation[data]'")
        sys.exit(1)

    if args.compare:
        results = evaluate_compare(args)
    else:
        results = evaluate_single(args)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
