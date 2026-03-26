"""
Evaluate depth estimation models on the DIODE val set (771 images).

Usage
-----
# Evaluate one model — auto-downloads val set (~2.6 GB) on first run
python examples/eval_diode.py

# Specific model
python examples/eval_diode.py --model depth-pro

# Indoors or outdoors only
python examples/eval_diode.py --scene-type indoors
python examples/eval_diode.py --scene-type outdoors

# Compare multiple models
python examples/eval_diode.py --compare

# Quick sanity check on 50 samples
python examples/eval_diode.py --num-samples 50

# Pre-downloaded dataset
python examples/eval_diode.py --dataset-root /data/diode

# Save results to JSON
python examples/eval_diode.py --model depth-anything-v2-vitl --output results.json

Notes
-----
- The DIODE val set (~2.6 GB) is auto-downloaded from the official S3 bucket
  on first use and cached at: ~/.cache/depth_estimation/datasets/diode/
- The train set is ~81 GB and is NOT auto-downloaded unless --split train is
  explicitly used together with --download-train.
- DIODE uses a laser scanner so all valid pixels have dense, accurate depth
  in metres — no alignment is needed for metric models.
- Relative-depth models are aligned per-sample (least-squares scale+shift)
  before metric computation.
- Max depth is set to 350 m to accommodate outdoor scenes (~300 m range).
  Pass --max-depth to change this.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Models to compare when --compare is passed ──────────────────────────────
COMPARE_MODELS = [
    "depth-anything-v2-vits",
    "depth-anything-v2-vitb",
    "depth-anything-v2-vitl",
    "zoedepth",
    "depth-pro",
    "midas-dpt-large",
]

# ── Default single model ─────────────────────────────────────────────────────
DEFAULT_MODEL = "depth-anything-v2-vitb"


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate depth models on the DIODE val set.",
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
        "--scene-type",
        default="all",
        choices=["indoors", "outdoors", "all"],
        help="Scene subset to evaluate (default: all).",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Dataset split (default: val). The test set has no public GT.",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        metavar="DIR",
        help="Path to DIODE root (containing val/ and/or train/ dirs). "
             "Auto-downloads to ~/.cache/... if not provided.",
    )
    p.add_argument(
        "--download-train",
        action="store_true",
        help="Allow auto-downloading the train split (~81 GB). "
             "Only relevant when --split train is used.",
    )
    p.add_argument(
        "--max-depth",
        type=float,
        default=350.0,
        metavar="M",
        help="Maximum valid depth in metres (default: 350.0). "
             "Use 10.0 for indoors-only evaluation.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N samples (for quick checks).",
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
        help="DataLoader worker processes (default: 4).",
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
        help="Save results to a JSON file.",
    )
    return p.parse_args()


def _scene_counts(scene_type: str) -> str:
    counts = {"indoors": "325", "outdoors": "446", "all": "771"}
    return counts.get(scene_type, "?")


def _dataset_kwargs(args) -> dict:
    kwargs = dict(scene_type=args.scene_type, max_depth=args.max_depth)
    if args.split == "train" and not args.download_train:
        kwargs["download"] = False
    return kwargs


def evaluate_single(args) -> dict:
    from depth_estimation.evaluation import evaluate

    model_id = args.model
    n_approx = _scene_counts(args.scene_type)

    print(f"\n{'=' * 65}")
    print(f"  DIODE Evaluation")
    print(f"  Model:      {model_id}")
    print(f"  Split:      {args.split}")
    print(f"  Scene type: {args.scene_type} (~{n_approx} images)")
    print(f"  Max depth:  {args.max_depth} m")
    print(f"  Samples:    {args.num_samples or 'all'}")
    print(f"  Align:      {not args.no_align}")
    print(f"{'=' * 65}\n")

    if args.split == "train" and not args.download_train:
        print("NOTE: --split train requested without --download-train.")
        print("      Auto-download is disabled for the train set (~81 GB).")
        print("      Pass --download-train to enable it, or use --dataset-root")
        print("      to point to a pre-downloaded copy.\n")

    t0 = time.time()
    results = evaluate(
        model=model_id,
        dataset="diode",
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        align=not args.no_align,
        num_samples=args.num_samples,
        **_dataset_kwargs(args),
    )
    elapsed = time.time() - t0

    print(f"\n{'─' * 65}")
    print(f"  Results for {model_id}  (DIODE {args.split} / {args.scene_type})")
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
    results["split"] = args.split
    results["scene_type"] = args.scene_type
    results["elapsed_s"] = round(elapsed, 2)
    return results


def evaluate_compare(args) -> dict:
    from depth_estimation.evaluation import compare

    n_approx = _scene_counts(args.scene_type)
    print(f"\n{'=' * 65}")
    print(f"  DIODE — Model Comparison")
    print(f"  Split:      {args.split}")
    print(f"  Scene type: {args.scene_type} (~{n_approx} images)")
    print(f"  Max depth:  {args.max_depth} m")
    print(f"  Models:     {len(COMPARE_MODELS)}")
    print(f"  Samples:    {args.num_samples or 'all'}")
    print(f"  Align:      {not args.no_align}")
    print(f"{'=' * 65}\n")

    t0 = time.time()
    results = compare(
        models=COMPARE_MODELS,
        dataset="diode",
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        align=not args.no_align,
        num_samples=args.num_samples,
        print_table=True,
        **_dataset_kwargs(args),
    )
    elapsed = time.time() - t0
    print(f"Total wall time: {elapsed:.1f}s")
    return results


def save_results(results: dict, path: str) -> None:
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

    if args.compare:
        results = evaluate_compare(args)
    else:
        results = evaluate_single(args)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
