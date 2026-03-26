"""
Evaluate depth estimation models on the KITTI Eigen split (697 test images).

Usage
-----
# Evaluate one model (dataset root required)
python examples/eval_kitti.py --dataset-root /data/kitti

# Specific model
python examples/eval_kitti.py --model zoedepth --dataset-root /data/kitti

# Compare multiple models
python examples/eval_kitti.py --compare --dataset-root /data/kitti

# Quick sanity check on 50 samples
python examples/eval_kitti.py --dataset-root /data/kitti --num-samples 50

# Save results to JSON
python examples/eval_kitti.py --model depth-pro --dataset-root /data/kitti --output results.json

Notes
-----
- KITTI requires free registration and manual download — files cannot be
  fetched automatically.

  Download instructions:
    1. Register at https://www.cvlibs.net/datasets/kitti/index.php
    2. Download raw data (city / residential / road):
       https://www.cvlibs.net/datasets/kitti/raw_data.php
    3. Download improved ground-truth depth (Garg/Eigen dense GT):
       https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
    4. Extract data_depth_annotated.zip into your dataset root.

  The Eigen split file lists are downloaded automatically from the BTS repo
  on first use and cached in your dataset root.

- Expected layout:
    /data/kitti/
      2011_09_26/
        2011_09_26_drive_0001_sync/
          image_02/data/0000000005.png
      data_depth_annotated/
        train/ ...
        val/   ...
      eigen_train_files_with_gt.txt  ← auto-downloaded
      eigen_val_files_with_gt.txt    ← auto-downloaded
      eigen_test_files_with_gt.txt   ← auto-downloaded

- Ground-truth is sparse projected LiDAR (~5 % of pixels per image).
  valid_mask covers only pixels with a LiDAR return.

- Relative-depth models are aligned per-sample (least-squares scale+shift)
  before metric computation. Metric models are evaluated as-is.

- Standard evaluation cap: max_depth=80 m.
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
        description="Evaluate depth models on the KITTI Eigen test split.",
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
        required=True,
        metavar="DIR",
        help="Path to KITTI raw data root. Required — KITTI cannot be "
             "auto-downloaded. See the module docstring for layout details.",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test).",
    )
    p.add_argument(
        "--filenames",
        default=None,
        metavar="FILE",
        help="Path to a custom split .txt file. Auto-resolved from "
             "--dataset-root if not provided.",
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


def _extra_kwargs(args) -> dict:
    """Build dataset_kwargs to forward to load_dataset / KITTIEigenDataset."""
    kwargs = {}
    if args.filenames:
        kwargs["filenames"] = args.filenames
    return kwargs


def evaluate_single(args) -> dict:
    from depth_estimation.evaluation import evaluate

    n_test = {"train": "~23 000", "val": "~4 000", "test": "697"}
    model_id = args.model

    print(f"\n{'=' * 65}")
    print(f"  KITTI Eigen Evaluation")
    print(f"  Model:   {model_id}")
    print(f"  Split:   {args.split} ({n_test.get(args.split, '?')} images)")
    print(f"  Samples: {args.num_samples or 'all'}")
    print(f"  Align:   {not args.no_align}")
    print(f"  Root:    {args.dataset_root}")
    print(f"{'=' * 65}\n")

    t0 = time.time()
    results = evaluate(
        model=model_id,
        dataset="kitti_eigen",
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        align=not args.no_align,
        num_samples=args.num_samples,
        **_extra_kwargs(args),
    )
    elapsed = time.time() - t0

    print(f"\n{'─' * 65}")
    print(f"  Results for {model_id}  (KITTI Eigen {args.split})")
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
    results["elapsed_s"] = round(elapsed, 2)
    return results


def evaluate_compare(args) -> dict:
    from depth_estimation.evaluation import compare

    n_test = {"train": "~23 000", "val": "~4 000", "test": "697"}
    print(f"\n{'=' * 65}")
    print(f"  KITTI Eigen — Model Comparison")
    print(f"  Split:   {args.split} ({n_test.get(args.split, '?')} images)")
    print(f"  Models:  {len(COMPARE_MODELS)}")
    print(f"  Samples: {args.num_samples or 'all'}")
    print(f"  Align:   {not args.no_align}")
    print(f"  Root:    {args.dataset_root}")
    print(f"{'=' * 65}\n")

    t0 = time.time()
    results = compare(
        models=COMPARE_MODELS,
        dataset="kitti_eigen",
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        align=not args.no_align,
        num_samples=args.num_samples,
        print_table=True,
        **_extra_kwargs(args),
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

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"ERROR: --dataset-root does not exist: {root}")
        print()
        print("KITTI requires manual download. Steps:")
        print("  1. Register at https://www.cvlibs.net/datasets/kitti/index.php")
        print("  2. Download raw data sequences from:")
        print("     https://www.cvlibs.net/datasets/kitti/raw_data.php")
        print("  3. Download improved GT depth:")
        print("     https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip")
        print("  4. Extract into your dataset root and pass it via --dataset-root.")
        sys.exit(1)

    if args.compare:
        results = evaluate_compare(args)
    else:
        results = evaluate_single(args)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
