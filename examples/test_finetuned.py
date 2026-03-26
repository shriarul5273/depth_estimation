"""
Evaluate a fine-tuned Depth Anything checkpoint and compare it to the base model.

Usage
-----
# Compare fine-tuned vs. base model on NYU Depth V2 (default):
python examples/test_finetuned.py \\
    --checkpoint ./checkpoints/depth_anything_nyu/best_model \\
    --model depth-anything-v2-vits

# Evaluate on a subset:
python examples/test_finetuned.py \\
    --checkpoint ./checkpoints/depth_anything_nyu/best_model \\
    --model depth-anything-v2-vits \\
    --num-samples 100

# Different dataset:
python examples/test_finetuned.py \\
    --checkpoint ./checkpoints/depth_anything_nyu/best_model \\
    --model depth-anything-v2-vits \\
    --dataset diode --scene-type indoors

# Skip base model comparison (fine-tuned only):
python examples/test_finetuned.py \\
    --checkpoint ./checkpoints/depth_anything_nyu/best_model \\
    --model depth-anything-v2-vits \\
    --no-compare
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

METRIC_NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
METRIC_HEADERS = ["AbsRel", "SqRel", "RMSE", "RMSElog", "δ<1.25", "δ<1.25²", "δ<1.25³"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a fine-tuned depth model checkpoint vs. the base model"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        metavar="DIR",
        help="Path to the checkpoint directory (must contain model.pt).",
    )
    p.add_argument(
        "--model",
        default="depth-anything-v2-vits",
        help="Model variant the checkpoint was fine-tuned from (default: depth-anything-v2-vits)",
    )
    p.add_argument(
        "--dataset",
        default="nyu_depth_v2",
        choices=["nyu_depth_v2", "diode", "kitti_eigen"],
        help="Evaluation dataset (default: nyu_depth_v2)",
    )
    p.add_argument(
        "--scene-type",
        default="all",
        choices=["indoors", "outdoors", "all"],
        help="DIODE scene type filter (default: all)",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        metavar="PATH",
        help="Local dataset root (auto-downloaded when omitted for NYU/DIODE).",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Spatial crop size for validation transforms (default: 518)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate on the first N samples (default: entire dataset).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device: cuda, cpu, mps. Auto-detected when omitted.",
    )
    p.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip base model evaluation; show fine-tuned results only.",
    )
    return p.parse_args()


def build_dataset(args):
    from depth_estimation import load_dataset
    from depth_estimation.data.transforms import get_val_transforms

    transform = get_val_transforms(args.input_size)
    kwargs = {"transform": transform}
    if args.dataset_root is not None:
        kwargs["root"] = args.dataset_root

    if args.dataset == "diode":
        kwargs["scene_type"] = args.scene_type
        ds = load_dataset("diode", split="val", **kwargs)
    elif args.dataset == "kitti_eigen":
        if args.dataset_root is None:
            logger.error(
                "KITTI requires a local root. Pass --dataset-root /path/to/kitti"
            )
            sys.exit(1)
        ds = load_dataset("kitti_eigen", split="test", **kwargs)
    else:
        ds = load_dataset("nyu_depth_v2", split="test", **kwargs)

    if args.num_samples is not None:
        from torch.utils.data import Subset
        ds = Subset(ds, range(min(args.num_samples, len(ds))))

    return ds


def evaluate_model(model, dataset, batch_size, num_workers, device):
    """Run evaluation and return a metrics dict."""
    import torch
    from torch.utils.data import DataLoader
    from depth_estimation.evaluation.metrics import Evaluator

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    ev = Evaluator()

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            depth_map    = batch["depth_map"].to(device, non_blocking=True)
            valid_mask   = batch["valid_mask"].to(device, non_blocking=True)

            pred = model(pixel_values)

            # Normalise to (B, H, W)
            if pred.dim() == 4:
                pred = pred.squeeze(1)
            if depth_map.dim() == 4:
                depth_map = depth_map.squeeze(1)
            if valid_mask.dim() == 4:
                valid_mask = valid_mask.squeeze(1)

            ev.update(pred, depth_map, valid_mask)

    return ev.compute()


def print_results_table(rows):
    """Print a formatted comparison table.

    rows: list of (label, metrics_dict)
    """
    col_w = 12
    header = f"{'Model':<28}" + "".join(f"{h:>{col_w}}" for h in METRIC_HEADERS)
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for label, metrics in rows:
        vals = "".join(f"{metrics[k]:>{col_w}.4f}" for k in METRIC_NAMES)
        print(f"{label:<28}{vals}")
    print(sep)
    print()


def main():
    args = parse_args()

    import torch
    from depth_estimation.models.depth_anything_v2 import DepthAnythingV2Model

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = build_dataset(args)
    n = len(dataset)
    logger.info(f"Evaluation samples: {n}")

    # ------------------------------------------------------------------
    # Fine-tuned model
    # ------------------------------------------------------------------
    import os
    from pathlib import Path

    checkpoint_dir = Path(args.checkpoint)
    model_pt = checkpoint_dir / "model.pt"
    if not model_pt.exists():
        logger.error(f"No model.pt found in {checkpoint_dir}")
        sys.exit(1)

    logger.info(f"Loading fine-tuned checkpoint from {checkpoint_dir}")
    finetuned_model = DepthAnythingV2Model.from_pretrained(args.model, device=device)
    finetuned_model.load_state_dict(
        torch.load(model_pt, map_location=device, weights_only=True)
    )

    logger.info("Evaluating fine-tuned model…")
    finetuned_metrics = evaluate_model(
        finetuned_model, dataset, args.batch_size, args.num_workers, device
    )
    logger.info(f"Fine-tuned results: {finetuned_metrics}")

    results = [(f"Fine-tuned ({args.model})", finetuned_metrics)]

    # ------------------------------------------------------------------
    # Base model comparison
    # ------------------------------------------------------------------
    if not args.no_compare:
        logger.info(f"Evaluating base model: {args.model}")
        del finetuned_model  # free memory before loading second model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        base_model = DepthAnythingV2Model.from_pretrained(args.model, device=device)

        logger.info("Evaluating base model…")
        base_metrics = evaluate_model(
            base_model, dataset, args.batch_size, args.num_workers, device
        )
        logger.info(f"Base model results: {base_metrics}")
        results.append((f"Base model ({args.model})", base_metrics))

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    print_results_table(results)

    # Show improvement summary when comparing
    if not args.no_compare and len(results) == 2:
        ft_m, base_m = results[0][1], results[1][1]
        abs_rel_delta = ft_m["abs_rel"] - base_m["abs_rel"]
        delta1_delta  = ft_m["delta1"]  - base_m["delta1"]
        direction = "better" if abs_rel_delta < 0 else "worse"
        print(
            f"AbsRel change: {abs_rel_delta:+.4f} ({direction} after fine-tuning)\n"
            f"δ<1.25 change: {delta1_delta:+.4f}\n"
        )


if __name__ == "__main__":
    main()
