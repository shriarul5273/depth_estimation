"""
Fine-tune Depth Anything V2 Small on NYU Depth V2.

Usage
-----
# Decoder-only fine-tuning (fastest, recommended to start):
python examples/train_depth_anything.py

# Full fine-tuning with backbone warm-up:
python examples/train_depth_anything.py --full-finetune

# Resume a previous run:
python examples/train_depth_anything.py --resume ./checkpoints/depth_anything_nyu/checkpoint_epoch_0004

# Custom settings:
python examples/train_depth_anything.py \\
    --model depth-anything-v2-vitb \\
    --epochs 30 \\
    --batch-size 4 \\
    --lr 5e-5 \\
    --output ./my_checkpoints \\
    --device cuda
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Depth Anything V2 on NYU Depth V2"
    )
    p.add_argument(
        "--model",
        default="depth-anything-v2-vits",
        help="Model variant to fine-tune (default: depth-anything-v2-vits)",
    )
    p.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs (default: 25)"
    )
    p.add_argument(
        "--batch-size", type=int, default=8, help="Batch size per GPU (default: 8)"
    )
    p.add_argument(
        "--lr", type=float, default=5e-5, help="Base learning rate (default: 5e-5)"
    )
    p.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="Backbone LR multiplier relative to base LR (default: 0.1)",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Spatial size of training crops (default: 518)",
    )
    p.add_argument(
        "--output",
        default="./checkpoints/depth_anything_nyu",
        help="Output directory for checkpoints (default: ./checkpoints/depth_anything_nyu)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device to use: cuda, cpu, mps. Auto-detected when omitted.",
    )
    p.add_argument(
        "--full-finetune",
        action="store_true",
        help="Full fine-tuning: backbone trainable from epoch 0 (no warm-up). "
        "Default: decoder-only with backbone warm-up for 5 epochs.",
    )
    p.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable AMP mixed precision (CUDA only).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training from a saved checkpoint directory.",
    )
    p.add_argument(
        "--nyu-root",
        default=None,
        metavar="PATH",
        help="Local NYU Depth V2 root (auto-downloaded to cache when omitted).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from depth_estimation.models.depth_anything_v2 import DepthAnythingV2Model
    from depth_estimation import (
        DepthTrainer,
        DepthTrainingArguments,
        load_dataset,
    )
    from depth_estimation.data.transforms import get_train_transforms, get_val_transforms

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
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {args.model}")
    model = DepthAnythingV2Model.from_pretrained(
        args.model, device=device, for_training=True
    )

    freeze_backbone_epochs = 0
    if not args.full_finetune:
        # Freeze backbone; warm up decoder for 5 epochs, then unfreeze
        freeze_backbone_epochs = 5
        model.freeze_backbone()
        logger.info(
            f"Backbone frozen. Trainable parameters: {model._count_trainable():,}"
        )
    else:
        logger.info(
            f"Full fine-tuning. Trainable parameters: {model._count_trainable():,}"
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading NYU Depth V2 datasets…")

    dataset_kwargs = {}
    if args.nyu_root is not None:
        dataset_kwargs["root"] = args.nyu_root

    train_ds = load_dataset(
        "nyu_depth_v2",
        split="train",
        transform=get_train_transforms(args.input_size),
        **dataset_kwargs,
    )
    val_ds = load_dataset(
        "nyu_depth_v2",
        split="test",
        transform=get_val_transforms(args.input_size),
        **dataset_kwargs,
    )

    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    training_args = DepthTrainingArguments(
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        backbone_lr_scale=args.backbone_lr_scale,
        batch_size=args.batch_size,
        freeze_backbone_epochs=freeze_backbone_epochs,
        lr_scheduler="cosine",
        warmup_epochs=2,
        mixed_precision=args.mixed_precision,
        dataloader_num_workers=args.num_workers,
        eval_metric="abs_rel",
        lower_is_better=True,
        save_every_n_epochs=5,
        eval_every_n_epochs=1,
        log_every_n_steps=50,
    )

    # Save args for reproducibility
    import os
    os.makedirs(args.output, exist_ok=True)
    training_args.to_json(f"{args.output}/training_args.json")
    logger.info(f"Training arguments saved to {args.output}/training_args.json")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = DepthTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    logger.info("Starting training…")
    trainer.train(resume_from=args.resume)

    logger.info(
        f"\nTraining complete. Checkpoints saved to: {args.output}\n"
        f"  Best model:  {args.output}/best_model/\n"
        f"  Final model: {args.output}/final/\n"
        f"\nTo evaluate the fine-tuned model:\n"
        f"  python examples/test_finetuned.py --checkpoint {args.output}/best_model "
        f"--model {args.model}"
    )


if __name__ == "__main__":
    main()
