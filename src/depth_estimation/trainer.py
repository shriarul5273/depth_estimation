"""
DepthTrainer — Full training and fine-tuning loop for depth estimation models.

Orchestrates: data loading, optimisation, LR scheduling, loss computation,
evaluation, and checkpointing.  Does NOT depend on any external training
framework (no PyTorch Lightning, no HF Trainer).

Example::

    from depth_estimation import DepthTrainer, DepthTrainingArguments, load_dataset
    from depth_estimation.data.transforms import get_train_transforms, get_val_transforms

    model = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vitb", for_training=True)
    model.freeze_backbone()

    train_ds = load_dataset("nyu_depth_v2", split="train", transform=get_train_transforms(518))
    val_ds   = load_dataset("nyu_depth_v2", split="test",  transform=get_val_transforms(518))

    args = DepthTrainingArguments(output_dir="./checkpoints", num_epochs=25, batch_size=8)
    trainer = DepthTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
"""

import dataclasses
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CombinedDepthLoss
from .training_args import DepthTrainingArguments

logger = logging.getLogger(__name__)


class DepthTrainer:
    """Trains and fine-tunes depth estimation models.

    Args:
        model:         A :class:`~depth_estimation.modeling_utils.BaseDepthModel`
                       instance. Should be in train mode
                       (use ``from_pretrained(..., for_training=True)``).
        args:          :class:`~depth_estimation.training_args.DepthTrainingArguments`.
        train_dataset: Dataset returning ``{"pixel_values", "depth_map", "valid_mask"}``.
        eval_dataset:  Validation dataset. Pass ``None`` to skip evaluation.
        processor:     Optional processor used only for logging purposes.
    """

    def __init__(
        self,
        model: nn.Module,
        args: DepthTrainingArguments,
        train_dataset,
        eval_dataset=None,
        processor=None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processor = processor

        # Derive device from model parameters
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            raise RuntimeError(
                "Model has no parameters. Ensure the model is fully initialised "
                "before passing it to DepthTrainer."
            )

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self._is_plateau_scheduler = False
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.loss_fn: Optional[CombinedDepthLoss] = None
        self.train_loader: Optional[DataLoader] = None
        self.eval_loader: Optional[DataLoader] = None

        self.best_metric = float("inf") if args.lower_is_better else float("-inf")
        self.global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run the full training loop.

        Args:
            resume_from: Optional path to a checkpoint directory produced by
                :meth:`_save_checkpoint`. Training resumes from the next epoch.
        """
        self._setup()
        start_epoch = 0

        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            logger.info(f"Resuming training from epoch {start_epoch}")

        # Backbone freeze schedule
        if self.args.freeze_backbone_epochs > 0 and start_epoch == 0:
            logger.info(
                f"Freezing backbone for first {self.args.freeze_backbone_epochs} epochs."
            )
            self.model.freeze_backbone()

        epoch_bar = tqdm(
            range(start_epoch, self.args.num_epochs),
            desc="Epochs",
            unit="epoch",
            initial=start_epoch,
            total=self.args.num_epochs,
        )
        for epoch in epoch_bar:
            # Unfreeze backbone after the warm-up period
            if (
                self.args.freeze_backbone_epochs > 0
                and epoch == self.args.freeze_backbone_epochs
            ):
                logger.info(f"Epoch {epoch}: unfreezing backbone.")
                self.model.unfreeze_backbone()

            train_metrics = self._train_epoch(epoch)
            if not self._is_plateau_scheduler and self.scheduler is not None:
                self.scheduler.step()
            logger.info(f"Epoch {epoch} train: {train_metrics}")
            epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in train_metrics.items()})

            if (epoch + 1) % self.args.eval_every_n_epochs == 0 and self.eval_dataset is not None:
                eval_metrics = self._eval_epoch(epoch)
                logger.info(f"Epoch {epoch} eval:  {eval_metrics}")
                epoch_bar.set_postfix({f"val_{k}": f"{v:.4f}" for k, v in eval_metrics.items()})

                if self._is_plateau_scheduler and self.scheduler is not None:
                    self.scheduler.step(eval_metrics[self.args.eval_metric])

            if (epoch + 1) % self.args.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(self.args.num_epochs - 1, tag="final")
        logger.info("Training complete.")

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load model weights and training state to resume training.

        Args:
            checkpoint_dir: Path to a checkpoint directory written by
                :meth:`_save_checkpoint`.

        Returns:
            The epoch number to resume from (= saved epoch + 1).
        """
        ckpt_dir = Path(checkpoint_dir)
        model_path = ckpt_dir / "model.pt"
        state_path = ckpt_dir / "training_state.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"No model.pt found in {ckpt_dir}")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        logger.info(f"Loaded model weights from {model_path}")

        if state_path.exists() and self.optimizer is not None:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.best_metric = state.get("best_metric", self.best_metric)
            return state.get("epoch", 0) + 1

        return 0

    # ------------------------------------------------------------------
    # Private: setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Initialise all training components."""
        self._setup_seed()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_scheduler(self.args.num_epochs)
        self._build_loss()
        if self.args.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

    def _setup_seed(self) -> None:
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_dataloaders(self) -> None:
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        if self.eval_dataset is not None:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

    def _build_optimizer(self) -> None:
        groups = self.model.get_parameter_groups(self.args.backbone_lr_scale)
        opt_groups = [
            {
                "params": g["params"],
                "lr": self.args.learning_rate * g["lr_scale"],
                "weight_decay": self.args.weight_decay,
            }
            for g in groups
            if len(g["params"]) > 0
        ]
        if not opt_groups:
            raise RuntimeError(
                "No trainable parameters found. Call model.train() or "
                "unfreeze_backbone() before creating the trainer."
            )
        self.optimizer = torch.optim.AdamW(opt_groups)

    def _build_scheduler(self, num_epochs: int) -> None:
        warmup_epochs = self.args.warmup_epochs
        after_warmup_epochs = max(1, num_epochs - warmup_epochs)

        if self.args.lr_scheduler == "plateau":
            # ReduceLROnPlateau is stepped with the eval metric, not per epoch
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min" if self.args.lower_is_better else "max",
                factor=0.5,
                patience=3,
            )
            self._is_plateau_scheduler = True
            return

        if self.args.lr_scheduler == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=after_warmup_epochs,
                eta_min=1e-7,
            )
        elif self.args.lr_scheduler == "linear":
            main_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1e-7,
                total_iters=after_warmup_epochs,
            )
        elif self.args.lr_scheduler == "step":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.args.lr_scheduler!r}")

        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = main_scheduler

        self._is_plateau_scheduler = False

    def _build_loss(self) -> None:
        self.loss_fn = CombinedDepthLoss(
            si_weight=self.args.si_loss_weight,
            grad_weight=self.args.grad_loss_weight,
            lam=self.args.si_lam,
        )

    # ------------------------------------------------------------------
    # Private: training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        running = {"loss": 0.0, "si_loss": 0.0, "grad_loss": 0.0}
        n_steps = 0

        batch_bar = tqdm(
            self.train_loader,
            desc=f"Train {epoch}",
            unit="batch",
            leave=False,
        )
        for batch in batch_bar:
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            depth_map    = batch["depth_map"].to(self.device, non_blocking=True)
            valid_mask   = batch["valid_mask"].to(self.device, non_blocking=True)

            # Squeeze channel dim: (B,1,H,W) → (B,H,W)
            depth_map  = depth_map.squeeze(1)
            valid_mask = valid_mask.squeeze(1)

            self.optimizer.zero_grad()

            use_amp = self.args.mixed_precision and self.device.type == "cuda"
            with torch.autocast(device_type=device_type, enabled=use_amp):
                pred = self.model(pixel_values)
                if pred.dim() == 4:
                    pred = pred.squeeze(1)
                losses = self.loss_fn(pred, depth_map, valid_mask)

            if self.scaler is not None:
                self.scaler.scale(losses["loss"]).backward()
                if self.args.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.gradient_clip_val
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["loss"].backward()
                if self.args.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.gradient_clip_val
                    )
                self.optimizer.step()

            for k in running:
                running[k] += losses[k].item()
            n_steps += 1
            self.global_step += 1

            avg_loss = running["loss"] / n_steps
            batch_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

            if self.global_step % self.args.log_every_n_steps == 0:
                avg_si   = running["si_loss"] / n_steps
                avg_grad = running["grad_loss"] / n_steps
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {self.global_step} | epoch {epoch} | "
                    f"loss={avg_loss:.4f} si={avg_si:.4f} grad={avg_grad:.4f} lr={lr:.2e}"
                )

        n = max(1, n_steps)
        return {k: v / n for k, v in running.items()}

    # ------------------------------------------------------------------
    # Private: evaluation epoch
    # ------------------------------------------------------------------

    def _eval_epoch(self, epoch: int) -> dict:
        from .evaluation.metrics import Evaluator

        self.model.eval()
        ev = Evaluator()

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc=f"Eval  {epoch}", unit="batch", leave=False):
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                depth_map    = batch["depth_map"].to(self.device, non_blocking=True)
                valid_mask   = batch["valid_mask"].to(self.device, non_blocking=True)

                pred = self.model(pixel_values)
                # Normalise shapes to (B, H, W)
                if pred.dim() == 4:
                    pred = pred.squeeze(1)
                if depth_map.dim() == 4:
                    depth_map = depth_map.squeeze(1)
                if valid_mask.dim() == 4:
                    valid_mask = valid_mask.squeeze(1)

                ev.update(pred, depth_map, valid_mask)

        metrics = ev.compute()

        # Track best checkpoint
        metric_val = metrics[self.args.eval_metric]
        is_better = (
            metric_val < self.best_metric
            if self.args.lower_is_better
            else metric_val > self.best_metric
        )
        if is_better:
            self.best_metric = metric_val
            self._save_checkpoint(epoch, tag="best_model")
            logger.info(
                f"Epoch {epoch}: new best {self.args.eval_metric}={metric_val:.4f} "
                f"— saved best_model checkpoint."
            )

        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    # Private: checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: Optional[str] = None) -> None:
        """Save model weights and training state.

        Args:
            epoch: Current epoch number (0-indexed).
            tag:   Checkpoint directory name. Defaults to
                   ``checkpoint_epoch_NNNN``.
        """
        name = tag or f"checkpoint_epoch_{epoch:04d}"
        save_dir = Path(self.args.output_dir) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Model weights
        torch.save(self.model.state_dict(), save_dir / "model.pt")

        # Training state (for resume)
        state: dict = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "args": dataclasses.asdict(self.args),
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None and not self._is_plateau_scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, save_dir / "training_state.pt")

        # Config JSON
        if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict"):
            config_dict = self.model.config.to_dict()
            (save_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

        logger.info(f"Checkpoint saved to {save_dir}")
