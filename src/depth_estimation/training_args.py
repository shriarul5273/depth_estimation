"""
DepthTrainingArguments — Hyperparameter configuration for training runs.

Serialisable to/from JSON for reproducibility.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DepthTrainingArguments:
    """All hyperparameters for a depth estimation training run.

    Args:
        output_dir:
            Directory to save checkpoints and logs. Created if missing.
        num_epochs:
            Total training epochs. Default 25.
        learning_rate:
            Base learning rate for the decoder. Default 5e-5.
        backbone_lr_scale:
            Multiplier applied to the backbone learning rate relative to
            ``learning_rate``. E.g. 0.1 → backbone LR = lr × 0.1.
            Default 0.1.
        weight_decay:
            L2 regularisation coefficient for AdamW. Default 0.01.
        batch_size:
            Samples per GPU per step. Default 8.
        gradient_clip_val:
            Maximum gradient norm (clipped via clip_grad_norm_).
            Set to 0.0 to disable. Default 1.0.
        freeze_backbone_epochs:
            Number of epochs at the start of training to keep the backbone
            frozen. The backbone is unfrozen after this many epochs (useful
            for warming up the decoder before full fine-tuning). Default 0
            (no freezing).
        lr_scheduler:
            One of ``"cosine"``, ``"linear"``, ``"step"``, ``"plateau"``.
            Default ``"cosine"``.
        warmup_epochs:
            Number of epochs for linear LR warmup. Default 0 (no warmup).
        si_loss_weight:
            Weight for :class:`~depth_estimation.losses.ScaleInvariantLoss`.
            Default 1.0.
        grad_loss_weight:
            Weight for :class:`~depth_estimation.losses.GradientLoss`.
            Default 0.5.
        si_lam:
            Variance weight (lambda) inside ScaleInvariantLoss. Default 0.85.
        save_every_n_epochs:
            Save a checkpoint every N epochs. Default 5.
        eval_every_n_epochs:
            Run validation every N epochs. Default 1.
        log_every_n_steps:
            Log training metrics every N gradient steps. Default 50.
        eval_metric:
            Primary metric for selecting the best checkpoint.
            One of ``"abs_rel"``, ``"sq_rel"``, ``"rmse"``, ``"rmse_log"``,
            ``"delta1"``, ``"delta2"``, ``"delta3"``. Default ``"abs_rel"``.
        lower_is_better:
            Whether a lower ``eval_metric`` is better. True for error metrics
            (``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``); False for
            threshold accuracy (``delta1``, ``delta2``, ``delta3``).
            Default True.
        mixed_precision:
            Enable automatic mixed precision (``torch.cuda.amp``). Default False.
        dataloader_num_workers:
            Number of DataLoader worker processes. Default 4.
        seed:
            Random seed for reproducibility. Default 42.

    Example::

        args = DepthTrainingArguments(
            output_dir="./checkpoints/nyu_vitb",
            num_epochs=25,
            learning_rate=1e-4,
            batch_size=8,
        )
        args.to_json("./checkpoints/nyu_vitb/args.json")
    """

    output_dir: str

    # Optimisation
    num_epochs: int = 25
    learning_rate: float = 5e-5
    backbone_lr_scale: float = 0.1
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_clip_val: float = 1.0

    # Backbone freeze schedule
    freeze_backbone_epochs: int = 0

    # LR scheduler
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 0

    # Loss
    si_loss_weight: float = 1.0
    grad_loss_weight: float = 0.5
    si_lam: float = 0.85

    # Checkpointing / logging
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    log_every_n_steps: int = 50

    # Evaluation
    eval_metric: str = "abs_rel"
    lower_is_better: bool = True

    # Hardware
    mixed_precision: bool = False
    dataloader_num_workers: int = 4
    seed: int = 42

    _VALID_SCHEDULERS = ("cosine", "linear", "step", "plateau")
    _VALID_METRICS = (
        "abs_rel", "sq_rel", "rmse", "rmse_log",
        "delta1", "delta2", "delta3",
    )

    def __post_init__(self):
        if self.lr_scheduler not in self._VALID_SCHEDULERS:
            raise ValueError(
                f"lr_scheduler must be one of {self._VALID_SCHEDULERS}, "
                f"got {self.lr_scheduler!r}"
            )
        if self.eval_metric not in self._VALID_METRICS:
            raise ValueError(
                f"eval_metric must be one of {self._VALID_METRICS}, "
                f"got {self.eval_metric!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")

    def to_json(self, path: str) -> None:
        """Serialise arguments to a JSON file.

        Args:
            path: File path to write. Parent directories are created if needed.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = dataclasses.asdict(self)
        # Remove class-level attributes that shouldn't be serialised
        for key in ("_VALID_SCHEDULERS", "_VALID_METRICS"):
            data.pop(key, None)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: str) -> "DepthTrainingArguments":
        """Load arguments from a JSON file.

        Args:
            path: File path to read.

        Returns:
            :class:`DepthTrainingArguments` instance.
        """
        data = json.loads(Path(path).read_text())
        # Drop unknown keys for forward compatibility
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)

    def __repr__(self) -> str:
        fields = dataclasses.fields(self)
        pairs = [f"  {f.name}={getattr(self, f.name)!r}" for f in fields
                 if not f.name.startswith("_")]
        return "DepthTrainingArguments(\n" + ",\n".join(pairs) + "\n)"
