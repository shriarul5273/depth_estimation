"""Tests for DepthTrainingArguments and DepthTrainer.

DepthTrainingArguments: fully tested with synthetic data.
DepthTrainer: smoke-tested with a tiny mock nn.Module — no model weights downloaded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from depth_estimation.training_args import DepthTrainingArguments


# ---------------------------------------------------------------------------
# DepthTrainingArguments
# ---------------------------------------------------------------------------

class TestDepthTrainingArguments:
    def test_defaults(self):
        args = DepthTrainingArguments(output_dir="/tmp/test")
        assert args.num_epochs == 25
        assert args.learning_rate == 5e-5
        assert args.batch_size == 8
        assert args.lr_scheduler == "cosine"
        assert args.eval_metric == "abs_rel"
        assert args.mixed_precision is False

    def test_invalid_scheduler_raises(self):
        with pytest.raises(ValueError, match="lr_scheduler"):
            DepthTrainingArguments(output_dir="/tmp/x", lr_scheduler="cyclic")

    def test_invalid_eval_metric_raises(self):
        with pytest.raises(ValueError, match="eval_metric"):
            DepthTrainingArguments(output_dir="/tmp/x", eval_metric="mae")

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            DepthTrainingArguments(output_dir="/tmp/x", batch_size=0)

    def test_num_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="num_epochs"):
            DepthTrainingArguments(output_dir="/tmp/x", num_epochs=0)

    @pytest.mark.parametrize("scheduler", ["cosine", "linear", "step", "plateau"])
    def test_all_valid_schedulers_accepted(self, scheduler):
        args = DepthTrainingArguments(output_dir="/tmp/x", lr_scheduler=scheduler)
        assert args.lr_scheduler == scheduler

    @pytest.mark.parametrize("metric", ["abs_rel", "sq_rel", "rmse", "rmse_log",
                                         "delta1", "delta2", "delta3"])
    def test_all_valid_metrics_accepted(self, metric):
        args = DepthTrainingArguments(output_dir="/tmp/x", eval_metric=metric)
        assert args.eval_metric == metric

    def test_to_json_and_from_json_roundtrip(self, tmp_path):
        args = DepthTrainingArguments(
            output_dir=str(tmp_path / "ckpts"),
            num_epochs=10,
            learning_rate=1e-4,
            batch_size=4,
            lr_scheduler="plateau",
            mixed_precision=True,
        )
        json_path = str(tmp_path / "args.json")
        args.to_json(json_path)

        loaded = DepthTrainingArguments.from_json(json_path)
        assert loaded.num_epochs == 10
        assert loaded.learning_rate == 1e-4
        assert loaded.batch_size == 4
        assert loaded.lr_scheduler == "plateau"
        assert loaded.mixed_precision is True

    def test_to_json_creates_parent_dirs(self, tmp_path):
        args = DepthTrainingArguments(output_dir="/tmp/x")
        deep_path = str(tmp_path / "a" / "b" / "args.json")
        args.to_json(deep_path)
        assert Path(deep_path).exists()

    def test_from_json_ignores_unknown_keys(self, tmp_path):
        """Forward compatibility: unknown keys in JSON should be silently dropped."""
        json_path = tmp_path / "args.json"
        data = {"output_dir": "/tmp/x", "num_epochs": 5, "UNKNOWN_KEY": "foo"}
        json_path.write_text(json.dumps(data))
        loaded = DepthTrainingArguments.from_json(str(json_path))
        assert loaded.num_epochs == 5

    def test_repr_contains_key_fields(self):
        args = DepthTrainingArguments(output_dir="/tmp/x", num_epochs=7)
        r = repr(args)
        assert "num_epochs" in r
        assert "7" in r
        assert "output_dir" in r


# ---------------------------------------------------------------------------
# DepthTrainer smoke tests (tiny mock model, no weights downloaded)
# ---------------------------------------------------------------------------

class _TinyDepthDataset(Dataset):
    """In-memory dataset: N samples of fixed small tensors."""

    def __init__(self, n=8, h=16, w=16):
        self._n = n
        self._h = h
        self._w = w

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        g = torch.Generator()
        g.manual_seed(idx)
        pv = torch.rand(3, self._h, self._w, generator=g)
        dm = torch.rand(1, self._h, self._w, generator=g).clamp(min=0.1) + 0.5
        vm = torch.ones(1, self._h, self._w, dtype=torch.bool)
        return {"pixel_values": pv, "depth_map": dm, "valid_mask": vm}


class _TinyDepthModel(nn.Module):
    """Minimal depth model: conv → relu → conv, outputs (B, 1, H, W)."""

    def __init__(self, h=16, w=16):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.decoder  = nn.Conv2d(8, 1, 1)
        self._h = h
        self._w = w

    def forward(self, x):
        return torch.sigmoid(self.decoder(self.backbone(x)))

    # Minimal BaseDepthModel-compatible training helpers
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_groups(self, backbone_lr_scale=0.1):
        bb_ids = {id(p) for p in self.backbone.parameters()}
        return [
            {"params": [p for p in self.parameters() if id(p) not in bb_ids and p.requires_grad],
             "lr_scale": 1.0},
            {"params": [p for p in self.parameters() if id(p) in bb_ids and p.requires_grad],
             "lr_scale": backbone_lr_scale},
        ]

    @property
    def config(self):
        return None   # no config.to_dict() needed for these tests


class TestDepthTrainer:
    def _make_trainer(self, tmp_path, **arg_overrides):
        from depth_estimation.trainer import DepthTrainer

        model = _TinyDepthModel()
        train_ds = _TinyDepthDataset(n=8)
        val_ds   = _TinyDepthDataset(n=4)

        base = dict(
            output_dir=str(tmp_path / "ckpts"),
            num_epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            warmup_epochs=0,
            save_every_n_epochs=2,
            eval_every_n_epochs=1,
            log_every_n_steps=1,
            dataloader_num_workers=0,
            mixed_precision=False,
        )
        base.update(arg_overrides)
        args = DepthTrainingArguments(**base)
        trainer = DepthTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        return trainer, model

    def test_train_runs_without_error(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        trainer.train()

    def test_final_checkpoint_saved(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        trainer.train()
        assert (tmp_path / "ckpts" / "final" / "model.pt").exists()
        assert (tmp_path / "ckpts" / "final" / "training_state.pt").exists()

    def test_best_model_checkpoint_saved(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path)
        trainer.train()
        # best_model is written whenever eval metric improves
        assert (tmp_path / "ckpts" / "best_model").exists()

    def test_model_weights_loadable(self, tmp_path):
        trainer, model = self._make_trainer(tmp_path)
        trainer.train()

        # Load the saved weights back into a fresh model
        fresh = _TinyDepthModel()
        state = torch.load(
            tmp_path / "ckpts" / "final" / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        fresh.load_state_dict(state)

    def test_resume_from_checkpoint(self, tmp_path):
        trainer, _ = self._make_trainer(tmp_path, save_every_n_epochs=1)
        trainer.train()

        ckpt_dir = tmp_path / "ckpts" / "checkpoint_epoch_0000"
        assert ckpt_dir.exists()

        # Resume — should not raise
        trainer2, _ = self._make_trainer(tmp_path / "resumed", save_every_n_epochs=2)
        trainer2.train(resume_from=str(ckpt_dir))

    def test_freeze_backbone_schedule(self, tmp_path):
        """freeze_backbone_epochs=1 with 2 total epochs: backbone unfrozen at epoch 1."""
        trainer, model = self._make_trainer(
            tmp_path, freeze_backbone_epochs=1, num_epochs=2
        )
        trainer.train()
        # After training, backbone should be unfrozen
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_full_finetune_backbone_always_trainable(self, tmp_path):
        """freeze_backbone_epochs=0 (default): backbone never frozen."""
        trainer, model = self._make_trainer(tmp_path, freeze_backbone_epochs=0)
        trainer.train()
        # Backbone should remain trainable after training
        assert all(p.requires_grad for p in model.backbone.parameters())

    @pytest.mark.parametrize("scheduler", ["cosine", "linear", "step", "plateau"])
    def test_all_schedulers_run(self, tmp_path, scheduler):
        trainer, _ = self._make_trainer(
            tmp_path / scheduler,
            lr_scheduler=scheduler,
        )
        trainer.train()  # must not raise

    def test_no_eval_dataset(self, tmp_path):
        from depth_estimation.trainer import DepthTrainer

        model = _TinyDepthModel()
        args = DepthTrainingArguments(
            output_dir=str(tmp_path / "ckpts"),
            num_epochs=2,
            batch_size=4,
            warmup_epochs=0,
            save_every_n_epochs=2,
            dataloader_num_workers=0,
        )
        trainer = DepthTrainer(
            model=model,
            args=args,
            train_dataset=_TinyDepthDataset(n=8),
            eval_dataset=None,
        )
        trainer.train()   # must not raise without eval dataset

    def test_no_trainable_params_raises(self, tmp_path):
        from depth_estimation.trainer import DepthTrainer

        model = _TinyDepthModel()
        model.freeze_backbone()
        # Freeze decoder too → nothing trainable
        for p in model.decoder.parameters():
            p.requires_grad = False

        args = DepthTrainingArguments(
            output_dir=str(tmp_path),
            num_epochs=1,
            dataloader_num_workers=0,
        )
        with pytest.raises(RuntimeError, match="No trainable parameters"):
            DepthTrainer(
                model=model,
                args=args,
                train_dataset=_TinyDepthDataset(n=4),
            ).train()
