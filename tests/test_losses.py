"""Tests for depth_estimation.losses.

All tests use synthetic tensors — no model weights are downloaded.
"""

from __future__ import annotations

import math

import pytest
import torch

from depth_estimation.losses import (
    BerHuLoss,
    CombinedDepthLoss,
    GradientLoss,
    ScaleInvariantLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred_gt(b=2, h=32, w=32, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    pred = torch.rand(b, h, w, generator=g).clamp(min=0.1) + 0.5
    gt   = torch.rand(b, h, w, generator=g).clamp(min=0.1) + 0.5
    mask = torch.ones(b, h, w, dtype=torch.bool)
    return pred, gt, mask


# ---------------------------------------------------------------------------
# ScaleInvariantLoss
# ---------------------------------------------------------------------------

class TestScaleInvariantLoss:
    def test_perfect_prediction_is_zero(self):
        loss_fn = ScaleInvariantLoss()
        pred, _, mask = _pred_gt()
        loss = loss_fn(pred, pred, mask)
        assert loss.item() < 1e-5

    def test_loss_is_non_negative(self):
        loss_fn = ScaleInvariantLoss()
        pred, gt, mask = _pred_gt()
        assert loss_fn(pred, gt, mask).item() >= 0.0

    def test_no_nan(self):
        loss_fn = ScaleInvariantLoss()
        pred, gt, mask = _pred_gt()
        assert not torch.isnan(loss_fn(pred, gt, mask))

    def test_empty_mask_returns_zero(self):
        loss_fn = ScaleInvariantLoss()
        pred, gt, _ = _pred_gt(b=1)
        mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        loss = loss_fn(pred, gt, mask)
        assert loss.item() == 0.0

    def test_lam_one_is_zero_for_constant_scale_error(self):
        """With lam=1, SI = Var(d). If pred = c*gt, d is constant → Var = 0."""
        loss_fn = ScaleInvariantLoss(lam=1.0)
        _, gt, mask = _pred_gt(b=1)
        pred = gt * 3.7   # wrong scale, but constant shift in log space
        assert loss_fn(pred, gt, mask).item() < 1e-5

    def test_returns_scalar(self):
        loss_fn = ScaleInvariantLoss()
        pred, gt, mask = _pred_gt()
        assert loss_fn(pred, gt, mask).dim() == 0

    def test_lam_zero_equals_log_mse(self):
        """When lam=0, SI loss reduces to mean(d_i^2) — pure log MSE."""
        loss_fn = ScaleInvariantLoss(lam=0.0)
        pred, gt, mask = _pred_gt(b=1)
        d = (torch.log(pred) - torch.log(gt))[mask]
        expected = (d ** 2).mean().item()
        got = loss_fn(pred, gt, mask).item()
        assert abs(got - expected) < 1e-5

    def test_batched_vs_single(self):
        """Loss on a 2-sample batch should equal mean of per-sample losses."""
        loss_fn = ScaleInvariantLoss()
        pred, gt, mask = _pred_gt(b=2)
        batch_loss = loss_fn(pred, gt, mask).item()

        losses = []
        for i in range(2):
            p = pred[i].unsqueeze(0)
            g = gt[i].unsqueeze(0)
            m = mask[i].unsqueeze(0)
            losses.append(loss_fn(p, g, m).item())

        assert abs(batch_loss - sum(losses) / 2) < 1e-5


# ---------------------------------------------------------------------------
# GradientLoss
# ---------------------------------------------------------------------------

class TestGradientLoss:
    def test_perfect_prediction_is_zero(self):
        loss_fn = GradientLoss()
        pred, _, mask = _pred_gt()
        assert loss_fn(pred, pred, mask).item() < 1e-5

    def test_no_nan(self):
        loss_fn = GradientLoss()
        pred, gt, mask = _pred_gt()
        assert not torch.isnan(loss_fn(pred, gt, mask))

    def test_loss_is_non_negative(self):
        loss_fn = GradientLoss()
        pred, gt, mask = _pred_gt()
        assert loss_fn(pred, gt, mask).item() >= 0.0

    def test_empty_mask_no_nan(self):
        loss_fn = GradientLoss()
        pred, gt, _ = _pred_gt(b=1)
        mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        loss = loss_fn(pred, gt, mask)
        assert not torch.isnan(loss)

    def test_returns_scalar(self):
        loss_fn = GradientLoss()
        pred, gt, mask = _pred_gt()
        assert loss_fn(pred, gt, mask).dim() == 0

    def test_nonzero_when_prediction_wrong(self):
        """Loss should be > 0 when pred differs from gt with spatial structure."""
        loss_fn = GradientLoss()
        pred, gt, mask = _pred_gt(b=2, h=32, w=32)
        assert loss_fn(pred, gt, mask).item() > 0.0


# ---------------------------------------------------------------------------
# BerHuLoss
# ---------------------------------------------------------------------------

class TestBerHuLoss:
    def test_perfect_prediction_is_zero(self):
        loss_fn = BerHuLoss()
        pred, _, mask = _pred_gt()
        assert loss_fn(pred, pred, mask).item() < 1e-5

    def test_no_nan(self):
        loss_fn = BerHuLoss()
        pred, gt, mask = _pred_gt()
        assert not torch.isnan(loss_fn(pred, gt, mask))

    def test_loss_is_non_negative(self):
        loss_fn = BerHuLoss()
        pred, gt, mask = _pred_gt()
        assert loss_fn(pred, gt, mask).item() >= 0.0

    def test_empty_mask_returns_zero(self):
        loss_fn = BerHuLoss()
        pred, gt, _ = _pred_gt(b=1)
        mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        assert loss_fn(pred, gt, mask).item() == 0.0

    def test_returns_scalar(self):
        loss_fn = BerHuLoss()
        pred, gt, mask = _pred_gt()
        assert loss_fn(pred, gt, mask).dim() == 0


# ---------------------------------------------------------------------------
# CombinedDepthLoss
# ---------------------------------------------------------------------------

class TestCombinedDepthLoss:
    def test_output_keys(self):
        loss_fn = CombinedDepthLoss()
        pred, gt, mask = _pred_gt()
        result = loss_fn(pred, gt, mask)
        assert set(result.keys()) == {"loss", "si_loss", "grad_loss"}

    def test_perfect_prediction_near_zero(self):
        loss_fn = CombinedDepthLoss()
        pred, _, mask = _pred_gt()
        result = loss_fn(pred, pred, mask)
        assert result["loss"].item() < 1e-4
        assert result["si_loss"].item() < 1e-5
        assert result["grad_loss"].item() < 1e-5

    def test_no_nan(self):
        loss_fn = CombinedDepthLoss()
        pred, gt, mask = _pred_gt()
        result = loss_fn(pred, gt, mask)
        for k, v in result.items():
            assert not torch.isnan(v), f"{k} is NaN"

    def test_total_equals_weighted_sum(self):
        si_w, grad_w = 1.5, 0.3
        loss_fn = CombinedDepthLoss(si_weight=si_w, grad_weight=grad_w)
        pred, gt, mask = _pred_gt()
        result = loss_fn(pred, gt, mask)
        expected = si_w * result["si_loss"] + grad_w * result["grad_loss"]
        assert torch.allclose(result["loss"], expected, atol=1e-6)

    def test_zero_grad_weight(self):
        loss_fn = CombinedDepthLoss(si_weight=1.0, grad_weight=0.0)
        pred, gt, mask = _pred_gt()
        result = loss_fn(pred, gt, mask)
        assert torch.allclose(result["loss"], result["si_loss"])

    def test_all_values_are_tensors(self):
        loss_fn = CombinedDepthLoss()
        pred, gt, mask = _pred_gt()
        result = loss_fn(pred, gt, mask)
        for k, v in result.items():
            assert isinstance(v, torch.Tensor), f"{k} is not a Tensor"

    def test_backward_works(self):
        loss_fn = CombinedDepthLoss()
        # Use values already in valid range so no clamp needed (clamp creates non-leaf)
        pred = torch.rand(2, 16, 16) + 0.5   # values in (0.5, 1.5) — safe for log
        pred.requires_grad_(True)
        gt   = (torch.rand(2, 16, 16) + 0.5).detach()
        mask = torch.ones(2, 16, 16, dtype=torch.bool)
        result = loss_fn(pred, gt, mask)
        result["loss"].backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
