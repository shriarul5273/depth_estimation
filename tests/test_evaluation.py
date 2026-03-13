"""Tests for the depth_estimation.evaluation module.

All tests use synthetic tensors / arrays — no model weights are downloaded.
The profile_latency smoke test runs on CPU with a mock model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from depth_estimation.evaluation import (
    DepthMetrics,
    Evaluator,
    align_least_squares,
)


# ---------------------------------------------------------------------------
# align_least_squares
# ---------------------------------------------------------------------------

class TestAlignLeastSquares:
    def test_perfect_prediction(self):
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.5, 5.0, (64, 64)).astype(np.float32)
        mask = np.ones((64, 64), dtype=bool)
        scale, shift = align_least_squares(pred, pred, mask)
        assert abs(scale - 1.0) < 1e-4
        assert abs(shift - 0.0) < 1e-4

    def test_known_scale_and_shift(self):
        rng = np.random.default_rng(1)
        pred = rng.uniform(1.0, 4.0, (32, 32)).astype(np.float32)
        true_scale, true_shift = 2.5, 1.0
        target = (true_scale * pred + true_shift).astype(np.float32)
        mask = np.ones((32, 32), dtype=bool)

        scale, shift = align_least_squares(pred, target, mask)
        assert abs(scale - true_scale) < 1e-3
        assert abs(shift - true_shift) < 1e-3

    def test_masked_pixels_ignored(self):
        rng = np.random.default_rng(42)
        pred = rng.uniform(1.0, 4.0, (16, 16)).astype(np.float32)
        true_scale, true_shift = 3.0, 0.5
        target = (true_scale * pred + true_shift).astype(np.float32)

        # Only use the top-left quadrant; the rest have a different (wrong) relationship
        mask = np.zeros((16, 16), dtype=bool)
        mask[:8, :8] = True

        scale, shift = align_least_squares(pred, target, mask)
        assert abs(scale - true_scale) < 1e-3
        assert abs(shift - true_shift) < 1e-3

    def test_fewer_than_2_valid_pixels_returns_identity(self):
        pred = np.ones((4, 4), dtype=np.float32)
        target = 2.0 * pred
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True  # only 1 pixel

        scale, shift = align_least_squares(pred, target, mask)
        assert scale == 1.0
        assert shift == 0.0

    def test_returns_floats(self):
        pred = np.random.rand(8, 8).astype(np.float32)
        target = pred * 1.5
        mask = np.ones((8, 8), dtype=bool)
        scale, shift = align_least_squares(pred, target, mask)
        assert isinstance(scale, float)
        assert isinstance(shift, float)


# ---------------------------------------------------------------------------
# DepthMetrics
# ---------------------------------------------------------------------------

class TestDepthMetrics:
    def test_perfect_prediction_zero_error(self):
        depth = torch.full((32, 32), 2.0)
        metrics = DepthMetrics()
        result = metrics(depth, depth)
        assert result["abs_rel"] < 1e-5
        assert result["sq_rel"] < 1e-5
        assert result["rmse"] < 1e-5
        assert result["rmse_log"] < 1e-5
        assert abs(result["delta1"] - 1.0) < 1e-5
        assert abs(result["delta2"] - 1.0) < 1e-5
        assert abs(result["delta3"] - 1.0) < 1e-5

    def test_output_keys(self):
        pred = torch.ones(16, 16)
        target = torch.ones(16, 16)
        result = DepthMetrics()(pred, target)
        expected_keys = {"abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"}
        assert set(result.keys()) == expected_keys

    def test_all_values_are_float(self):
        pred = torch.rand(16, 16) + 0.1
        target = torch.rand(16, 16) + 0.1
        result = DepthMetrics()(pred, target)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_no_nan_values(self):
        pred = torch.rand(32, 32) + 0.1
        target = torch.rand(32, 32) + 0.1
        result = DepthMetrics()(pred, target)
        for k, v in result.items():
            assert not math.isnan(v), f"{k} is NaN"

    def test_zero_valid_pixels_returns_zeros(self):
        depth = torch.ones(16, 16)
        mask = torch.zeros(16, 16, dtype=torch.bool)
        result = DepthMetrics()(depth, depth, valid_mask=mask)
        for k, v in result.items():
            assert v == 0.0, f"{k} should be 0 but got {v}"

    def test_abs_rel_known_value(self):
        # pred = 2.0, target = 1.0 → abs_rel = |2-1|/1 = 1.0
        pred = torch.full((8, 8), 2.0)
        target = torch.full((8, 8), 1.0)
        result = DepthMetrics()(pred, target)
        assert abs(result["abs_rel"] - 1.0) < 1e-5

    def test_delta1_range(self):
        pred = torch.rand(32, 32) + 0.1
        target = torch.rand(32, 32) + 0.1
        result = DepthMetrics()(pred, target)
        assert 0.0 <= result["delta1"] <= 1.0
        assert result["delta1"] <= result["delta2"] <= result["delta3"]

    def test_rmse_known_value(self):
        # pred = 3.0, target = 1.0 → rmse = sqrt(mean((3-1)²)) = 2.0
        pred = torch.full((16, 16), 3.0)
        target = torch.full((16, 16), 1.0)
        result = DepthMetrics()(pred, target)
        assert abs(result["rmse"] - 2.0) < 1e-4

    def test_accepts_4d_input(self):
        pred = torch.ones(1, 1, 16, 16) * 2.0
        target = torch.ones(1, 1, 16, 16) * 2.0
        result = DepthMetrics()(pred, target)
        assert result["abs_rel"] < 1e-5

    def test_valid_mask_applied(self):
        # pred = 2.0, target = 1.0 → abs_rel = 1.0 without mask
        # but if mask masks everything out → 0.0
        pred = torch.full((8, 8), 2.0)
        target = torch.full((8, 8), 1.0)
        mask = torch.zeros(8, 8, dtype=torch.bool)
        result = DepthMetrics()(pred, target, valid_mask=mask)
        assert result["abs_rel"] == 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_single_batch_matches_depth_metrics(self):
        pred = torch.rand(1, 16, 16) + 0.1
        target = torch.rand(1, 16, 16) + 0.1

        ev = Evaluator()
        ev.update(pred, target)
        result_ev = ev.compute()

        dm = DepthMetrics()
        result_dm = dm(pred, target)

        for key in ("abs_rel", "sq_rel", "delta1", "delta2", "delta3"):
            assert abs(result_ev[key] - result_dm[key]) < 1e-4, (
                f"{key}: Evaluator={result_ev[key]:.6f}, DepthMetrics={result_dm[key]:.6f}"
            )

    def test_rmse_correct_accumulation(self):
        """RMSE over two batches must equal sqrt(SSE_total / N_total), not mean of per-batch RMSEs."""
        # Batch 1: 100 pixels with error = 1.0 → SSE1 = 100
        # Batch 2: 1 pixel with error = 10.0 → SSE2 = 100
        # Correct RMSE = sqrt(200 / 101) ≈ 1.407
        # Mean of per-batch RMSEs = (1.0 + 10.0) / 2 = 5.5 — wrong

        pred1 = torch.full((1, 10, 10), 2.0)
        target1 = torch.full((1, 10, 10), 1.0)  # error = 1.0 per pixel

        pred2 = torch.full((1, 1, 1), 11.0)
        target2 = torch.full((1, 1, 1), 1.0)   # error = 10.0

        ev = Evaluator()
        ev.update(pred1, target1)
        ev.update(pred2, target2)
        result = ev.compute()

        expected_rmse = math.sqrt(200.0 / 101.0)
        assert abs(result["rmse"] - expected_rmse) < 1e-4

    def test_reset_clears_state(self):
        pred = torch.ones(1, 8, 8) * 2.0
        target = torch.ones(1, 8, 8) * 1.0

        ev = Evaluator()
        ev.update(pred, target)
        assert ev.n_pixels > 0

        ev.reset()
        assert ev.n_pixels == 0

        result = ev.compute()
        for k, v in result.items():
            assert v == 0.0

    def test_empty_compute_returns_zeros(self):
        ev = Evaluator()
        result = ev.compute()
        assert result["abs_rel"] == 0.0
        assert result["n_pixels"] == 0

    def test_n_pixels_accumulates(self):
        ev = Evaluator()
        for _ in range(3):
            pred = torch.rand(1, 4, 4) + 0.1
            target = torch.rand(1, 4, 4) + 0.1
            ev.update(pred, target)
        assert ev.n_pixels == 48  # 3 * 4 * 4

    def test_result_contains_n_pixels(self):
        ev = Evaluator()
        ev.update(torch.ones(1, 4, 4) * 2.0, torch.ones(1, 4, 4) * 1.0)
        result = ev.compute()
        assert "n_pixels" in result
        assert result["n_pixels"] == 16

    def test_skips_zero_valid_pixels(self):
        ev = Evaluator()
        pred = torch.ones(1, 4, 4)
        target = torch.zeros(1, 4, 4)  # all zeros → all invalid
        ev.update(pred, target)
        assert ev.n_pixels == 0

    def test_4d_input(self):
        ev = Evaluator()
        pred = torch.ones(2, 1, 8, 8) * 2.0
        target = torch.ones(2, 1, 8, 8) * 1.0
        ev.update(pred, target)
        assert ev.n_pixels == 128  # 2 * 8 * 8


# ---------------------------------------------------------------------------
# profile_latency (CPU smoke test — no model download)
# ---------------------------------------------------------------------------

class TestProfileLatency:
    def test_cpu_smoke_test(self):
        """Run profile_latency with a minimal nn.Module to verify the API without downloading weights."""
        import torch.nn as nn
        from depth_estimation.evaluation import profile_latency

        class _IdentityDepthModel(nn.Module):
            """Minimal stand-in: returns an all-ones depth map."""
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, _, h, w = x.shape
                return torch.ones(b, 1, h, w, device=x.device)

        model = _IdentityDepthModel()

        result = profile_latency(
            model,
            input_size=64,
            batch_size=1,
            num_warmup=2,
            num_runs=5,
            device="cpu",
        )

        assert "mean_ms" in result
        assert "fps" in result
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert result["mean_ms"] > 0
        assert result["fps"] > 0
        assert result["device"] == "cpu"
        assert result["memory_mb"] is None  # CPU has no GPU memory
        assert result["input_shape"] == (1, 3, 64, 64)
