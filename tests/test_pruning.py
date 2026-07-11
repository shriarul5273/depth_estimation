"""Tests for depth_estimation.pruning."""

import torch
import torch.nn as nn
import pytest

from depth_estimation.pruning import prune_model, compute_sparsity, make_pruning_permanent
from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
)
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import (
    DepthAnythingV2Model,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = DepthAnythingV2Config(backbone="vits")
    return DepthAnythingV2Model(config).eval()


class TestPruneModel:
    def test_achieves_requested_sparsity(self, model):
        prune_model(model, amount=0.3)
        sparsity = compute_sparsity(model)
        assert sparsity["overall"] == pytest.approx(0.3, abs=0.01)

    def test_make_permanent_removes_reparameterization(self, model):
        prune_model(model, amount=0.3, make_permanent=True)
        keys = model.state_dict().keys()
        assert not any("weight_orig" in k or "weight_mask" in k for k in keys)

    def test_make_permanent_false_keeps_reparameterization(self, model):
        prune_model(model, amount=0.3, make_permanent=False)
        keys = model.state_dict().keys()
        assert any("weight_orig" in k for k in keys)
        assert any("weight_mask" in k for k in keys)

    def test_forward_pass_still_works_after_pruning(self, model):
        prune_model(model, amount=0.3)
        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 518, 518)
        assert not torch.isnan(out).any()

    def test_exclude_skips_matching_modules(self, model):
        prune_model(model, amount=0.3, exclude=["pretrained"])
        sparsity = compute_sparsity(model)
        backbone_layers = [k for k in sparsity if "pretrained" in k]
        decoder_layers = [
            k for k in sparsity if "pretrained" not in k and k != "overall"
        ]
        assert backbone_layers, "expected at least one backbone layer name match"
        assert decoder_layers, "expected at least one decoder layer name match"
        # Excluded layers should show ~0 sparsity — allow a tiny tolerance for
        # stray exact-zero weights that can occur by chance at random init,
        # same as TestComputeSparsity.test_baseline_near_zero below.
        assert all(sparsity[k] < 0.01 for k in backbone_layers)
        assert any(sparsity[k] > 0.2 for k in decoder_layers)

    def test_returns_model_for_chaining(self, model):
        result = prune_model(model, amount=0.3)
        assert result is model

    def test_invalid_method_raises(self, model):
        with pytest.raises(ValueError, match="Unknown pruning method"):
            prune_model(model, method="not_a_real_method")

    @pytest.mark.parametrize("amount", [-0.1, 1.0, 1.5])
    def test_invalid_amount_raises(self, model, amount):
        with pytest.raises(ValueError, match="amount must be in"):
            prune_model(model, amount=amount)

    def test_random_unstructured_also_achieves_sparsity(self, model):
        prune_model(model, amount=0.3, method="random_unstructured")
        sparsity = compute_sparsity(model)
        assert sparsity["overall"] == pytest.approx(0.3, abs=0.01)

    def test_no_matching_modules_warns_and_no_ops(self, model, caplog):
        # No Embedding layers exist in this model.
        prune_model(model, amount=0.3, module_types=(nn.Embedding,))
        assert "nothing was pruned" in caplog.text
        sparsity = compute_sparsity(model)
        assert sparsity["overall"] == pytest.approx(0.0, abs=1e-6)


class TestMakePruningPermanent:
    """Covers the prune-aware fine-tuning workflow documented in
    docs/pruning.md — prune(make_permanent=False), [fine-tune], then
    make_pruning_permanent() before export.
    """

    def test_removes_reparameterization(self, model):
        prune_model(model, amount=0.3, make_permanent=False)
        assert any("weight_orig" in k for k in model.state_dict())

        make_pruning_permanent(model)

        keys = model.state_dict().keys()
        assert not any("weight_orig" in k or "weight_mask" in k for k in keys)

    def test_preserves_sparsity(self, model):
        prune_model(model, amount=0.3, make_permanent=False)
        before = compute_sparsity(model)["overall"]

        make_pruning_permanent(model)

        after = compute_sparsity(model)["overall"]
        assert after == pytest.approx(before, abs=1e-6)

    def test_forward_still_works(self, model):
        prune_model(model, amount=0.3, make_permanent=False)
        make_pruning_permanent(model)

        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 518, 518)
        assert not torch.isnan(out).any()

    def test_no_op_when_nothing_pruned(self, model):
        # Must not raise even if pruning was never applied.
        make_pruning_permanent(model)
        assert compute_sparsity(model)["overall"] < 0.01

    def test_returns_model_for_chaining(self, model):
        prune_model(model, amount=0.3, make_permanent=False)
        result = make_pruning_permanent(model)
        assert result is model


class TestComputeSparsity:
    def test_baseline_near_zero(self, model):
        sparsity = compute_sparsity(model)
        assert sparsity["overall"] < 0.01

    def test_reports_per_layer(self, model):
        prune_model(model, amount=0.3)
        sparsity = compute_sparsity(model)
        assert len(sparsity) > 1  # per-layer entries + "overall"
        assert "overall" in sparsity

    def test_custom_module_types(self, model):
        # Restricting to only Conv2d should still find the DPT decoder's convs.
        sparsity = compute_sparsity(model, module_types=(nn.Conv2d,))
        assert "overall" in sparsity


class TestPruningExportIntegration:
    """Requires the optional onnx/onnxruntime packages — skipped entirely if
    they're not installed (they aren't core dependencies).
    """

    def test_pruned_model_exports_and_verifies(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        from depth_estimation.export import export_onnx

        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone="vits")
        m = DepthAnythingV2Model(config).eval()
        prune_model(m, amount=0.3)

        out = tmp_path / "pruned.onnx"
        export_onnx(m, out, input_size=518, verify=True)
        assert out.exists()
