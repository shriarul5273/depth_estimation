"""Tests for model instantiation and forward pass shapes."""

import pytest
import torch

from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import DepthAnythingV2Config
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import DepthAnythingV2Model


class TestDepthAnythingV2Model:
    """Test v2 model (uses vendored DINOv2 backbone, no external downloads)."""

    @pytest.fixture
    def model_vits(self):
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config)
        model.eval()
        return model

    def test_instantiation(self, model_vits):
        assert isinstance(model_vits, DepthAnythingV2Model)
        assert model_vits.config.model_type == "depth-anything-v2"

    def test_forward_shape(self, model_vits):
        """Forward pass with random input should produce (B, H, W) depth."""
        x = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            depth = model_vits(x)
        assert depth.dim() == 3  # (B, H, W)
        assert depth.shape[0] == 1
        assert depth.shape[1] == 518
        assert depth.shape[2] == 518

    def test_forward_batch(self, model_vits):
        """Batch forward should work."""
        x = torch.randn(2, 3, 518, 518)
        with torch.no_grad():
            depth = model_vits(x)
        assert depth.shape[0] == 2

    def test_forward_non_square(self, model_vits):
        """Non-square input (but multiple of 14) should work."""
        x = torch.randn(1, 3, 518, 714)  # 714 = 51 * 14
        with torch.no_grad():
            depth = model_vits(x)
        assert depth.shape == (1, 518, 714)


# Note: v1 model tests require torch.hub DINOv2 download.
# We skip them in CI but they can be run locally.
class TestDepthAnythingV1ModelSkip:
    @pytest.mark.skip(reason="Requires torch.hub DINOv2 download")
    def test_instantiation(self):
        from depth_estimation.models.depth_anything_v1.configuration_depth_anything_v1 import DepthAnythingV1Config
        from depth_estimation.models.depth_anything_v1.modeling_depth_anything_v1 import DepthAnythingV1Model
        config = DepthAnythingV1Config(backbone="vits")
        model = DepthAnythingV1Model(config)
        assert model.config.model_type == "depth-anything-v1"
