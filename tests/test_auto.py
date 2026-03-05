"""Tests for Auto-class resolution of all model variants."""

import pytest

from depth_estimation.registry import MODEL_REGISTRY
from depth_estimation.models.auto.modeling_auto import AutoDepthModel
from depth_estimation.models.auto.processing_auto import AutoProcessor
from depth_estimation.processing_utils import DepthProcessor


# Import models to trigger registration
import depth_estimation.models.depth_anything_v1  # noqa: F401
import depth_estimation.models.depth_anything_v2  # noqa: F401


class TestModelRegistry:
    def test_registered_types(self):
        types = MODEL_REGISTRY.list_model_types()
        assert "depth-anything-v1" in types
        assert "depth-anything-v2" in types

    def test_registered_variants(self):
        variants = MODEL_REGISTRY.list_variants()
        expected = [
            "depth-anything-v1-vits", "depth-anything-v1-vitb", "depth-anything-v1-vitl",
            "depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl",
        ]
        for v in expected:
            assert v in variants

    def test_resolve_by_type(self):
        model_type = MODEL_REGISTRY.resolve_model_type("depth-anything-v1")
        assert model_type == "depth-anything-v1"

    def test_resolve_by_variant(self):
        model_type = MODEL_REGISTRY.resolve_model_type("depth-anything-v2-vitb")
        assert model_type == "depth-anything-v2"

    def test_resolve_unknown(self):
        with pytest.raises(ValueError, match="Unknown model identifier"):
            MODEL_REGISTRY.resolve_model_type("nonexistent-model")


class TestAutoDepthModel:
    @pytest.mark.parametrize("variant_id", [
        "depth-anything-v1-vits", "depth-anything-v1-vitb", "depth-anything-v1-vitl",
        "depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl",
    ])
    def test_resolves_correct_class(self, variant_id):
        """AutoDepthModel should resolve the correct model class."""
        model_cls = MODEL_REGISTRY.get_model_cls(variant_id)
        if "v1" in variant_id:
            assert model_cls.__name__ == "DepthAnythingV1Model"
        else:
            assert model_cls.__name__ == "DepthAnythingV2Model"

    def test_cannot_instantiate(self):
        with pytest.raises(RuntimeError):
            AutoDepthModel()


class TestAutoProcessor:
    @pytest.mark.parametrize("variant_id", [
        "depth-anything-v1-vits", "depth-anything-v1-vitb", "depth-anything-v1-vitl",
        "depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl",
    ])
    def test_returns_processor(self, variant_id):
        """AutoProcessor should return a DepthProcessor for all variants."""
        processor = AutoProcessor.from_pretrained(variant_id)
        assert isinstance(processor, DepthProcessor)
        assert processor.input_size == 518
        assert processor.patch_size == 14

    def test_cannot_instantiate(self):
        with pytest.raises(RuntimeError):
            AutoProcessor()
