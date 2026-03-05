"""Tests for Auto-class resolution of all model variants."""

import pytest

from depth_estimation.registry import MODEL_REGISTRY
from depth_estimation.models.auto.modeling_auto import AutoDepthModel
from depth_estimation.models.auto.processing_auto import AutoProcessor
from depth_estimation.processing_utils import DepthProcessor

# Import all models to trigger registration
import depth_estimation.models.depth_anything_v1  # noqa: F401
import depth_estimation.models.depth_anything_v2  # noqa: F401
import depth_estimation.models.depth_anything_v3  # noqa: F401
import depth_estimation.models.zoedepth  # noqa: F401
import depth_estimation.models.midas  # noqa: F401
import depth_estimation.models.depth_pro  # noqa: F401
import depth_estimation.models.pixel_perfect_depth  # noqa: F401


class TestModelRegistry:
    def test_registered_types(self):
        types = MODEL_REGISTRY.list_model_types()
        expected_types = [
            "depth-anything-v1", "depth-anything-v2", "depth-anything-v3",
            "zoedepth", "midas", "depth-pro", "pixel-perfect-depth",
        ]
        for t in expected_types:
            assert t in types, f"{t} not registered"

    def test_registered_variants(self):
        variants = MODEL_REGISTRY.list_variants()
        expected = [
            # v1
            "depth-anything-v1-vits", "depth-anything-v1-vitb", "depth-anything-v1-vitl",
            # v2
            "depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl",
            # v3
            "depth-anything-v3-small", "depth-anything-v3-base", "depth-anything-v3-large",
            "depth-anything-v3-giant", "depth-anything-v3-nested-giant-large",
            "depth-anything-v3-metric-large", "depth-anything-v3-mono-large",
            # zoedepth
            "zoedepth",
            # midas
            "midas-dpt-large", "midas-dpt-hybrid", "midas-beit-large",
            # depth-pro
            "depth-pro",
            # ppd
            "pixel-perfect-depth",
        ]
        for v in expected:
            assert v in variants, f"{v} not registered"

    def test_total_variants(self):
        """Total: 3 + 3 + 7 + 1 + 3 + 1 + 1 = 19 variants."""
        variants = MODEL_REGISTRY.list_variants()
        assert len(variants) >= 19

    def test_resolve_by_type(self):
        model_type = MODEL_REGISTRY.resolve_model_type("depth-anything-v1")
        assert model_type == "depth-anything-v1"

    def test_resolve_by_variant(self):
        model_type = MODEL_REGISTRY.resolve_model_type("depth-anything-v2-vitb")
        assert model_type == "depth-anything-v2"

    def test_resolve_new_models(self):
        assert MODEL_REGISTRY.resolve_model_type("zoedepth") == "zoedepth"
        assert MODEL_REGISTRY.resolve_model_type("midas-dpt-large") == "midas"
        assert MODEL_REGISTRY.resolve_model_type("depth-pro") == "depth-pro"
        assert MODEL_REGISTRY.resolve_model_type("pixel-perfect-depth") == "pixel-perfect-depth"
        assert MODEL_REGISTRY.resolve_model_type("depth-anything-v3-large") == "depth-anything-v3"

    def test_resolve_unknown(self):
        with pytest.raises(ValueError, match="Unknown model identifier"):
            MODEL_REGISTRY.resolve_model_type("nonexistent-model")


class TestAutoDepthModel:
    @pytest.mark.parametrize("variant_id,expected_class", [
        ("depth-anything-v1-vits", "DepthAnythingV1Model"),
        ("depth-anything-v2-vitb", "DepthAnythingV2Model"),
        ("depth-anything-v3-large", "DepthAnythingV3Model"),
        ("zoedepth", "ZoeDepthModel"),
        ("midas-dpt-large", "MiDaSModel"),
        ("midas-dpt-hybrid", "MiDaSModel"),
        ("midas-beit-large", "MiDaSModel"),
        ("depth-pro", "DepthProModel"),
        ("pixel-perfect-depth", "PixelPerfectDepthModel"),
    ])
    def test_resolves_correct_class(self, variant_id, expected_class):
        model_cls = MODEL_REGISTRY.get_model_cls(variant_id)
        assert model_cls.__name__ == expected_class

    def test_cannot_instantiate(self):
        with pytest.raises(RuntimeError):
            AutoDepthModel()


class TestAutoProcessor:
    @pytest.mark.parametrize("variant_id", [
        "depth-anything-v1-vits", "depth-anything-v1-vitb", "depth-anything-v1-vitl",
        "depth-anything-v2-vits", "depth-anything-v2-vitb", "depth-anything-v2-vitl",
        "depth-anything-v3-small", "depth-anything-v3-large",
        "zoedepth",
        "midas-dpt-large", "midas-beit-large",
        "depth-pro",
        "pixel-perfect-depth",
    ])
    def test_returns_processor(self, variant_id):
        """AutoProcessor should return a DepthProcessor for all variants."""
        processor = AutoProcessor.from_pretrained(variant_id)
        assert isinstance(processor, DepthProcessor)

    def test_cannot_instantiate(self):
        with pytest.raises(RuntimeError):
            AutoProcessor()
