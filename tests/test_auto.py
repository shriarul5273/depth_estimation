"""Tests for Auto-class resolution of all model variants."""

import pytest

from depth_estimation.registry import MODEL_REGISTRY
from depth_estimation.models.auto.modeling_auto import AutoDepthModel
from depth_estimation.models.auto.processing_auto import AutoProcessor
from depth_estimation.processing_utils import DepthProcessor

# Import all models to trigger registration


class TestModelRegistry:
    def test_registered_types(self):
        types = MODEL_REGISTRY.list_model_types()
        expected_types = [
            "depth-anything-v1",
            "depth-anything-v2",
            "depth-anything-v3",
            "zoedepth",
            "midas",
            "depth-pro",
            "depth-pro-2",
            "pixel-perfect-depth",
            "marigold-dc",
            "moge",
            "vggt",
            "omnivggt",
            "s2m2",
        ]
        for t in expected_types:
            assert t in types, f"{t} not registered"

    def test_registered_variants(self):
        variants = MODEL_REGISTRY.list_variants()
        expected = [
            # v1
            "depth-anything-v1-vits",
            "depth-anything-v1-vitb",
            "depth-anything-v1-vitl",
            # v2
            "depth-anything-v2-vits",
            "depth-anything-v2-vitb",
            "depth-anything-v2-vitl",
            # v3
            "depth-anything-v3-small",
            "depth-anything-v3-base",
            "depth-anything-v3-large",
            "depth-anything-v3-giant",
            "depth-anything-v3-nested-giant-large",
            "depth-anything-v3-metric-large",
            "depth-anything-v3-mono-large",
            # zoedepth
            "zoedepth",
            # midas
            "midas-dpt-large",
            "midas-dpt-hybrid",
            "midas-beit-large",
            # depth-pro
            "depth-pro",
            # depth-pro-2 (4 variants)
            "depth-pro-2-s",
            "depth-pro-2-b",
            "depth-pro-2-l",
            "depth-pro-2-h",
            # ppd
            "pixel-perfect-depth",
            # marigold-dc
            "marigold-dc",
            # moge (5 variants)
            "moge-v1",
            "moge-v2-vitl",
            "moge-v2-vitl-normal",
            "moge-v2-vitb-normal",
            "moge-v2-vits-normal",
            # vggt (2 variants)
            "vggt",
            "vggt-commercial",
            # omnivggt (1 variant)
            "omnivggt",
            # s2m2 (4 variants, stereo)
            "s2m2-s",
            "s2m2-m",
            "s2m2-l",
            "s2m2-xl",
        ]
        for v in expected:
            assert v in variants, f"{v} not registered"

    def test_total_variants(self):
        """Total: 3 + 3 + 7 + 1 + 3 + 1 + 1 + 1 + 5 + 2 + 1 = 28 variants."""
        variants = MODEL_REGISTRY.list_variants()
        assert len(variants) >= 28

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
        assert MODEL_REGISTRY.resolve_model_type("depth-pro-2-l") == "depth-pro-2"
        assert (
            MODEL_REGISTRY.resolve_model_type("pixel-perfect-depth")
            == "pixel-perfect-depth"
        )
        assert (
            MODEL_REGISTRY.resolve_model_type("depth-anything-v3-large")
            == "depth-anything-v3"
        )
        assert MODEL_REGISTRY.resolve_model_type("marigold-dc") == "marigold-dc"
        assert MODEL_REGISTRY.resolve_model_type("s2m2-xl") == "s2m2"

    def test_resolve_unknown(self):
        with pytest.raises(ValueError, match="Unknown model identifier"):
            MODEL_REGISTRY.resolve_model_type("nonexistent-model")


class TestAutoDepthModel:
    @pytest.mark.parametrize(
        "variant_id,expected_class",
        [
            ("depth-anything-v1-vits", "DepthAnythingV1Model"),
            ("depth-anything-v2-vitb", "DepthAnythingV2Model"),
            ("depth-anything-v3-large", "DepthAnythingV3Model"),
            ("zoedepth", "ZoeDepthModel"),
            ("midas-dpt-large", "MiDaSModel"),
            ("midas-dpt-hybrid", "MiDaSModel"),
            ("midas-beit-large", "MiDaSModel"),
            ("depth-pro", "DepthProModel"),
            ("depth-pro-2-s", "DepthPro2Model"),
            ("depth-pro-2-h", "DepthPro2Model"),
            ("pixel-perfect-depth", "PixelPerfectDepthModel"),
            ("marigold-dc", "MarigoldDCModel"),
            ("moge-v1", "MoGeModel"),
            ("moge-v2-vitl", "MoGeModel"),
            ("moge-v2-vits-normal", "MoGeModel"),
            ("vggt", "VGGTModel"),
            ("vggt-commercial", "VGGTModel"),
            ("omnivggt", "OmniVGGTModel"),
            ("s2m2-s", "S2M2Model"),
            ("s2m2-xl", "S2M2Model"),
        ],
    )
    def test_resolves_correct_class(self, variant_id, expected_class):
        model_cls = MODEL_REGISTRY.get_model_cls(variant_id)
        assert model_cls.__name__ == expected_class

    def test_cannot_instantiate(self):
        with pytest.raises(RuntimeError):
            AutoDepthModel()


class TestAutoProcessor:
    @pytest.mark.parametrize(
        "variant_id",
        [
            "depth-anything-v1-vits",
            "depth-anything-v1-vitb",
            "depth-anything-v1-vitl",
            "depth-anything-v2-vits",
            "depth-anything-v2-vitb",
            "depth-anything-v2-vitl",
            "depth-anything-v3-small",
            "depth-anything-v3-large",
            "zoedepth",
            "midas-dpt-large",
            "midas-beit-large",
            "depth-pro",
            "depth-pro-2-s",
            "depth-pro-2-h",
            "pixel-perfect-depth",
            "marigold-dc",
            "moge-v1",
            "moge-v2-vitl",
            "moge-v2-vits-normal",
            "vggt",
            "vggt-commercial",
            "omnivggt",
            "s2m2-s",
            "s2m2-xl",
        ],
    )
    def test_returns_processor(self, variant_id):
        """AutoProcessor should return a DepthProcessor for all variants."""
        processor = AutoProcessor.from_pretrained(variant_id)
        assert isinstance(processor, DepthProcessor)

    def test_s2m2_processor_is_stereo(self):
        processor = AutoProcessor.from_pretrained("s2m2-l")
        assert processor.is_stereo
        assert processor.patch_size == 32
        assert processor.mean == processor.std == [0.5, 0.5, 0.5]

    def test_cannot_instantiate(self):
        with pytest.raises(RuntimeError):
            AutoProcessor()
