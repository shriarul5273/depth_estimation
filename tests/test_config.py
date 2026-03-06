"""Tests for config serialization, inheritance, and defaults."""

import pytest
from depth_estimation.configuration_utils import BaseDepthConfig
from depth_estimation.models.depth_anything_v1.configuration_depth_anything_v1 import (
    DepthAnythingV1Config,
    _V1_VARIANT_MAP,
)
from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
    _V2_VARIANT_MAP,
)
from depth_estimation.models.depth_anything_v3.configuration_depth_anything_v3 import (
    DepthAnythingV3Config,
    _V3_VARIANT_MAP,
)
from depth_estimation.models.zoedepth.configuration_zoedepth import (
    ZoeDepthConfig,
    _ZOEDEPTH_VARIANT_MAP,
)
from depth_estimation.models.midas.configuration_midas import (
    MiDaSConfig,
    _MIDAS_VARIANT_MAP,
)
from depth_estimation.models.depth_pro.configuration_depth_pro import (
    DepthProConfig,
    _DEPTHPRO_VARIANT_MAP,
)
from depth_estimation.models.pixel_perfect_depth.configuration_ppd import (
    PixelPerfectDepthConfig,
    _PPD_VARIANT_MAP,
)


class TestBaseDepthConfig:
    def test_defaults(self):
        config = BaseDepthConfig()
        assert config.backbone == "vitl"
        assert config.input_size == 518
        assert config.patch_size == 14
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]

    def test_override(self):
        config = BaseDepthConfig(backbone="vits", input_size=384)
        assert config.backbone == "vits"
        assert config.input_size == 384

    def test_round_trip(self):
        config = BaseDepthConfig(backbone="vitb", features=128, is_metric=True)
        d = config.to_dict()
        config2 = BaseDepthConfig.from_dict(d)
        assert config == config2

    def test_extra_kwargs(self):
        config = BaseDepthConfig(custom_field="hello")
        assert config.custom_field == "hello"
        d = config.to_dict()
        assert d["custom_field"] == "hello"


class TestDepthAnythingV1Config:
    def test_model_type(self):
        config = DepthAnythingV1Config()
        assert config.model_type == "depth-anything-v1"

    @pytest.mark.parametrize("backbone", ["vits", "vitb", "vitl"])
    def test_backbone_defaults(self, backbone):
        config = DepthAnythingV1Config(backbone=backbone)
        assert config.backbone == backbone
        assert config.input_size == 518
        assert len(config.out_channels) == 4

    def test_from_variant(self):
        for variant_id, backbone in _V1_VARIANT_MAP.items():
            config = DepthAnythingV1Config.from_variant(variant_id)
            assert config.backbone == backbone

    def test_from_variant_invalid(self):
        with pytest.raises(ValueError):
            DepthAnythingV1Config.from_variant("nonexistent-model")

    def test_hub_model_id(self):
        config = DepthAnythingV1Config(backbone="vitb")
        assert "depth_anything" in config.hub_model_id

    def test_round_trip(self):
        config = DepthAnythingV1Config(backbone="vits")
        d = config.to_dict()
        config2 = DepthAnythingV1Config.from_dict(d)
        assert config.backbone == config2.backbone
        assert config.features == config2.features


class TestDepthAnythingV2Config:
    def test_model_type(self):
        config = DepthAnythingV2Config()
        assert config.model_type == "depth-anything-v2"

    @pytest.mark.parametrize("backbone,expected_features", [
        ("vits", 64), ("vitb", 128), ("vitl", 256),
    ])
    def test_backbone_defaults(self, backbone, expected_features):
        config = DepthAnythingV2Config(backbone=backbone)
        assert config.features == expected_features

    def test_from_variant(self):
        for variant_id, backbone in _V2_VARIANT_MAP.items():
            config = DepthAnythingV2Config.from_variant(variant_id)
            assert config.backbone == backbone

    def test_intermediate_layer_idx(self):
        config = DepthAnythingV2Config(backbone="vitl")
        assert config.intermediate_layer_idx == [4, 11, 17, 23]

    def test_hub_repo_id(self):
        config = DepthAnythingV2Config(backbone="vitb")
        assert "Depth-Anything-V2" in config.hub_repo_id


class TestDepthAnythingV3Config:
    def test_model_type(self):
        config = DepthAnythingV3Config()
        assert config.model_type == "depth-anything-v3"

    def test_from_variant(self):
        for variant_id, backbone in _V3_VARIANT_MAP.items():
            config = DepthAnythingV3Config.from_variant(variant_id)
            assert config.backbone == backbone

    def test_hub_repo_id(self):
        config = DepthAnythingV3Config(backbone="large")
        assert "DA3" in config.hub_repo_id

    def test_all_variants_have_repos(self):
        for variant_id in _V3_VARIANT_MAP:
            config = DepthAnythingV3Config.from_variant(variant_id)
            assert config.hub_repo_id.startswith("depth-anything/")


class TestZoeDepthConfig:
    def test_model_type(self):
        config = ZoeDepthConfig()
        assert config.model_type == "zoedepth"

    def test_from_variant(self):
        for variant_id in _ZOEDEPTH_VARIANT_MAP:
            config = ZoeDepthConfig.from_variant(variant_id)
            assert config.hf_model_id == "Intel/zoedepth-nyu-kitti"

    def test_is_metric(self):
        config = ZoeDepthConfig()
        assert config.is_metric is True


class TestMiDaSConfig:
    def test_model_type(self):
        config = MiDaSConfig()
        assert config.model_type == "midas"

    @pytest.mark.parametrize("variant_id,expected_hf_id", [
        ("midas-dpt-large", "Intel/dpt-large"),
        ("midas-dpt-hybrid", "Intel/dpt-hybrid-midas"),
        ("midas-beit-large", "Intel/dpt-beit-large-512"),
    ])
    def test_from_variant(self, variant_id, expected_hf_id):
        config = MiDaSConfig.from_variant(variant_id)
        assert config.hf_model_id == expected_hf_id


class TestDepthProConfig:
    def test_model_type(self):
        config = DepthProConfig()
        assert config.model_type == "depth-pro"

    def test_from_variant(self):
        config = DepthProConfig.from_variant("depth-pro")
        assert config.hub_repo_id == "apple/DepthPro"

    def test_is_metric(self):
        config = DepthProConfig()
        assert config.is_metric is True


class TestPixelPerfectDepthConfig:
    def test_model_type(self):
        config = PixelPerfectDepthConfig()
        assert config.model_type == "pixel-perfect-depth"

    def test_from_variant(self):
        config = PixelPerfectDepthConfig.from_variant("pixel-perfect-depth")
        assert config.hub_repo_id == "gangweix/Pixel-Perfect-Depth"

    def test_sampling_steps(self):
        config = PixelPerfectDepthConfig(sampling_steps=10)
        assert config.sampling_steps == 10



class TestMarigoldDCConfig:
    def test_model_type(self):
        from depth_estimation.models.marigold_dc.configuration_marigold_dc import MarigoldDCConfig
        config = MarigoldDCConfig()
        assert config.model_type == "marigold-dc"

    def test_from_variant(self):
        from depth_estimation.models.marigold_dc.configuration_marigold_dc import MarigoldDCConfig
        config = MarigoldDCConfig.from_variant("marigold-dc")
        assert config.hub_model_id == "prs-eth/marigold-depth-v1-0"

    def test_inference_steps(self):
        from depth_estimation.models.marigold_dc.configuration_marigold_dc import MarigoldDCConfig
        config = MarigoldDCConfig(num_inference_steps=10)
        assert config.num_inference_steps == 10
