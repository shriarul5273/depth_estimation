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
)
from depth_estimation.models.depth_pro.configuration_depth_pro import (
    DepthProConfig,
)
from depth_estimation.models.depth_pro_2.configuration_depth_pro_2 import (
    DepthPro2Config,
    _DEPTHPRO2_VARIANT_MAP,
)
from depth_estimation.models.pixel_perfect_depth.configuration_ppd import (
    PixelPerfectDepthConfig,
)
from depth_estimation.models.moge.configuration_moge import (
    MoGeConfig,
    _MOGE_VARIANT_MAP,
)
from depth_estimation.models.vggt.configuration_vggt import (
    VGGTConfig,
    _VGGT_VARIANT_MAP,
)
from depth_estimation.models.omnivggt.configuration_omnivggt import (
    OmniVGGTConfig,
    _OMNIVGGT_VARIANT_MAP,
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

    @pytest.mark.parametrize(
        "backbone,expected_features",
        [
            ("vits", 64),
            ("vitb", 128),
            ("vitl", 256),
        ],
    )
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

    @pytest.mark.parametrize(
        "variant_id,expected_hf_id",
        [
            ("midas-dpt-large", "Intel/dpt-large"),
            ("midas-dpt-hybrid", "Intel/dpt-hybrid-midas"),
            ("midas-beit-large", "Intel/dpt-beit-large-512"),
        ],
    )
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


class TestDepthPro2Config:
    def test_model_type(self):
        config = DepthPro2Config()
        assert config.model_type == "depth-pro-2"
        assert config.is_metric is True

    @pytest.mark.parametrize(
        "variant_id,expected_repo,preset",
        [
            ("depth-pro-2-s", "apple/DepthPro2-S", "vit_small_patch16_dinov3"),
            ("depth-pro-2-b", "apple/DepthPro2-B", "vit_base_patch16_dinov3"),
            ("depth-pro-2-l", "apple/DepthPro2-L", "vit_large_patch16_dinov3.lvd1689m"),
            ("depth-pro-2-h", "apple/DepthPro2-H", "vit_huge_plus_patch16_dinov3"),
        ],
    )
    def test_from_variant(self, variant_id, expected_repo, preset):
        config = DepthPro2Config.from_variant(variant_id)
        assert config.hub_repo_id == expected_repo
        assert config.backbone_preset == preset
        assert config.fov_repo_id == "apple/DepthPro2-focal"
        assert len(config.reassembled_blocks) == 4
        assert len(config.dims_reassemble) == 4

    def test_variant_count(self):
        assert len(_DEPTHPRO2_VARIANT_MAP) == 4

    def test_patch_size_is_input_multiple(self):
        # The network needs inputs that are multiples of 32.
        assert DepthPro2Config().patch_size == 32

    def test_unknown_variant(self):
        with pytest.raises(ValueError, match="Unknown Depth Pro 2 variant"):
            DepthPro2Config.from_variant("depth-pro-2-xl")

    def test_roundtrip(self):
        config = DepthPro2Config.from_variant("depth-pro-2-l")
        restored = DepthPro2Config.from_dict(config.to_dict())
        assert restored == config


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


class TestMoGeConfig:
    def test_model_type(self):
        config = MoGeConfig()
        assert config.model_type == "moge"

    def test_from_variant(self):
        for variant_id in _MOGE_VARIANT_MAP:
            config = MoGeConfig.from_variant(variant_id)
            assert config.hub_repo_id == _MOGE_VARIANT_MAP[variant_id]["hub_repo_id"]

    def test_from_variant_invalid(self):
        with pytest.raises(ValueError):
            MoGeConfig.from_variant("nonexistent-moge")

    @pytest.mark.parametrize(
        "variant_id,expected_metric",
        [
            ("moge-v1", False),
            ("moge-v2-vitl", True),
            ("moge-v2-vits-normal", True),
        ],
    )
    def test_is_metric(self, variant_id, expected_metric):
        config = MoGeConfig.from_variant(variant_id)
        assert config.is_metric is expected_metric

    @pytest.mark.parametrize(
        "variant_id,expected_backbone",
        [
            ("moge-v1", "vitl"),
            ("moge-v2-vitb-normal", "vitb"),
            ("moge-v2-vits-normal", "vits"),
        ],
    )
    def test_backbone(self, variant_id, expected_backbone):
        config = MoGeConfig.from_variant(variant_id)
        assert config.backbone == expected_backbone

    def test_hub_repo_id(self):
        config = MoGeConfig.from_variant("moge-v2-vitl")
        assert "Ruicheng" in config.hub_repo_id


class TestVGGTConfig:
    def test_model_type(self):
        config = VGGTConfig()
        assert config.model_type == "vggt"

    def test_from_variant(self):
        for variant_id in _VGGT_VARIANT_MAP:
            config = VGGTConfig.from_variant(variant_id)
            assert config.hub_repo_id == _VGGT_VARIANT_MAP[variant_id]["hub_repo_id"]

    def test_from_variant_invalid(self):
        with pytest.raises(ValueError):
            VGGTConfig.from_variant("nonexistent-vggt")

    def test_is_metric(self):
        for variant_id in _VGGT_VARIANT_MAP:
            config = VGGTConfig.from_variant(variant_id)
            assert config.is_metric is True

    def test_hub_repo_ids(self):
        assert VGGTConfig.from_variant("vggt").hub_repo_id == "facebook/VGGT-1B"
        assert (
            VGGTConfig.from_variant("vggt-commercial").hub_repo_id
            == "facebook/VGGT-1B-Commercial"
        )

    def test_backbone(self):
        config = VGGTConfig.from_variant("vggt")
        assert config.backbone == "vitl"


class TestOmniVGGTConfig:
    def test_model_type(self):
        config = OmniVGGTConfig()
        assert config.model_type == "omnivggt"

    def test_from_variant(self):
        config = OmniVGGTConfig.from_variant("omnivggt")
        assert config.hub_repo_id == "Livioni/OmniVGGT"

    def test_from_variant_invalid(self):
        with pytest.raises(ValueError):
            OmniVGGTConfig.from_variant("nonexistent-omnivggt")

    def test_is_metric(self):
        config = OmniVGGTConfig.from_variant("omnivggt")
        assert config.is_metric is True

    def test_all_variants_have_repos(self):
        for variant_id in _OMNIVGGT_VARIANT_MAP:
            config = OmniVGGTConfig.from_variant(variant_id)
            assert config.hub_repo_id


class TestMarigoldDCConfig:
    def test_model_type(self):
        from depth_estimation.models.marigold_dc.configuration_marigold_dc import (
            MarigoldDCConfig,
        )

        config = MarigoldDCConfig()
        assert config.model_type == "marigold-dc"

    def test_from_variant(self):
        from depth_estimation.models.marigold_dc.configuration_marigold_dc import (
            MarigoldDCConfig,
        )

        config = MarigoldDCConfig.from_variant("marigold-dc")
        assert config.hub_model_id == "prs-eth/marigold-depth-v1-0"

    def test_inference_steps(self):
        from depth_estimation.models.marigold_dc.configuration_marigold_dc import (
            MarigoldDCConfig,
        )

        config = MarigoldDCConfig(num_inference_steps=10)
        assert config.num_inference_steps == 10


class TestMarigoldV2Config:
    # Folder names that exist in huawei-bayerlab/marigold-v2-0 (its manifest
    # has no "depth/Log-base"; the paper model is "depth/Log-stage2").
    @pytest.mark.parametrize(
        "variant_id,checkpoint",
        [
            ("marigold-v2", "depth/Log-stage2"),
            ("marigold-v2-log-layered", "depth/Log-layered"),
            ("marigold-v2-uniform", "depth/Uniform-base"),
            ("marigold-v2-disparity", "depth/Disparity-base"),
        ],
    )
    def test_checkpoint_folders(self, variant_id, checkpoint):
        from depth_estimation.models.marigold_v2.configuration_marigold_v2 import (
            MarigoldV2Config,
        )

        config = MarigoldV2Config.from_variant(variant_id)
        assert config.checkpoint == checkpoint
        assert config.embed_prefix == "qwen_edit_2509_qwen_depth_realimg512"
