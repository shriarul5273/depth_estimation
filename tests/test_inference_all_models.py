"""Inference smoke tests across every registered model family.

Three tiers, split by cost:

- ``TestFastOffline``: variants that build real randomly-initialized weights
  fully offline (no network, no download). Cheap — runs on every commit/CI.
- ``TestSlowOffline`` (``@pytest.mark.slow``): also offline/no-download, but
  architecturally heavy (large ViT-L backbones, diffusion sampling), so
  excluded from the default CI run.
- ``TestPretrainedVariants`` (``@pytest.mark.slow``): every registered model
  variant (28+) run through the real ``pipeline()`` factory, which downloads
  actual pretrained weights from the Hugging Face Hub. Needs network and
  disk space, so it's opt-in only — run explicitly with ``pytest -m slow``.
"""

import numpy as np
import pytest

from depth_estimation.pipeline_utils import DepthPipeline
from depth_estimation.processing_utils import DepthProcessor
from depth_estimation.output import DepthOutput
from depth_estimation.registry import MODEL_REGISTRY

from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
)
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import (
    DepthAnythingV2Model,
)
from depth_estimation.models.depth_anything_v3.configuration_depth_anything_v3 import (
    DepthAnythingV3Config,
)
from depth_estimation.models.depth_anything_v3.modeling_depth_anything_v3 import (
    DepthAnythingV3Model,
)
from depth_estimation.models.depth_pro.configuration_depth_pro import DepthProConfig
from depth_estimation.models.depth_pro.modeling_depth_pro import DepthProModel
from depth_estimation.models.pixel_perfect_depth.configuration_ppd import (
    PixelPerfectDepthConfig,
)
from depth_estimation.models.pixel_perfect_depth.modeling_ppd import (
    PixelPerfectDepthModel,
)

WIDTH, HEIGHT = 224, 224


def _random_image():
    return np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _assert_valid_output(result):
    assert isinstance(result, DepthOutput)
    assert result.depth.shape == (HEIGHT, WIDTH)
    assert result.colored_depth.shape == (HEIGHT, WIDTH, 3)
    assert not np.isnan(result.depth).any()
    assert result.depth.min() >= 0.0
    assert result.depth.max() <= 1.0


def _run_pipeline(config, model):
    processor = DepthProcessor.from_config(config)
    pipe = DepthPipeline(model=model, processor=processor, device="cpu")
    return pipe(_random_image())


class TestFastOffline:
    """Random-weight, no-download models. Runs on every CI push."""

    @pytest.mark.parametrize("backbone", ["vits", "vitb", "vitl"])
    def test_depth_anything_v2(self, backbone):
        config = DepthAnythingV2Config(backbone=backbone)
        model = DepthAnythingV2Model(config)
        _assert_valid_output(_run_pipeline(config, model))

    def test_depth_anything_v3_small(self):
        config = DepthAnythingV3Config(backbone="small")
        model = DepthAnythingV3Model(config)
        _assert_valid_output(_run_pipeline(config, model))


@pytest.mark.slow
class TestSlowOffline:
    """Random-weight, no-download, but architecturally heavy models."""

    def test_depth_pro(self):
        # DepthPro lazily builds ~1B params of random weights on first
        # forward() — device="cpu" below (via BaseDepthModel.device) keeps
        # this deterministic and off a shared GPU.
        config = DepthProConfig()
        model = DepthProModel(config)
        _assert_valid_output(_run_pipeline(config, model))

    def test_pixel_perfect_depth(self):
        # dit_hidden_size must stay 1024 (proj_fusion concatenates it with
        # the fixed 1024-dim ViT-L semantics tokens) and dit_depth must stay
        # >= 12 (the token-upsample step is hardcoded at block index 11) —
        # only input_size and sampling_steps are safe to shrink.
        config = PixelPerfectDepthConfig(input_size=256, sampling_steps=2)
        model = PixelPerfectDepthModel(config)
        _assert_valid_output(_run_pipeline(config, model))


@pytest.mark.slow
class TestPretrainedVariants:
    """Every registered variant with real pretrained weights (network required)."""

    @pytest.mark.parametrize("variant_id", MODEL_REGISTRY.list_variants())
    def test_variant_inference(self, variant_id):
        from depth_estimation import pipeline

        pipe = pipeline("depth-estimation", model=variant_id, device="cpu")
        result = pipe(_random_image())
        assert isinstance(result, DepthOutput)
        assert result.depth.ndim == 2
        assert not np.isnan(result.depth).any()
