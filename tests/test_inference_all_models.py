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
from depth_estimation.models.depth_pro_2.configuration_depth_pro_2 import (
    DepthPro2Config,
)
from depth_estimation.models.depth_pro_2.modeling_depth_pro_2 import (
    DepthPro2Model,
)
from depth_estimation.models.s2m2.configuration_s2m2 import S2M2Config
from depth_estimation.models.s2m2.modeling_s2m2 import S2M2Model
from depth_estimation.models.pixel_perfect_depth.configuration_ppd import (
    PixelPerfectDepthConfig,
)
from depth_estimation.models.pixel_perfect_depth.modeling_ppd import (
    PixelPerfectDepthModel,
)

WIDTH, HEIGHT = 224, 224
# Deliberately non-square — every model in this suite used to only ever be
# exercised with a square image (WIDTH == HEIGHT above), which is exactly
# why the MiDaS dpt-large/dpt-hybrid non-square crash and the
# dynamic_spatial silent-wrong-output bug (both confirmed against real
# non-square photos, not synthetic ones) went uncaught for so long. This
# doesn't need real weights or network, so it belongs in the fast tier —
# it catches preprocessing/shape-handling bugs generically, even though it
# can't catch model-weight-dependent issues like quantization accuracy.
NONSQUARE_WIDTH, NONSQUARE_HEIGHT = 336, 224


def _random_image():
    return np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _random_nonsquare_image():
    return np.random.randint(
        0, 255, (NONSQUARE_HEIGHT, NONSQUARE_WIDTH, 3), dtype=np.uint8
    )


def _assert_valid_output(result, height=HEIGHT, width=WIDTH):
    assert isinstance(result, DepthOutput)
    assert result.depth.shape == (height, width)
    assert result.colored_depth.shape == (height, width, 3)
    assert not np.isnan(result.depth).any()
    assert result.depth.min() >= 0.0
    assert result.depth.max() <= 1.0


def _run_pipeline(config, model, image=None):
    processor = DepthProcessor.from_config(config)
    pipe = DepthPipeline(model=model, processor=processor, device="cpu")
    return pipe(image if image is not None else _random_image())


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

    @pytest.mark.parametrize("backbone", ["vits", "vitb", "vitl"])
    def test_depth_anything_v2_nonsquare_image(self, backbone):
        config = DepthAnythingV2Config(backbone=backbone)
        model = DepthAnythingV2Model(config)
        result = _run_pipeline(config, model, image=_random_nonsquare_image())
        _assert_valid_output(result, height=NONSQUARE_HEIGHT, width=NONSQUARE_WIDTH)

    def test_depth_anything_v3_small_nonsquare_image(self):
        config = DepthAnythingV3Config(backbone="small")
        model = DepthAnythingV3Model(config)
        result = _run_pipeline(config, model, image=_random_nonsquare_image())
        _assert_valid_output(result, height=NONSQUARE_HEIGHT, width=NONSQUARE_WIDTH)


    def test_depth_pro_2_small(self):
        # ViT-S + the ViT-S focal estimator (~48M params); input_size shrunk
        # to keep the random-weight forward cheap.
        config = DepthPro2Config(input_size=128)
        model = DepthPro2Model(config)
        _assert_valid_output(_run_pipeline(config, model))

    def test_depth_pro_2_small_nonsquare_image(self):
        config = DepthPro2Config(input_size=128)
        model = DepthPro2Model(config)
        result = _run_pipeline(config, model, image=_random_nonsquare_image())
        _assert_valid_output(result, height=NONSQUARE_HEIGHT, width=NONSQUARE_WIDTH)

    def test_depth_pro_2_infer_clip_and_padding(self):
        # 100x140 is not a multiple of 32: infer() must pad and crop back.
        import torch

        model = DepthPro2Model(DepthPro2Config()).eval()
        clip = torch.randn(3, 3, 100, 140)
        out = model.infer(clip, f_px=120.0)
        assert out["inverse_depth"].shape == (3, 100, 140)
        assert out["depth"].shape == (3, 100, 140)
        single = model.infer(clip[0])
        assert single["depth"].shape == (100, 140)
        assert single["focallength_px"].ndim == 0

    def test_depth_pro_2_old_timm_raises_clear_error(self, monkeypatch):
        # timm < 1.0.20 has no DINOv3 models; create_model raises
        # RuntimeError("Unknown model ...") there.
        import timm

        def _unknown(name, **kwargs):
            raise RuntimeError(f"Unknown model ({name})")

        monkeypatch.setattr(timm, "create_model", _unknown)
        with pytest.raises(ImportError, match=r"timm>=1\.0\.20"):
            DepthPro2Model(DepthPro2Config())

    def test_depth_pro_2_training_helpers(self):
        import torch

        model = DepthPro2Model(DepthPro2Config())
        model.freeze_backbone()
        assert not any(p.requires_grad for p in model.encoder.parameters())
        model.unfreeze_top_k_backbone_layers(2)
        assert all(p.requires_grad for p in model.encoder.blocks[-1].parameters())
        assert not any(p.requires_grad for p in model.encoder.blocks[0].parameters())
        groups = model.get_parameter_groups()
        assert groups[0]["params"] and groups[1]["params"]

        model.train()
        depth = model(torch.randn(1, 3, 64, 96))
        assert depth.shape == (1, 64, 96)
        depth.sum().backward()
        assert model.head[0].weight.grad is not None


    # S2M2 (stereo) -------------------------------------------------------

    def test_s2m2_small_pair(self):
        config = S2M2Config(input_size=96)
        model = S2M2Model(config)
        pair = (_random_image(), _random_image())
        _assert_valid_output(_run_pipeline(config, model, image=pair))

    def test_s2m2_small_nonsquare_pair(self):
        config = S2M2Config(input_size=96)
        model = S2M2Model(config)
        pair = (_random_nonsquare_image(), _random_nonsquare_image())
        result = _run_pipeline(config, model, image=pair)
        _assert_valid_output(result, height=NONSQUARE_HEIGHT, width=NONSQUARE_WIDTH)

    def test_s2m2_rejects_single_image(self):
        config = S2M2Config(input_size=96)
        with pytest.raises(TypeError, match="stereo model"):
            _run_pipeline(config, S2M2Model(config))
        import torch

        with pytest.raises(ValueError, match="stereo model"):
            S2M2Model(config)(torch.randn(1, 3, 64, 64))

    def test_s2m2_infer_padding_and_outputs(self):
        # 70x100 is not a multiple of 32: infer() must pad and crop back.
        model = S2M2Model(S2M2Config()).eval()
        left = np.random.randint(0, 255, (70, 100, 3), dtype=np.uint8)
        out = model.infer(left, left.copy())
        for key in ("disparity", "occlusion", "confidence"):
            assert out[key].shape == (70, 100), key
            assert out[key].dtype.is_floating_point
        assert out["depth"] is None
        assert (out["disparity"] >= 0).all()  # positivity unless allow_negative

    def test_s2m2_infer_batched_tensor(self):
        import torch

        model = S2M2Model(S2M2Config()).eval()
        x = torch.rand(2, 3, 64, 96) * 2 - 1
        out = model.infer(x, x)
        assert out["disparity"].shape == (2, 64, 96)

    def test_s2m2_disparity_to_depth(self, monkeypatch):
        import torch

        model = S2M2Model(S2M2Config()).eval()
        disp = torch.tensor([[[10.0, 0.0], [20.0, -1.0]]])
        monkeypatch.setattr(
            model,
            "predict_stereo",
            lambda left, right: {
                "disparity": disp,
                "occlusion": torch.ones_like(disp),
                "confidence": torch.ones_like(disp),
            },
        )
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = model.infer(img, img, focal_length_px=100.0, baseline=0.5)["depth"]
        assert depth[0, 0].item() == pytest.approx(5.0)
        assert depth[1, 0].item() == pytest.approx(2.5)
        assert torch.isinf(depth[0, 1]) and torch.isinf(depth[1, 1])
        # doffs shifts the denominator (Middlebury convention)
        depth = model.infer(img, img, focal_length_px=100.0, baseline=0.5, doffs=10.0)["depth"]
        assert depth[0, 0].item() == pytest.approx(2.5)

    def test_s2m2_pad_fill_matches_upstream(self):
        # Upstream zero-pads raw [0, 255] pixels; after [-1, 1] normalisation
        # that is -1, so a black image must stay uniformly black when padded.
        import torch
        from depth_estimation.models.s2m2.modeling_s2m2 import _image_crop, _image_pad

        black = torch.full((1, 3, 50, 70), -1.0)
        padded = _image_pad(black, torch.full((1, 3, 1, 1), -1.0))
        assert padded.shape == (1, 3, 64, 96)
        assert torch.allclose(padded, torch.full_like(padded, -1.0))
        assert torch.equal(_image_crop(padded, (50, 70)), black)

    def test_s2m2_strict_loading_drops_training_keys(self, monkeypatch):
        import huggingface_hub
        import torch

        state = S2M2Model(S2M2Config()).net.state_dict()
        extra = dict(state)
        extra["feature_flow_attn.q_proj.0.weight"] = torch.zeros(1)
        extra["upsample_mask_4x.conv_x.weight"] = torch.zeros(1)
        extra["disp_init.layer_norm1.weight"] = torch.zeros(1)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **k: "unused")
        monkeypatch.setattr(torch, "load", lambda *a, **k: {"state_dict": extra})
        model = S2M2Model.from_pretrained("s2m2-s", device="cpu")
        assert not model.training

        del extra["refiner.gru.convz1.weight"]
        with pytest.raises(RuntimeError, match="Missing key"):
            S2M2Model.from_pretrained("s2m2-s", device="cpu")


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

        try:
            pipe = pipeline("depth-estimation", model=variant_id, device="cpu")
        except Exception as e:
            # vggt-commercial's repo is gated on Hugging Face (confirmed:
            # raises huggingface_hub.errors.GatedRepoError, a 401) —
            # downloading it requires a logged-in huggingface-hub session
            # that has already accepted the repo's license, which this CI
            # runner doesn't have configured. Not a code bug — see
            # docs/models.md's footnote on this variant. Skip rather than
            # fail so this one credential-gated variant doesn't mask a real
            # regression in the other 27.
            if "gated" in str(e).lower() or "GatedRepoError" in type(e).__name__:
                pytest.skip(
                    f"{variant_id} is a gated Hugging Face repo — requires "
                    f"authenticated access with the repo's license accepted, "
                    f"not configured in this environment: {e}"
                )
            raise
        if getattr(pipe.model.config, "is_stereo", False):
            result = pipe((_random_image(), _random_image()))
        else:
            result = pipe(_random_image())
        assert isinstance(result, DepthOutput)
        assert result.depth.ndim == 2
        assert not np.isnan(result.depth).any()
