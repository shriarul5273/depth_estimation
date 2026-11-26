"""Tests for the shared DepthProcessor (preprocess and postprocess)."""

import pytest
import numpy as np
import torch
from PIL import Image

from depth_estimation.processing_utils import DepthProcessor
from depth_estimation.configuration_utils import BaseDepthConfig
from depth_estimation.output import DepthOutput


@pytest.fixture
def processor():
    return DepthProcessor(config=BaseDepthConfig())


@pytest.fixture
def sample_image_rgb():
    """Create a test RGB image as numpy array."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image():
    """Create a test PIL image."""
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestPreprocess:
    def test_numpy_input(self, processor, sample_image_rgb):
        result = processor.preprocess(sample_image_rgb)
        assert "pixel_values" in result
        assert "original_sizes" in result
        assert result["pixel_values"].dim() == 4  # (B, C, H, W)
        assert result["pixel_values"].shape[0] == 1
        assert result["pixel_values"].shape[1] == 3
        assert result["original_sizes"] == [(480, 640)]

    def test_pil_input(self, processor, sample_pil_image):
        result = processor.preprocess(sample_pil_image)
        assert result["pixel_values"].shape[1] == 3

    def test_batch_input(self, processor, sample_image_rgb):
        result = processor.preprocess([sample_image_rgb, sample_image_rgb])
        assert result["pixel_values"].shape[0] == 2
        assert len(result["original_sizes"]) == 2

    def test_output_dimensions_multiple_of_patch_size(
        self, processor, sample_image_rgb
    ):
        result = processor.preprocess(sample_image_rgb)
        h, w = result["pixel_values"].shape[2], result["pixel_values"].shape[3]
        assert h % processor.patch_size == 0
        assert w % processor.patch_size == 0

    def test_callable(self, processor, sample_image_rgb):
        """Processor should be callable (alias for preprocess)."""
        result = processor(sample_image_rgb)
        assert "pixel_values" in result

    def test_normalization(self, processor, sample_image_rgb):
        """Output tensor values should be normalized (not in [0, 255])."""
        result = processor.preprocess(sample_image_rgb)
        vals = result["pixel_values"]
        assert vals.max() < 10.0  # Should be roughly in [-2, 3] range
        assert vals.min() > -5.0

    def test_load_from_url_uses_timeout(self, monkeypatch, sample_pil_image):
        """_load_from_url must not hang forever on a stalled connection."""
        import io
        import urllib.request

        buf = io.BytesIO()
        sample_pil_image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        captured = {}

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return png_bytes

        def _fake_urlopen(url, timeout=None):
            captured["timeout"] = timeout
            return _FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        arr = DepthProcessor._load_from_url("http://example.com/fake.png")
        assert captured["timeout"] == 30
        assert arr.shape == (480, 640, 3)


class TestPostprocess:
    def test_basic(self, processor):
        depth_tensor = torch.rand(1, 37, 37)  # simulated model output
        result = processor.postprocess(depth_tensor, [(480, 640)])
        assert isinstance(result, DepthOutput)
        assert result.depth.shape == (480, 640)
        assert result.depth.dtype == np.float32
        assert result.depth.min() >= 0.0
        assert result.depth.max() <= 1.0

    def test_colored_depth(self, processor):
        depth_tensor = torch.rand(1, 37, 37)
        result = processor.postprocess(depth_tensor, [(480, 640)], colorize=True)
        assert result.colored_depth is not None
        assert result.colored_depth.shape == (480, 640, 3)
        assert result.colored_depth.dtype == np.uint8

    def test_no_colorize(self, processor):
        depth_tensor = torch.rand(1, 37, 37)
        result = processor.postprocess(depth_tensor, [(480, 640)], colorize=False)
        assert result.colored_depth is None

    def test_batch_postprocess(self, processor):
        depth_tensor = torch.rand(3, 37, 37)
        sizes = [(480, 640), (240, 320), (100, 200)]
        results = processor.postprocess(depth_tensor, sizes)
        assert isinstance(results, list)
        assert len(results) == 3
        for i, (r, (h, w)) in enumerate(zip(results, sizes)):
            assert r.depth.shape == (h, w)

    def test_4d_input(self, processor):
        """Postprocess should handle (B, 1, H, W) input."""
        depth_tensor = torch.rand(1, 1, 37, 37)
        result = processor.postprocess(depth_tensor, [(480, 640)])
        assert result.depth.shape == (480, 640)


class TestFromConfig:
    def test_from_config(self):
        config = BaseDepthConfig(input_size=384, patch_size=16)
        processor = DepthProcessor.from_config(config)
        assert processor.input_size == 384
        assert processor.patch_size == 16


class TestKeepAspectRatio:
    """Regression tests: MiDaS dpt-large/dpt-hybrid crash on a non-square
    real image under the default aspect-ratio-preserving resize (confirmed:
    dpt-hybrid raises "Input image size doesn't match model", dpt-large
    raises a patch-grid reshape error) — both hardcode a square patch grid
    internally. keep_aspect_ratio=False makes the processor resize to an
    exact square instead, matching what these backbones require.
    """

    def test_default_preserves_aspect_ratio(self, sample_image_rgb):
        # sample_image_rgb is (480, 640) — not square.
        config = BaseDepthConfig(input_size=384, patch_size=14)
        processor = DepthProcessor.from_config(config)
        assert processor.keep_aspect_ratio is True
        pixel_values = processor.preprocess(sample_image_rgb)["pixel_values"]
        assert pixel_values.shape[2] != pixel_values.shape[3]

    def test_keep_aspect_ratio_false_forces_square(self, sample_image_rgb):
        config = BaseDepthConfig(input_size=384, patch_size=14, keep_aspect_ratio=False)
        processor = DepthProcessor.from_config(config)
        assert processor.keep_aspect_ratio is False
        pixel_values = processor.preprocess(sample_image_rgb)["pixel_values"]
        assert pixel_values.shape[2] == pixel_values.shape[3] == 384

    def test_midas_dpt_large_defaults_to_square(self):
        from depth_estimation.models.midas.configuration_midas import MiDaSConfig

        config = MiDaSConfig(backbone="dpt-large")
        assert config.keep_aspect_ratio is False

    def test_midas_dpt_hybrid_defaults_to_square(self):
        from depth_estimation.models.midas.configuration_midas import MiDaSConfig

        config = MiDaSConfig(backbone="dpt-hybrid")
        assert config.keep_aspect_ratio is False

    def test_midas_beit_large_keeps_aspect_ratio(self):
        """beit-large has no such issue — different HF implementation,
        interpolated position embeddings handle non-square inputs fine.
        """
        from depth_estimation.models.midas.configuration_midas import MiDaSConfig

        config = MiDaSConfig(backbone="beit-large")
        assert config.keep_aspect_ratio is True

    def test_midas_dpt_large_processor_produces_square_input(self, sample_image_rgb):
        from depth_estimation.models.midas.configuration_midas import MiDaSConfig

        config = MiDaSConfig(backbone="dpt-large")
        processor = DepthProcessor.from_config(config)
        pixel_values = processor.preprocess(sample_image_rgb)["pixel_values"]
        assert pixel_values.shape[2] == pixel_values.shape[3] == config.input_size


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_postprocess_accepts_reduced_precision(dtype):
    # NumPy has no bfloat16; postprocess must not call .numpy() on it directly.
    processor = DepthProcessor(config=BaseDepthConfig())
    depth = torch.rand(1, 16, 24).to(dtype)
    out = processor.postprocess(depth, [(32, 48)])
    assert out.depth.dtype == np.float32
    assert out.depth.shape == (32, 48)


class TestStereoPairs:
    """Stereo models (S2M2) take a (left, right) tuple per sample."""

    @pytest.fixture
    def stereo_processor(self):
        from depth_estimation.models.s2m2.configuration_s2m2 import S2M2Config

        return DepthProcessor.from_config(S2M2Config(input_size=64))

    def test_pair_stacks_on_channels(self, stereo_processor):
        left = np.random.randint(0, 255, (48, 80, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (48, 80, 3), dtype=np.uint8)
        out = stereo_processor.preprocess((left, right))
        pv = out["pixel_values"]
        assert pv.shape[:2] == (1, 6)
        assert pv.shape[2] % 32 == 0 and pv.shape[3] % 32 == 0
        assert out["original_sizes"] == [(48, 80)]
        # each half is exactly the single-image transform of that view
        assert torch.equal(pv[:, :3], stereo_processor._transform(left))
        assert torch.equal(pv[:, 3:], stereo_processor._transform(right))

    def test_list_of_pairs_batches(self, stereo_processor):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = stereo_processor.preprocess([(img, img), (img, img)])
        assert out["pixel_values"].shape[:2] == (2, 6)

    def test_single_image_rejected(self, stereo_processor, sample_image_rgb):
        with pytest.raises(TypeError, match="stereo model"):
            stereo_processor.preprocess(sample_image_rgb)

    def test_mismatched_sizes_rejected(self, stereo_processor):
        left = np.zeros((48, 80, 3), dtype=np.uint8)
        right = np.zeros((48, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="same size"):
            stereo_processor.preprocess((left, right))

    def test_monocular_rejects_pair(self, processor, sample_image_rgb):
        with pytest.raises(TypeError, match="monocular"):
            processor.preprocess((sample_image_rgb, sample_image_rgb))
