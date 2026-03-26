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

    def test_output_dimensions_multiple_of_patch_size(self, processor, sample_image_rgb):
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
