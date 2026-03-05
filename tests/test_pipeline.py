"""Tests for the pipeline factory and end-to-end inference."""

import pytest
import numpy as np
import torch

from depth_estimation.pipeline_utils import DepthPipeline, pipeline as pipeline_fn
from depth_estimation.processing_utils import DepthProcessor
from depth_estimation.output import DepthOutput
from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import DepthAnythingV2Config
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import DepthAnythingV2Model


class TestDepthPipeline:
    @pytest.fixture
    def pipe(self):
        """Create a pipeline with a v2 vits model (smallest, no download needed)."""
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config)
        processor = DepthProcessor.from_config(config)
        return DepthPipeline(model=model, processor=processor, device="cpu")

    def test_single_image(self, pipe):
        """Pipeline should process a single numpy image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe(image)
        assert isinstance(result, DepthOutput)
        assert result.depth.shape == (480, 640)
        assert result.colored_depth.shape == (480, 640, 3)

    def test_batch(self, pipe):
        """Pipeline should process a batch of images."""
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
        ]
        results = pipe(images)
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].depth.shape == (480, 640)
        assert results[1].depth.shape == (240, 320)

    def test_metadata(self, pipe):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe(image)
        assert "model_type" in result.metadata
        assert "device" in result.metadata
        assert "latency_seconds" in result.metadata

    def test_no_colorize(self, pipe):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe(image, colorize=False)
        assert result.colored_depth is None

    def test_depth_range(self, pipe):
        """Depth output should be normalized to [0, 1]."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe(image)
        assert result.depth.min() >= 0.0
        assert result.depth.max() <= 1.0


class TestPipelineFactory:
    def test_invalid_task(self):
        with pytest.raises(ValueError, match="Unsupported task"):
            pipeline_fn("image-classification", model="depth-anything-v2-vitb")

    def test_no_model(self):
        with pytest.raises(ValueError, match="must specify"):
            pipeline_fn("depth-estimation")
