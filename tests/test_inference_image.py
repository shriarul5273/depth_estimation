"""End-to-end inference smoke test on a synthetic image."""

import numpy as np
import pytest
from PIL import Image

from depth_estimation.pipeline_utils import DepthPipeline
from depth_estimation.processing_utils import DepthProcessor
from depth_estimation.output import DepthOutput
from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
)
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import (
    DepthAnythingV2Model,
)

WIDTH, HEIGHT = 640, 480


@pytest.fixture
def pipe():
    """Pipeline with a v2 vits model (smallest, randomly initialized — no download)."""
    config = DepthAnythingV2Config(backbone="vits")
    model = DepthAnythingV2Model(config)
    processor = DepthProcessor.from_config(config)
    return DepthPipeline(model=model, processor=processor, device="cpu")


@pytest.fixture
def image_array():
    """A random RGB image, standing in for a real photo."""
    return np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def image_path(tmp_path, image_array):
    """Write the random image to disk to exercise path-based loading."""
    path = tmp_path / "image.png"
    Image.fromarray(image_array).save(path)
    return str(path)


def test_inference_from_path(pipe, image_path):
    """Pipeline should run end-to-end given an image file path."""
    result = pipe(image_path)

    assert isinstance(result, DepthOutput)
    assert result.depth.shape == (HEIGHT, WIDTH)
    assert result.colored_depth.shape == (HEIGHT, WIDTH, 3)
    assert result.depth.dtype == np.float32
    assert result.depth.min() >= 0.0
    assert result.depth.max() <= 1.0


def test_inference_from_pil_image(pipe, image_array):
    """Pipeline should also accept an in-memory PIL image."""
    image = Image.fromarray(image_array)
    result = pipe(image)
    assert result.depth.shape == (HEIGHT, WIDTH)


def test_inference_from_ndarray(pipe, image_array):
    """Pipeline should also accept a raw numpy array."""
    result = pipe(image_array)
    assert result.depth.shape == (HEIGHT, WIDTH)


def test_metadata_populated(pipe, image_path):
    result = pipe(image_path)
    assert result.metadata["model_type"]
    assert result.metadata["device"] == "cpu"
    assert result.metadata["latency_seconds"] >= 0.0
