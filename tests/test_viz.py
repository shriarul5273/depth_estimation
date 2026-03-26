"""Tests for depth_estimation.viz visualization utilities."""

import numpy as np
import pytest

from depth_estimation.output import DepthOutput
from depth_estimation.processing_utils import DepthProcessor
from depth_estimation.viz import (
    animate_3d,
    compare_depths,
    create_anaglyph,
    overlay_depth,
    plot_error_map,
    show_depth,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

H, W = 64, 80


@pytest.fixture
def depth_arr():
    rng = np.random.default_rng(0)
    return rng.random((H, W)).astype(np.float32)


@pytest.fixture
def image_arr():
    rng = np.random.default_rng(1)
    return (rng.random((H, W, 3)) * 255).astype(np.uint8)


@pytest.fixture
def depth_output(depth_arr):
    return DepthOutput(depth=depth_arr)


@pytest.fixture
def depth_output_with_color(depth_arr):
    colored = DepthProcessor._colorize(depth_arr, "Spectral_r")
    return DepthOutput(depth=depth_arr, colored_depth=colored)


# ---------------------------------------------------------------------------
# show_depth
# ---------------------------------------------------------------------------

class TestShowDepth:
    def test_accepts_depth_output(self, tmp_path, depth_output):
        show_depth(depth_output, save=str(tmp_path / "out.png"))
        assert (tmp_path / "out.png").exists()

    def test_accepts_depth_output_with_color(self, tmp_path, depth_output_with_color):
        show_depth(depth_output_with_color, save=str(tmp_path / "out.png"))
        assert (tmp_path / "out.png").exists()

    def test_accepts_ndarray(self, tmp_path, depth_arr):
        show_depth(depth_arr, save=str(tmp_path / "out.png"))
        assert (tmp_path / "out.png").exists()

    def test_with_title(self, tmp_path, depth_output):
        show_depth(depth_output, title="Test", save=str(tmp_path / "out.png"))
        assert (tmp_path / "out.png").exists()

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            show_depth("not_a_valid_input", save="/dev/null")


# ---------------------------------------------------------------------------
# compare_depths
# ---------------------------------------------------------------------------

class TestCompareDepths:
    def test_multiple_depth_outputs(self, tmp_path, depth_output):
        compare_depths([depth_output, depth_output, depth_output], save=str(tmp_path / "cmp.png"))
        assert (tmp_path / "cmp.png").exists()

    def test_mixed_types(self, tmp_path, depth_output, depth_arr):
        compare_depths([depth_output, depth_arr], labels=["A", "B"], save=str(tmp_path / "cmp.png"))
        assert (tmp_path / "cmp.png").exists()

    def test_label_count_mismatch_raises(self, depth_output):
        with pytest.raises(ValueError, match=r"len\(labels\)"):
            compare_depths([depth_output, depth_output], labels=["only_one"])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            compare_depths([])


# ---------------------------------------------------------------------------
# overlay_depth
# ---------------------------------------------------------------------------

class TestOverlayDepth:
    def test_output_shape(self, image_arr, depth_arr):
        result = overlay_depth(image_arr, depth_arr)
        assert result.shape == (H, W, 3)

    def test_output_dtype(self, image_arr, depth_arr):
        result = overlay_depth(image_arr, depth_arr)
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_image(self, image_arr, depth_arr):
        result = overlay_depth(image_arr, depth_arr, alpha=0.0)
        np.testing.assert_array_equal(result, image_arr)

    def test_alpha_one_returns_colormap(self, image_arr, depth_arr):
        result = overlay_depth(image_arr, depth_arr, alpha=1.0)
        expected = DepthProcessor._colorize(depth_arr, "inferno")
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# create_anaglyph
# ---------------------------------------------------------------------------

class TestCreateAnaglyph:
    def test_output_shape(self, image_arr, depth_arr):
        result = create_anaglyph(image_arr, depth_arr)
        assert result.shape == (H, W, 3)

    def test_output_dtype(self, image_arr, depth_arr):
        result = create_anaglyph(image_arr, depth_arr)
        assert result.dtype == np.uint8

    def test_zero_baseline(self, image_arr, depth_arr):
        # Zero baseline → no shift; right-eye channels equal original image channels
        result = create_anaglyph(image_arr, depth_arr, baseline=0.0)
        assert result.shape == (H, W, 3)


# ---------------------------------------------------------------------------
# animate_3d
# ---------------------------------------------------------------------------

class TestAnimate3D:
    def test_gif_created(self, tmp_path, image_arr, depth_arr):
        out = str(tmp_path / "out.gif")
        animate_3d(image_arr, depth_arr, out, frames=4, fps=4)
        assert (tmp_path / "out.gif").exists()
        assert (tmp_path / "out.gif").stat().st_size > 0

    def test_invalid_extension_raises(self, tmp_path, image_arr, depth_arr):
        with pytest.raises(ValueError, match="gif or .mp4"):
            animate_3d(image_arr, depth_arr, str(tmp_path / "out.avi"), frames=2)


# ---------------------------------------------------------------------------
# plot_error_map
# ---------------------------------------------------------------------------

class TestPlotErrorMap:
    def test_abs_rel_smoke(self, tmp_path, depth_arr):
        pred = depth_arr
        gt = depth_arr * 0.9 + 0.05
        plot_error_map(pred, gt, metric="abs_rel", save=str(tmp_path / "err.png"))
        assert (tmp_path / "err.png").exists()

    @pytest.mark.parametrize("metric", ["abs_rel", "sq_rel", "log10", "rmse"])
    def test_all_metrics(self, tmp_path, depth_arr, metric):
        gt = depth_arr * 0.8 + 0.1
        plot_error_map(depth_arr, gt, metric=metric, save=str(tmp_path / f"{metric}.png"))
        assert (tmp_path / f"{metric}.png").exists()

    def test_invalid_metric_raises(self, depth_arr):
        with pytest.raises(ValueError, match="metric must be one of"):
            plot_error_map(depth_arr, depth_arr, metric="bad_metric")

    def test_gt_zeros_handled(self, tmp_path):
        pred = np.ones((H, W), dtype=np.float32) * 0.5
        gt = np.zeros((H, W), dtype=np.float32)  # all invalid
        plot_error_map(pred, gt, metric="abs_rel", save=str(tmp_path / "zeros.png"))
        assert (tmp_path / "zeros.png").exists()
