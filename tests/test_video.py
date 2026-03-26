"""Tests for depth_estimation.video (VideoStream and process_video)."""

import glob
import unittest.mock as mock
from pathlib import Path

import cv2
import numpy as np
import pytest

from depth_estimation.output import DepthOutput
from depth_estimation.video import VideoStream, process_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 48, 64


def _make_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((H, W, 3)) * 255).astype(np.uint8)


def _write_png_frames(directory: Path, n: int = 3):
    """Write n synthetic PNG frames and return the glob pattern."""
    for i in range(n):
        frame_bgr = cv2.cvtColor(_make_frame(i), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(directory / f"frame_{i:04d}.png"), frame_bgr)
    return str(directory / "frame_*.png")


def _write_video(path: Path, n_frames: int = 5, fps: float = 10.0) -> None:
    """Write a minimal synthetic video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for i in range(n_frames):
        frame_bgr = cv2.cvtColor(_make_frame(i), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()


# ---------------------------------------------------------------------------
# VideoStream — glob source
# ---------------------------------------------------------------------------

class TestVideoStreamGlob:
    def test_iterates_all_frames(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=3)
        frames = list(VideoStream(pattern))
        assert len(frames) == 3

    def test_metadata_keys(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=2)
        for frame_rgb, meta in VideoStream(pattern):
            assert "frame_index" in meta
            assert "timestamp_seconds" in meta
            assert "fps" in meta

    def test_frame_indices_sequential(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=4)
        indices = [meta["frame_index"] for _, meta in VideoStream(pattern)]
        assert indices == list(range(4))

    def test_frame_rgb_shape(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=1)
        for frame_rgb, _ in VideoStream(pattern):
            assert frame_rgb.ndim == 3
            assert frame_rgb.shape[2] == 3
            assert frame_rgb.dtype == np.uint8

    def test_context_manager(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=2)
        with VideoStream(pattern) as vs:
            frames = list(vs)
        assert len(frames) == 2

    def test_total_frames(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=5)
        vs = VideoStream(pattern)
        assert vs.total_frames == 5
        vs.close()

    def test_empty_glob_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files matched"):
            VideoStream(str(tmp_path / "nonexistent_*.png"))

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            VideoStream("not_a_valid_source.xyz")


# ---------------------------------------------------------------------------
# VideoStream — video file source
# ---------------------------------------------------------------------------

class TestVideoStreamFile:
    def test_iterates_frames(self, tmp_path):
        vid = tmp_path / "test.mp4"
        _write_video(vid, n_frames=5)
        frames = list(VideoStream(str(vid)))
        assert len(frames) == 5

    def test_metadata_has_fps(self, tmp_path):
        vid = tmp_path / "test.mp4"
        _write_video(vid, n_frames=3, fps=10.0)
        for _, meta in VideoStream(str(vid)):
            assert meta["fps"] > 0

    def test_close_releases(self, tmp_path):
        vid = tmp_path / "test.mp4"
        _write_video(vid, n_frames=2)
        vs = VideoStream(str(vid))
        vs.close()
        # After close, _cap should be None
        assert vs._cap is None

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoStream(str(tmp_path / "ghost.mp4"))


# ---------------------------------------------------------------------------
# VideoStream — temporal filter
# ---------------------------------------------------------------------------

class TestTemporalFilter:
    def test_alpha_zero_is_identity(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=1)
        vs = VideoStream(pattern, temporal_smoothing=0.0)
        depth = np.ones((H, W), dtype=np.float32) * 0.5
        result = vs._temporal_filter(depth, alpha=0.0)
        np.testing.assert_array_equal(result, depth)
        vs.close()

    def test_first_call_returns_unchanged(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=1)
        vs = VideoStream(pattern)
        depth = np.full((H, W), 0.7, dtype=np.float32)
        result = vs._temporal_filter(depth, alpha=0.9)
        np.testing.assert_array_equal(result, depth)
        vs.close()

    def test_ema_math(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=2)
        vs = VideoStream(pattern)
        d0 = np.full((H, W), 0.2, dtype=np.float32)
        d1 = np.full((H, W), 0.8, dtype=np.float32)
        alpha = 0.5
        vs._temporal_filter(d0, alpha)        # initialises _prev_depth = d0
        result = vs._temporal_filter(d1, alpha)
        expected = alpha * d0 + (1 - alpha) * d1
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        vs.close()


# ---------------------------------------------------------------------------
# process_video
# ---------------------------------------------------------------------------

class TestProcessVideo:
    def test_writes_output_file(self, tmp_path):
        vid_in = tmp_path / "in.mp4"
        vid_out = tmp_path / "out.mp4"
        _write_video(vid_in, n_frames=5)

        # Create a mock pipeline that returns a synthetic DepthOutput
        depth_arr = np.random.default_rng(0).random((H, W)).astype(np.float32)
        fake_result = DepthOutput(depth=depth_arr)
        mock_pipe = mock.MagicMock(return_value=fake_result)

        process_video(mock_pipe, str(vid_in), str(vid_out))
        assert vid_out.exists()
        assert vid_out.stat().st_size > 0

    def test_depth_only_output(self, tmp_path):
        vid_in = tmp_path / "in.mp4"
        vid_out = tmp_path / "out.mp4"
        _write_video(vid_in, n_frames=3)

        depth_arr = np.random.default_rng(0).random((H, W)).astype(np.float32)
        fake_result = DepthOutput(depth=depth_arr)
        mock_pipe = mock.MagicMock(return_value=fake_result)

        process_video(mock_pipe, str(vid_in), str(vid_out), side_by_side=False)
        assert vid_out.exists()

    def test_side_by_side_double_width(self, tmp_path):
        """When side_by_side=True, written frame width should be 2 × input width."""
        vid_in = tmp_path / "in.mp4"
        vid_out = tmp_path / "out.mp4"
        _write_video(vid_in, n_frames=3)

        depth_arr = np.random.default_rng(0).random((H, W)).astype(np.float32)
        fake_result = DepthOutput(depth=depth_arr)
        mock_pipe = mock.MagicMock(return_value=fake_result)

        process_video(mock_pipe, str(vid_in), str(vid_out), side_by_side=True)

        cap = cv2.VideoCapture(str(vid_out))
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        assert out_w == 2 * W


# ---------------------------------------------------------------------------
# DepthPipeline.stream integration (lightweight — no real model)
# ---------------------------------------------------------------------------

class TestDepthPipelineStream:
    def test_stream_yields_depth_output(self, tmp_path):
        pattern = _write_png_frames(tmp_path, n=3)

        depth_arr = np.random.default_rng(0).random((H, W)).astype(np.float32)
        fake_result = DepthOutput(depth=depth_arr)

        # Build a minimal mock pipeline with a stream() method
        from depth_estimation.pipeline_utils import DepthPipeline
        mock_model = mock.MagicMock()
        mock_processor = mock.MagicMock()
        mock_processor.preprocess.return_value = {
            "pixel_values": __import__("torch").zeros(1, 3, H, W),
            "original_sizes": [(H, W)],
        }
        mock_processor.postprocess.return_value = fake_result

        import torch
        mock_model.return_value = torch.zeros(1, H, W)
        mock_model.config = mock.MagicMock()
        mock_model.config.model_type = "mock"
        mock_model.config.backbone = "mock"
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        pipe = DepthPipeline(model=mock_model, processor=mock_processor)
        results = list(pipe.stream(pattern))

        assert len(results) == 3
        for r in results:
            assert isinstance(r, DepthOutput)
            assert "frame_index" in r.metadata
