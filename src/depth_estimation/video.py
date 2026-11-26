"""
Video and streaming inference for depth estimation.

VideoStream       — iterable wrapper over video files, webcam, or frame globs
StereoVideoStream — the same for stereo: two synchronized sources, or one
                    side-by-side / top-and-bottom packed source
process_video     — read a video, run depth estimation, write output video
"""

import glob as _glob
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .processing_utils import DepthProcessor

logger = logging.getLogger(__name__)

Source = Union[str, int]
StereoSource = Union[Source, Tuple[Source, Source]]

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


class VideoStream:
    """Iterable wrapper over a video source.

    Yields ``(frame_rgb, metadata)`` tuples where ``frame_rgb`` is an
    ``(H, W, 3)`` uint8 RGB array and ``metadata`` is a dict with keys:
    ``frame_index``, ``timestamp_seconds``, ``fps``.

    Supports three source types:

    - **int** — webcam device index (e.g., ``0`` for the default camera).
    - **str (video file)** — path to a video file (.mp4, .avi, .mov, …).
    - **str (glob pattern)** — a pattern like ``"frames/*.png"``; matched
      files are sorted alphabetically and treated as sequential frames.

    Args:
        source: Video source (int, file path, or glob pattern).
        batch_size: Unused internally; stored for use by callers.
        temporal_smoothing: EMA coefficient for smoothing depth across frames.
            0.0 = disabled, 0.9 = heavy smoothing.

    Example::

        stream = VideoStream("video.mp4")
        for frame_rgb, meta in stream:
            print(meta["frame_index"], frame_rgb.shape)
        stream.close()

        # Or as a context manager:
        with VideoStream("frames/*.png") as stream:
            for frame_rgb, meta in stream:
                ...
    """

    def __init__(
        self,
        source: Union[str, int],
        batch_size: int = 1,
        temporal_smoothing: float = 0.0,
    ) -> None:
        self.source = source
        self.batch_size = batch_size
        self.temporal_smoothing = temporal_smoothing

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_paths: Optional[list] = None
        self._prev_depth: Optional[np.ndarray] = None

        self._open_source()

    def _open_source(self) -> None:
        """Detect source type and open it."""
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open webcam device {self.source}")
            return

        if isinstance(self.source, str):
            suffix = Path(self.source).suffix.lower()
            if suffix in _VIDEO_EXTENSIONS:
                self._cap = cv2.VideoCapture(self.source)
                if not self._cap.isOpened():
                    raise FileNotFoundError(
                        f"Could not open video file: {self.source!r}"
                    )
                return

            # Must contain a glob wildcard to be treated as a frame pattern
            if "*" not in self.source and "?" not in self.source:
                raise ValueError(
                    f"source string has an unrecognised extension ({suffix!r}) and contains "
                    "no glob wildcards. Expected a video file path, webcam index, or glob "
                    f"pattern (e.g. 'frames/*.png'). Got: {self.source!r}"
                )
            matched = sorted(_glob.glob(self.source))
            if not matched:
                raise FileNotFoundError(
                    f"No files matched glob pattern: {self.source!r}"
                )
            self._frame_paths = matched
            return

        raise ValueError(
            f"source must be an int (webcam), a video file path, or a glob pattern. Got: {self.source!r}"
        )

    def __iter__(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Yield ``(frame_rgb, metadata)`` for each frame."""
        self._prev_depth = None  # reset EMA state at start of each iteration

        if self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_index = 0
            while True:
                ret, frame_bgr = self._cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                metadata = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 4),
                    "fps": fps,
                }
                yield frame_rgb, metadata
                frame_index += 1
        else:
            # Glob-based frame list — derive FPS from count (assume 30 fps)
            fps = 30.0
            for frame_index, path in enumerate(self._frame_paths):
                frame_bgr = cv2.imread(path)
                if frame_bgr is None:
                    logger.warning("Could not read frame: %s — skipping.", path)
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                metadata = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 4),
                    "fps": fps,
                }
                yield frame_rgb, metadata

    def _temporal_filter(self, depth: np.ndarray, alpha: float) -> np.ndarray:
        """Apply EMA smoothing across frames.

        ``smoothed_t = alpha * prev_t + (1 - alpha) * current_t``

        On the first call ``self._prev_depth`` is ``None`` and the depth is
        returned unchanged.  ``alpha=0.0`` is always a no-op.
        """
        if alpha == 0.0 or self._prev_depth is None:
            self._prev_depth = depth
            return depth
        smoothed = alpha * self._prev_depth + (1.0 - alpha) * depth
        self._prev_depth = smoothed
        return smoothed

    def close(self) -> None:
        """Release the VideoCapture handle if open."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def fps(self) -> float:
        """Frames per second of the source (30.0 for glob sources)."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    @property
    def total_frames(self) -> Optional[int]:
        """Total frame count, or ``None`` for live webcam sources."""
        if self._cap is not None:
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return count if count > 0 else None
        if self._frame_paths is not None:
            return len(self._frame_paths)
        return None


_STEREO_LAYOUTS = ("sbs", "tab")


def split_stereo_frame(frame: np.ndarray, layout: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split a packed stereo frame into ``(left, right)``.

    ``layout="sbs"``: left | right side by side. ``layout="tab"``: left above
    right. An odd trailing row/column is dropped so both views match.
    """
    if layout not in _STEREO_LAYOUTS:
        raise ValueError(f"layout must be one of {_STEREO_LAYOUTS}, got {layout!r}.")
    h, w = frame.shape[:2]
    if layout == "sbs":
        half = w // 2
        return frame[:, :half], frame[:, half : 2 * half]
    half = h // 2
    return frame[:half], frame[half : 2 * half]


class StereoVideoStream:
    """Iterable over stereo frame pairs, for stereo models such as S2M2.

    Yields ``((left_rgb, right_rgb), metadata)``, with the same metadata keys
    as :class:`VideoStream` (taken from the left source).

    Two ways to provide the views:

    - ``source=(left, right)`` — two synchronized sources, each anything
      :class:`VideoStream` accepts (video files, webcam indices or frame
      globs). Iteration stops at the shorter one.
    - ``source=video, layout="sbs"`` (left | right, side by side) or
      ``layout="tab"`` (left above right) — one source with both views packed
      into each frame, as many stereo cameras record.

    Example::

        with StereoVideoStream(("left.mp4", "right.mp4")) as stream:
            for (left, right), meta in stream:
                ...
    """

    def __init__(
        self,
        source: StereoSource,
        layout: Optional[str] = None,
        batch_size: int = 1,
        temporal_smoothing: float = 0.0,
    ) -> None:
        self.source = source
        self.layout = layout
        self.batch_size = batch_size
        self.temporal_smoothing = temporal_smoothing
        self._prev_depth: Optional[np.ndarray] = None

        if isinstance(source, tuple):
            if len(source) != 2:
                raise ValueError(
                    f"Stereo source tuple must be (left, right), got {len(source)} items."
                )
            if layout is not None:
                raise ValueError(
                    "Pass either a (left, right) source tuple or a single packed "
                    "source with layout='sbs'/'tab', not both."
                )
            self._left = VideoStream(source[0])
            try:
                self._right: Optional[VideoStream] = VideoStream(source[1])
            except Exception:
                self._left.close()
                raise
        else:
            if layout not in _STEREO_LAYOUTS:
                raise ValueError(
                    "A single stereo source needs layout='sbs' (side by side) or "
                    "'tab' (top and bottom); for two separate sources pass a "
                    f"(left, right) tuple. Got layout={layout!r}."
                )
            self._left = VideoStream(source)
            self._right = None

    def __iter__(self) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], dict]]:
        self._prev_depth = None
        if self._right is None:
            for frame, meta in self._left:
                yield split_stereo_frame(frame, self.layout), meta
            return
        right_iter = iter(self._right)
        for left, meta in self._left:
            right = next(right_iter, None)
            if right is None:
                logger.warning(
                    "Right source ended before the left one at frame %d; stopping.",
                    meta["frame_index"],
                )
                return
            right_frame = right[0]
            if right_frame.shape != left.shape:
                raise ValueError(
                    f"Left and right frames differ in size at frame "
                    f"{meta['frame_index']}: {left.shape[:2]} vs {right_frame.shape[:2]}."
                )
            yield (left, right_frame), meta

    _temporal_filter = VideoStream._temporal_filter

    def close(self) -> None:
        self._left.close()
        if self._right is not None:
            self._right.close()

    def __enter__(self) -> "StereoVideoStream":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def fps(self) -> float:
        return self._left.fps

    @property
    def total_frames(self) -> Optional[int]:
        counts = [s.total_frames for s in (self._left, self._right) if s is not None]
        if any(c is None for c in counts):
            return None
        return min(counts)


def open_stream(
    source: StereoSource,
    is_stereo: bool = False,
    stereo_layout: Optional[str] = None,
    batch_size: int = 1,
    temporal_smoothing: float = 0.0,
) -> Union[VideoStream, StereoVideoStream]:
    """Open the right stream type for a model.

    Stereo models get a :class:`StereoVideoStream`; monocular models a
    :class:`VideoStream`. Raises a clear error when the source shape does not
    match the model (a single unpacked source for a stereo model, or stereo
    input for a monocular one).
    """
    if is_stereo:
        if not isinstance(source, tuple) and stereo_layout is None:
            raise ValueError(
                "This is a stereo model: pass source=(left, right) with two "
                "synchronized sources, or a single packed source with "
                "stereo_layout='sbs' (side by side) or 'tab' (top and bottom)."
            )
        return StereoVideoStream(
            source,
            layout=stereo_layout,
            batch_size=batch_size,
            temporal_smoothing=temporal_smoothing,
        )
    if isinstance(source, tuple) or stereo_layout is not None:
        raise ValueError(
            "Stereo video input ((left, right) sources or stereo_layout) needs "
            "a stereo model such as s2m2-l; this model is monocular."
        )
    return VideoStream(source, batch_size=batch_size, temporal_smoothing=temporal_smoothing)


def process_video(
    pipeline,
    input_path: StereoSource,
    output_path: str,
    colormap: str = "inferno",
    side_by_side: bool = True,
    fps: Optional[float] = None,
    temporal_smoothing: float = 0.0,
    batch_size: int = 1,
    stereo_layout: Optional[str] = None,
) -> None:
    """Read a video, run depth estimation on each frame, and write an output video.

    Args:
        pipeline: A ``DepthPipeline`` instance.
        input_path: Path to the input video file. For stereo models, a
            ``(left, right)`` tuple of sources, or one packed source with
            ``stereo_layout``.
        output_path: Path for the output video file (must end in ``.mp4`` or ``.avi``).
        colormap: Matplotlib colormap for depth visualization.
        side_by_side: If ``True``, write RGB | depth side-by-side; otherwise
            write depth only.
        fps: Output FPS.  ``None`` matches the input video FPS.
        temporal_smoothing: EMA coefficient for depth smoothing (0.0 = disabled).
        batch_size: Number of frames to process per forward pass.
        stereo_layout: For stereo models with a single packed source:
            ``"sbs"`` (left | right) or ``"tab"`` (left above right). The
            side-by-side output shows the left view next to its depth.
    """
    stream = open_stream(
        input_path,
        is_stereo=getattr(getattr(pipeline.model, "config", None), "is_stereo", False) is True,
        stereo_layout=stereo_layout,
        batch_size=batch_size,
        temporal_smoothing=temporal_smoothing,
    )

    src_fps = fps or stream.fps
    total = stream.total_frames

    # Output writer is lazily initialised once the first frame's size is known.
    out_writer = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        with tqdm(total=total, desc="Processing video", unit="frame") as pbar:
            for frame_rgb, meta in stream:
                # Run depth inference (colorize=False — we'll colorize after EMA)
                result = pipeline(frame_rgb, batch_size=1, colorize=False)
                raw_depth = result.depth  # (H, W) float32 [0, 1]

                # Apply temporal smoothing on raw depth
                smoothed = stream._temporal_filter(raw_depth, temporal_smoothing)

                # Colorize smoothed depth
                colored_depth = DepthProcessor._colorize(smoothed, colormap)

                # Build output frame (the left view for stereo pairs)
                if isinstance(frame_rgb, tuple):
                    frame_rgb = frame_rgb[0]
                colored_bgr = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)

                if side_by_side:
                    rgb_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    out_frame = np.concatenate([rgb_bgr, colored_bgr], axis=1)
                else:
                    out_frame = colored_bgr

                # Initialise writer on first frame (size now known)
                if out_writer is None:
                    out_h, out_w = out_frame.shape[:2]
                    out_writer = cv2.VideoWriter(
                        output_path, fourcc, src_fps, (out_w, out_h)
                    )

                out_writer.write(out_frame)
                pbar.update(1)

    finally:
        stream.close()
        if out_writer is not None:
            out_writer.release()

    logger.info("Video saved to %s", output_path)
