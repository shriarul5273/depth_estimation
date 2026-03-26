"""
DepthPipeline — Highest-level abstraction for depth estimation inference.

Chains model + processor into a single callable.
``pipeline()`` factory function resolves model IDs via the Auto classes.
"""

import logging
import time
from typing import Any, Generator, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .output import DepthOutput
from .modeling_utils import BaseDepthModel, _auto_detect_device
from .processing_utils import DepthProcessor

logger = logging.getLogger(__name__)


class DepthPipeline:
    """End-to-end depth estimation pipeline.

    Usage::

        pipe = DepthPipeline(model=model, processor=processor)
        result = pipe("image.jpg")
        depth_map = result.depth

    Accepts a single image or a list of images (auto-batching).
    """

    def __init__(
        self,
        model: BaseDepthModel,
        processor: DepthProcessor,
        device: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device or _auto_detect_device()
        self.model = self.model.to(self.device).eval()

    def __call__(
        self,
        images: Union[str, Image.Image, np.ndarray, List],
        batch_size: int = 1,
        colorize: bool = True,
        colormap: str = "Spectral_r",
    ) -> Union[DepthOutput, List[DepthOutput]]:
        """Run end-to-end depth estimation.

        Args:
            images: Single image or list. Accepts paths, URLs, PIL, ndarray.
            batch_size: Batch size for processing multiple images.
            colorize: Whether to produce a colored depth visualization.
            colormap: Matplotlib colormap name for visualization.

        Returns:
            DepthOutput for single image, or list of DepthOutput for batch.
        """
        # Normalize to list
        if not isinstance(images, list):
            images = [images]
            single = True
        else:
            single = False

        all_results = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            results = self._process_batch(batch, colorize=colorize, colormap=colormap)
            if isinstance(results, list):
                all_results.extend(results)
            else:
                all_results.append(results)

        if single and len(all_results) == 1:
            return all_results[0]
        return all_results

    def _process_batch(
        self,
        images: List,
        colorize: bool = True,
        colormap: str = "Spectral_r",
    ) -> Union[DepthOutput, List[DepthOutput]]:
        """Process a batch of images."""
        start = time.perf_counter()

        # Preprocess
        inputs = self.processor.preprocess(images)
        pixel_values = inputs["pixel_values"].to(self.device)
        original_sizes = inputs["original_sizes"]

        # Forward pass
        with torch.no_grad():
            depth = self.model(pixel_values)

        elapsed = time.perf_counter() - start

        # Postprocess
        outputs = self.processor.postprocess(
            depth, original_sizes, colorize=colorize, colormap=colormap
        )

        # Inject metadata
        if not isinstance(outputs, list):
            outputs = [outputs]
        for j, out in enumerate(outputs):
            out.metadata.update(
                {
                    "model_type": getattr(self.model.config, "model_type", "unknown"),
                    "backbone": getattr(self.model.config, "backbone", "unknown"),
                    "device": str(self.device),
                    "latency_seconds": round(elapsed / len(outputs), 4),
                    "input_resolution": (
                        pixel_values.shape[-2],
                        pixel_values.shape[-1],
                    ),
                }
            )

        return outputs[0] if len(outputs) == 1 else outputs

    def stream(
        self,
        source: Union[str, int],
        batch_size: int = 1,
        colormap: str = "inferno",
        temporal_smoothing: float = 0.0,
    ) -> Generator[DepthOutput, None, None]:
        """Stream depth results frame-by-frame from a video source.

        Yields a :class:`~depth_estimation.output.DepthOutput` for each frame.
        The ``metadata`` dict includes ``frame_index``, ``timestamp_seconds``,
        and ``fps`` in addition to the standard pipeline keys.

        Args:
            source: Video file path, webcam device index, or glob pattern
                (e.g. ``"frames/*.png"``).
            batch_size: Frames per forward pass.
            colormap: Matplotlib colormap for ``colored_depth``.
            temporal_smoothing: EMA coefficient (0.0 = disabled, 0.9 = heavy).

        Yields:
            :class:`~depth_estimation.output.DepthOutput` per frame.

        Example::

            pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
            for result in pipe.stream("video.mp4", temporal_smoothing=0.5):
                print(result.metadata["frame_index"], result.depth.shape)
        """
        from .video import VideoStream
        from .processing_utils import DepthProcessor

        stream = VideoStream(source, batch_size=batch_size,
                             temporal_smoothing=temporal_smoothing)
        try:
            for frame_rgb, frame_meta in stream:
                # Infer without colorizing — apply EMA first, then colorize
                result = self(frame_rgb, batch_size=1, colorize=False, colormap=colormap)
                smoothed = stream._temporal_filter(result.depth, temporal_smoothing)
                colored = DepthProcessor._colorize(smoothed, colormap)

                result.depth = smoothed
                result.colored_depth = colored
                result.metadata.update(frame_meta)
                yield result
        finally:
            stream.close()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        colormap: str = "inferno",
        side_by_side: bool = True,
        fps: Optional[float] = None,
        temporal_smoothing: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        """Process a video file and write the depth output to disk.

        Delegates to :func:`depth_estimation.video.process_video`.

        Args:
            input_path: Path to the input video file.
            output_path: Destination path for the output video.
            colormap: Matplotlib colormap for depth visualization.
            side_by_side: Write RGB | depth side-by-side when ``True``.
            fps: Output FPS. ``None`` matches the source FPS.
            temporal_smoothing: EMA coefficient (0.0 = disabled).
            batch_size: Frames per forward pass.
        """
        from .video import process_video as _process_video
        _process_video(
            self, input_path, output_path,
            colormap=colormap, side_by_side=side_by_side,
            fps=fps, temporal_smoothing=temporal_smoothing,
            batch_size=batch_size,
        )


def pipeline(
    task: str = "depth-estimation",
    model: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> DepthPipeline:
    """Factory function to create a depth estimation pipeline.

    Usage::

        from depth_estimation import pipeline

        pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
        result = pipe("image.jpg")

    Args:
        task: Task name (must be "depth-estimation").
        model: Model identifier (e.g. "depth-anything-v2-vitb").
        device: Device string. Auto-detected if None.
        **kwargs: Additional arguments passed to model.from_pretrained().

    Returns:
        Configured DepthPipeline.
    """
    if task != "depth-estimation":
        raise ValueError(f"Unsupported task '{task}'. Only 'depth-estimation' is supported.")

    if model is None:
        raise ValueError("You must specify a model identifier (e.g. model='depth-anything-v2-vitb').")

    # Import here to avoid circular imports
    from .models.auto import AutoDepthModel, AutoProcessor

    loaded_model = AutoDepthModel.from_pretrained(model, device=device, **kwargs)
    processor = AutoProcessor.from_pretrained(model)

    return DepthPipeline(model=loaded_model, processor=processor, device=device)
