"""
DepthEstimationPipeline — Concrete pipeline for the "depth-estimation" task.
"""

from ..pipeline_utils import DepthPipeline


class DepthEstimationPipeline(DepthPipeline):
    """Pipeline for monocular depth estimation.

    This is the concrete implementation of the DepthPipeline for the
    "depth-estimation" task. It chains model + processor for end-to-end inference.

    Usage::

        from depth_estimation import pipeline

        pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
        result = pipe("image.jpg")
        depth_map = result.depth
    """

    pass
