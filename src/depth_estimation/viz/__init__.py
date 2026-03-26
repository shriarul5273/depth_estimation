"""
depth_estimation.viz — Visualization utilities for depth maps.

Functions::

    from depth_estimation.viz import show_depth, compare_depths
    from depth_estimation.viz import overlay_depth, create_anaglyph
    from depth_estimation.viz import animate_3d
    from depth_estimation.viz import plot_error_map

Examples::

    from depth_estimation import pipeline
    from depth_estimation.viz import show_depth, compare_depths, overlay_depth

    pipe = pipeline("depth-estimation", model="depth-anything-v2-vitb")
    result = pipe("image.jpg")

    # Show depth map
    show_depth(result, title="Depth Anything V2")

    # Side-by-side comparison
    result2 = pipeline("depth-estimation", model="midas-dpt-large")("image.jpg")
    compare_depths([result, result2], labels=["DA V2", "MiDaS"], save="compare.png")

    # Overlay on original image
    import numpy as np
    import cv2
    image = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
    overlay = overlay_depth(image, result.depth, alpha=0.5)
"""

from .display import show_depth, compare_depths
from .composite import overlay_depth, create_anaglyph
from .animate import animate_3d
from .error import plot_error_map

__all__ = [
    "show_depth",
    "compare_depths",
    "overlay_depth",
    "create_anaglyph",
    "animate_3d",
    "plot_error_map",
]
