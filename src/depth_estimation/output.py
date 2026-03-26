"""
DepthOutput — Standard output dataclass for all depth estimation models.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DepthOutput:
    """Standard output returned by all depth estimation inference paths.

    Attributes:
        depth: Raw depth map, shape (H, W), float32.
            Normalized to [0, 1] for relative depth, or in meters for metric depth.
        colored_depth: Colormapped RGB visualization, shape (H, W, 3), uint8.
            None if colorization was disabled.
        metadata: Dictionary containing model name, variant, input resolution,
            inference device, latency, and any model-specific fields.
    """

    depth: np.ndarray
    colored_depth: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
