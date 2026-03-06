"""
depth_estimation — A Transformers-style Python library for monocular depth estimation.

Provides a unified, modular API for running, comparing, and integrating
depth estimation models.
"""

from .output import DepthOutput
from .configuration_utils import BaseDepthConfig
from .modeling_utils import BaseDepthModel
from .processing_utils import DepthProcessor
from .pipeline_utils import DepthPipeline, pipeline
from .registry import MODEL_REGISTRY

# Auto classes
from .models.auto import AutoDepthModel, AutoProcessor

# Ensure model modules are imported so they self-register
from .models import depth_anything_v1  # noqa: F401
from .models import depth_anything_v2  # noqa: F401
from .models import depth_anything_v3  # noqa: F401
from .models import zoedepth  # noqa: F401
from .models import midas  # noqa: F401
from .models import depth_pro  # noqa: F401
from .models import pixel_perfect_depth  # noqa: F401
from .models import marigold_dc  # noqa: F401

__all__ = [
    "DepthOutput",
    "BaseDepthConfig",
    "BaseDepthModel",
    "DepthProcessor",
    "DepthPipeline",
    "pipeline",
    "AutoDepthModel",
    "AutoProcessor",
    "MODEL_REGISTRY",
]

__version__ = "0.0.3"
