"""
depth_estimation — A Transformers-style Python library for monocular depth estimation.

Provides a unified, modular API for running, comparing, and integrating
depth estimation models.
"""

from .output import DepthOutput
from .configuration_utils import BaseDepthConfig
from .registry import MODEL_REGISTRY

# Auto classes
from .models.auto import AutoDepthModel, AutoProcessor

# Ensure model modules are imported so they self-register.
# These are now torch-free — modeling classes are loaded lazily on first use.
from .models import depth_anything_v1  # noqa: F401
from .models import depth_anything_v2  # noqa: F401
from .models import depth_anything_v3  # noqa: F401
from .models import zoedepth  # noqa: F401
from .models import midas  # noqa: F401
from .models import depth_pro  # noqa: F401
from .models import pixel_perfect_depth  # noqa: F401
from .models import marigold_dc  # noqa: F401
from .models import moge  # noqa: F401
from .models import omnivggt  # noqa: F401
from .models import vggt  # noqa: F401


def __getattr__(name):
    """Defer torch-heavy imports until first use."""
    if name == "BaseDepthModel":
        from .modeling_utils import BaseDepthModel
        globals()["BaseDepthModel"] = BaseDepthModel
        return BaseDepthModel
    if name == "DepthProcessor":
        from .processing_utils import DepthProcessor
        globals()["DepthProcessor"] = DepthProcessor
        return DepthProcessor
    if name == "DepthPipeline":
        from .pipeline_utils import DepthPipeline
        globals()["DepthPipeline"] = DepthPipeline
        return DepthPipeline
    if name == "pipeline":
        from .pipeline_utils import pipeline
        globals()["pipeline"] = pipeline
        return pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
