"""
Depth Anything v1 — Model package.

Exports config + model and self-registers with MODEL_REGISTRY.
"""

from .configuration_depth_anything_v1 import (
    DepthAnythingV1Config,
    _V1_VARIANT_MAP,
)
from .modeling_depth_anything_v1 import DepthAnythingV1Model
from ...registry import MODEL_REGISTRY

# Self-register with the global registry
MODEL_REGISTRY.register(
    model_type="depth-anything-v1",
    config_cls=DepthAnythingV1Config,
    model_cls=DepthAnythingV1Model,
    variant_ids=list(_V1_VARIANT_MAP.keys()),
)

__all__ = [
    "DepthAnythingV1Config",
    "DepthAnythingV1Model",
]
