"""
Depth Anything v2 — Model package.

Exports config + model and self-registers with MODEL_REGISTRY.
"""

from .configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
    _V2_VARIANT_MAP,
)
from .modeling_depth_anything_v2 import DepthAnythingV2Model
from ...registry import MODEL_REGISTRY

# Self-register with the global registry
MODEL_REGISTRY.register(
    model_type="depth-anything-v2",
    config_cls=DepthAnythingV2Config,
    model_cls=DepthAnythingV2Model,
    variant_ids=list(_V2_VARIANT_MAP.keys()),
)

__all__ = [
    "DepthAnythingV2Config",
    "DepthAnythingV2Model",
]
