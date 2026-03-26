"""
Depth Anything v2 — Model package.

Exports config + model and self-registers with MODEL_REGISTRY.
"""

from .configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
    _V2_VARIANT_MAP,
)
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_depth_anything_v2 import DepthAnythingV2Model
    return DepthAnythingV2Model


# Self-register with the global registry
MODEL_REGISTRY.register(
    model_type="depth-anything-v2",
    config_cls=DepthAnythingV2Config,
    model_cls=_load_model_cls,
    variant_ids=list(_V2_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "DepthAnythingV2Model":
        from .modeling_depth_anything_v2 import DepthAnythingV2Model
        return DepthAnythingV2Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DepthAnythingV2Config",
    "DepthAnythingV2Model",
]
