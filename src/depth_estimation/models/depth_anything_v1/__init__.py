"""
Depth Anything v1 — Model package.

Exports config + model and self-registers with MODEL_REGISTRY.
"""

from .configuration_depth_anything_v1 import (
    DepthAnythingV1Config,
    _V1_VARIANT_MAP,
)
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_depth_anything_v1 import DepthAnythingV1Model
    return DepthAnythingV1Model


# Self-register with the global registry
MODEL_REGISTRY.register(
    model_type="depth-anything-v1",
    config_cls=DepthAnythingV1Config,
    model_cls=_load_model_cls,
    variant_ids=list(_V1_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "DepthAnythingV1Model":
        from .modeling_depth_anything_v1 import DepthAnythingV1Model
        return DepthAnythingV1Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DepthAnythingV1Config",
    "DepthAnythingV1Model",
]
