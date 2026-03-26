"""Depth Anything v3 model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_anything_v3 import (
    DepthAnythingV3Config,
    _V3_VARIANT_MAP,
    _V3_NESTED_VARIANT_IDS,
)
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_depth_anything_v3 import DepthAnythingV3Model
    return DepthAnythingV3Model


def _load_nested_model_cls():
    from .modeling_depth_anything_v3 import DepthAnythingV3NestedModel
    return DepthAnythingV3NestedModel


MODEL_REGISTRY.register(
    model_type="depth-anything-v3",
    config_cls=DepthAnythingV3Config,
    model_cls=_load_model_cls,
    variant_ids=list(_V3_VARIANT_MAP.keys()),
)

MODEL_REGISTRY.register(
    model_type="depth-anything-v3-nested",
    config_cls=DepthAnythingV3Config,
    model_cls=_load_nested_model_cls,
    variant_ids=_V3_NESTED_VARIANT_IDS,
)


def __getattr__(name):
    if name == "DepthAnythingV3Model":
        from .modeling_depth_anything_v3 import DepthAnythingV3Model
        return DepthAnythingV3Model
    if name == "DepthAnythingV3NestedModel":
        from .modeling_depth_anything_v3 import DepthAnythingV3NestedModel
        return DepthAnythingV3NestedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DepthAnythingV3Config",
    "DepthAnythingV3Model",
    "DepthAnythingV3NestedModel",
]
