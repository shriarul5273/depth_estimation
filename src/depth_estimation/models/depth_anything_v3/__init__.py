"""Depth Anything v3 model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_anything_v3 import (
    DepthAnythingV3Config,
    _V3_VARIANT_MAP,
    _V3_NESTED_VARIANT_IDS,
)
from .modeling_depth_anything_v3 import DepthAnythingV3Model, DepthAnythingV3NestedModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="depth-anything-v3",
    config_cls=DepthAnythingV3Config,
    model_cls=DepthAnythingV3Model,
    variant_ids=list(_V3_VARIANT_MAP.keys()),
)

MODEL_REGISTRY.register(
    model_type="depth-anything-v3-nested",
    config_cls=DepthAnythingV3Config,
    model_cls=DepthAnythingV3NestedModel,
    variant_ids=_V3_NESTED_VARIANT_IDS,
)

__all__ = [
    "DepthAnythingV3Config",
    "DepthAnythingV3Model",
    "DepthAnythingV3NestedModel",
]
