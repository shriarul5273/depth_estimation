"""Depth Anything v3 model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_anything_v3 import DepthAnythingV3Config, _DA3_VARIANT_MAP
from .modeling_depth_anything_v3 import DepthAnythingV3Model
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="depth-anything-v3",
    config_cls=DepthAnythingV3Config,
    model_cls=DepthAnythingV3Model,
    variant_ids=list(_DA3_VARIANT_MAP.keys()),
)

__all__ = ["DepthAnythingV3Config", "DepthAnythingV3Model"]
