"""Apple DepthPro model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_pro import DepthProConfig, _DEPTHPRO_VARIANT_MAP
from .modeling_depth_pro import DepthProModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="depth-pro",
    config_cls=DepthProConfig,
    model_cls=DepthProModel,
    variant_ids=list(_DEPTHPRO_VARIANT_MAP.keys()),
)

__all__ = ["DepthProConfig", "DepthProModel"]
