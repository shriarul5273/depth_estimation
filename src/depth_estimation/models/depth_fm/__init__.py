"""DepthFM model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_fm import DepthFMConfig, _DEPTHFM_VARIANT_MAP
from .modeling_depth_fm import DepthFMModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="depth-fm",
    config_cls=DepthFMConfig,
    model_cls=DepthFMModel,
    variant_ids=list(_DEPTHFM_VARIANT_MAP.keys()),
)

__all__ = ["DepthFMConfig", "DepthFMModel"]
