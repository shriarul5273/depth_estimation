"""ZoeDepth model package — self-registers with MODEL_REGISTRY."""

from .configuration_zoedepth import ZoeDepthConfig, _ZOEDEPTH_VARIANT_MAP
from .modeling_zoedepth import ZoeDepthModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="zoedepth",
    config_cls=ZoeDepthConfig,
    model_cls=ZoeDepthModel,
    variant_ids=list(_ZOEDEPTH_VARIANT_MAP.keys()),
)

__all__ = ["ZoeDepthConfig", "ZoeDepthModel"]
