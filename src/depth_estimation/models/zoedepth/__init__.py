"""ZoeDepth model package — self-registers with MODEL_REGISTRY."""

from .configuration_zoedepth import ZoeDepthConfig, _ZOEDEPTH_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_zoedepth import ZoeDepthModel
    return ZoeDepthModel


MODEL_REGISTRY.register(
    model_type="zoedepth",
    config_cls=ZoeDepthConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_ZOEDEPTH_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "ZoeDepthModel":
        from .modeling_zoedepth import ZoeDepthModel
        return ZoeDepthModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ZoeDepthConfig", "ZoeDepthModel"]
