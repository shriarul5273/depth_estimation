"""Apple DepthPro model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_pro import DepthProConfig, _DEPTHPRO_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_depth_pro import DepthProModel
    return DepthProModel


MODEL_REGISTRY.register(
    model_type="depth-pro",
    config_cls=DepthProConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_DEPTHPRO_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "DepthProModel":
        from .modeling_depth_pro import DepthProModel
        return DepthProModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DepthProConfig", "DepthProModel"]
