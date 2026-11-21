"""Apple Depth Pro 2 model package — self-registers with MODEL_REGISTRY."""

from .configuration_depth_pro_2 import DepthPro2Config, _DEPTHPRO2_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_depth_pro_2 import DepthPro2Model

    return DepthPro2Model


MODEL_REGISTRY.register(
    model_type="depth-pro-2",
    config_cls=DepthPro2Config,
    model_cls=_load_model_cls,
    variant_ids=list(_DEPTHPRO2_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "DepthPro2Model":
        from .modeling_depth_pro_2 import DepthPro2Model

        return DepthPro2Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DepthPro2Config", "DepthPro2Model"]
