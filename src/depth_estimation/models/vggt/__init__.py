"""VGGT model package — self-registers with MODEL_REGISTRY."""

from .configuration_vggt import VGGTConfig, _VGGT_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_vggt import VGGTModel
    return VGGTModel


MODEL_REGISTRY.register(
    model_type="vggt",
    config_cls=VGGTConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_VGGT_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "VGGTModel":
        from .modeling_vggt import VGGTModel
        return VGGTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VGGTConfig", "VGGTModel"]
