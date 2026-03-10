"""Marigold-DC model package — self-registers with MODEL_REGISTRY."""

from .configuration_marigold_dc import MarigoldDCConfig, _MARIGOLD_DC_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_marigold_dc import MarigoldDCModel
    return MarigoldDCModel


MODEL_REGISTRY.register(
    model_type="marigold-dc",
    config_cls=MarigoldDCConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_MARIGOLD_DC_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "MarigoldDCModel":
        from .modeling_marigold_dc import MarigoldDCModel
        return MarigoldDCModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MarigoldDCConfig", "MarigoldDCModel"]
