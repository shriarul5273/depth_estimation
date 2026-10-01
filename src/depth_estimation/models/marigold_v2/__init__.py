"""Marigold V2 model package — self-registers with MODEL_REGISTRY."""

from .configuration_marigold_v2 import MarigoldV2Config, _MARIGOLD_V2_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_marigold_v2 import MarigoldV2Model

    return MarigoldV2Model


MODEL_REGISTRY.register(
    model_type="marigold-v2",
    config_cls=MarigoldV2Config,
    model_cls=_load_model_cls,
    variant_ids=list(_MARIGOLD_V2_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "MarigoldV2Model":
        from .modeling_marigold_v2 import MarigoldV2Model

        return MarigoldV2Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MarigoldV2Config", "MarigoldV2Model"]
