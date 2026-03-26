"""MoGe model package — self-registers with MODEL_REGISTRY."""

from .configuration_moge import MoGeConfig, _MOGE_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_moge import MoGeModel
    return MoGeModel


MODEL_REGISTRY.register(
    model_type="moge",
    config_cls=MoGeConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_MOGE_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "MoGeModel":
        from .modeling_moge import MoGeModel
        return MoGeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MoGeConfig", "MoGeModel"]
