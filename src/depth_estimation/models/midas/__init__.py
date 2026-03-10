"""MiDaS model package — self-registers with MODEL_REGISTRY."""

from .configuration_midas import MiDaSConfig, _MIDAS_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_midas import MiDaSModel
    return MiDaSModel


MODEL_REGISTRY.register(
    model_type="midas",
    config_cls=MiDaSConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_MIDAS_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "MiDaSModel":
        from .modeling_midas import MiDaSModel
        return MiDaSModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MiDaSConfig", "MiDaSModel"]
