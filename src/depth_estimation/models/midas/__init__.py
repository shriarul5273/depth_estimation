"""MiDaS model package — self-registers with MODEL_REGISTRY."""

from .configuration_midas import MiDaSConfig, _MIDAS_VARIANT_MAP
from .modeling_midas import MiDaSModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="midas",
    config_cls=MiDaSConfig,
    model_cls=MiDaSModel,
    variant_ids=list(_MIDAS_VARIANT_MAP.keys()),
)

__all__ = ["MiDaSConfig", "MiDaSModel"]
