"""Marigold-DC model package — self-registers with MODEL_REGISTRY."""

from .configuration_marigold_dc import MarigoldDCConfig, _MARIGOLD_DC_VARIANT_MAP
from .modeling_marigold_dc import MarigoldDCModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="marigold-dc",
    config_cls=MarigoldDCConfig,
    model_cls=MarigoldDCModel,
    variant_ids=list(_MARIGOLD_DC_VARIANT_MAP.keys()),
)

__all__ = ["MarigoldDCConfig", "MarigoldDCModel"]
