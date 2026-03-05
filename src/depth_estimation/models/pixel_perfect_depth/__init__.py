"""Pixel-Perfect Depth model package — self-registers with MODEL_REGISTRY."""

from .configuration_ppd import PixelPerfectDepthConfig, _PPD_VARIANT_MAP
from .modeling_ppd import PixelPerfectDepthModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="pixel-perfect-depth",
    config_cls=PixelPerfectDepthConfig,
    model_cls=PixelPerfectDepthModel,
    variant_ids=list(_PPD_VARIANT_MAP.keys()),
)

__all__ = ["PixelPerfectDepthConfig", "PixelPerfectDepthModel"]
