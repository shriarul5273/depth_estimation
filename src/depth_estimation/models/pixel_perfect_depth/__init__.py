"""Pixel-Perfect Depth model package — self-registers with MODEL_REGISTRY."""

from .configuration_ppd import PixelPerfectDepthConfig, _PPD_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_ppd import PixelPerfectDepthModel
    return PixelPerfectDepthModel


MODEL_REGISTRY.register(
    model_type="pixel-perfect-depth",
    config_cls=PixelPerfectDepthConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_PPD_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "PixelPerfectDepthModel":
        from .modeling_ppd import PixelPerfectDepthModel
        return PixelPerfectDepthModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PixelPerfectDepthConfig", "PixelPerfectDepthModel"]
