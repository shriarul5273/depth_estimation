"""OmniVGGT model package — self-registers with MODEL_REGISTRY."""

from .configuration_omnivggt import OmniVGGTConfig, _OMNIVGGT_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_omnivggt import OmniVGGTModel
    return OmniVGGTModel


MODEL_REGISTRY.register(
    model_type="omnivggt",
    config_cls=OmniVGGTConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_OMNIVGGT_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "OmniVGGTModel":
        from .modeling_omnivggt import OmniVGGTModel
        return OmniVGGTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OmniVGGTConfig", "OmniVGGTModel"]
