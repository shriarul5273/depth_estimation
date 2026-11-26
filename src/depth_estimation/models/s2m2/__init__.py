"""S2M2 stereo model package — self-registers with MODEL_REGISTRY."""

from .configuration_s2m2 import S2M2Config, _S2M2_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_s2m2 import S2M2Model

    return S2M2Model


MODEL_REGISTRY.register(
    model_type="s2m2",
    config_cls=S2M2Config,
    model_cls=_load_model_cls,
    variant_ids=list(_S2M2_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "S2M2Model":
        from .modeling_s2m2 import S2M2Model

        return S2M2Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["S2M2Config", "S2M2Model"]
