"""
MiDaS configuration.
"""

from ...configuration_utils import BaseDepthConfig


_MIDAS_VARIANT_MAP = {
    "midas-dpt-large": "dpt-large",
    "midas-dpt-hybrid": "dpt-hybrid",
    "midas-beit-large": "beit-large",
}

_MIDAS_HF_MODELS = {
    "dpt-large": "Intel/dpt-large",
    "dpt-hybrid": "Intel/dpt-hybrid-midas",
    "beit-large": "Intel/dpt-beit-large-512",
}

_MIDAS_DISPLAY_NAMES = {
    "dpt-large": "MiDaS v3.0 (DPT-Large)",
    "dpt-hybrid": "MiDaS v3.0 (DPT-Hybrid)",
    "beit-large": "MiDaS v3.1 (BEiT-Large)",
}


class MiDaSConfig(BaseDepthConfig):
    """Configuration for Intel MiDaS depth estimation models.

    Three variants:
        - ``midas-dpt-large``: MiDaS v3.0 DPT-Large
        - ``midas-dpt-hybrid``: MiDaS v3.0 DPT-Hybrid (smaller)
        - ``midas-beit-large``: MiDaS v3.1 DPT-BEiT-Large (most accurate)
    """

    model_type = "midas"

    def __init__(
        self,
        backbone: str = "dpt-large",
        input_size: int = 384,
        patch_size: int = 16,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            **kwargs,
        )

    @property
    def hf_model_id(self) -> str:
        return _MIDAS_HF_MODELS.get(self.backbone, self.backbone)

    @property
    def display_name(self) -> str:
        return _MIDAS_DISPLAY_NAMES.get(self.backbone, f"MiDaS ({self.backbone})")

    @classmethod
    def from_variant(cls, variant_id: str) -> "MiDaSConfig":
        backbone = _MIDAS_VARIANT_MAP.get(variant_id)
        if backbone is None:
            raise ValueError(
                f"Unknown MiDaS variant '{variant_id}'. "
                f"Available: {list(_MIDAS_VARIANT_MAP.keys())}"
            )
        return cls(backbone=backbone)
