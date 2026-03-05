"""
ZoeDepth configuration.
"""

from ...configuration_utils import BaseDepthConfig


_ZOEDEPTH_VARIANT_MAP = {
    "zoedepth": "zoedepth-nyu-kitti",
}

_ZOEDEPTH_HF_MODELS = {
    "zoedepth-nyu-kitti": "Intel/zoedepth-nyu-kitti",
}


class ZoeDepthConfig(BaseDepthConfig):
    """Configuration for Intel ZoeDepth.

    ZoeDepth is a metric depth estimation model fine-tuned on NYU and KITTI.
    Uses HuggingFace transformers pipeline internally.
    """

    model_type = "zoedepth"

    def __init__(
        self,
        backbone: str = "zoedepth-nyu-kitti",
        input_size: int = 384,
        patch_size: int = 16,
        is_metric: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            **kwargs,
        )

    @property
    def hf_model_id(self) -> str:
        return _ZOEDEPTH_HF_MODELS.get(self.backbone, self.backbone)

    @classmethod
    def from_variant(cls, variant_id: str) -> "ZoeDepthConfig":
        backbone = _ZOEDEPTH_VARIANT_MAP.get(variant_id)
        if backbone is None:
            raise ValueError(
                f"Unknown ZoeDepth variant '{variant_id}'. "
                f"Available: {list(_ZOEDEPTH_VARIANT_MAP.keys())}"
            )
        return cls(backbone=backbone)
