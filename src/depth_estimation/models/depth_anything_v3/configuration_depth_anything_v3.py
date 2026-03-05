"""
Depth Anything v3 configuration.
"""

from ...configuration_utils import BaseDepthConfig


_DA3_VARIANT_MAP = {
    "depth-anything-v3-small": "small",
    "depth-anything-v3-base": "base",
    "depth-anything-v3-large": "large",
    "depth-anything-v3-giant": "giant",
    "depth-anything-v3-nested-giant-large": "nested_giant_large",
    "depth-anything-v3-metric-large": "metric_large",
    "depth-anything-v3-mono-large": "mono_large",
}

_DA3_REPO_IDS = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3-LARGE",
    "giant": "depth-anything/DA3-GIANT",
    "nested_giant_large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "metric_large": "depth-anything/DA3METRIC-LARGE",
    "mono_large": "depth-anything/DA3MONO-LARGE",
}


class DepthAnythingV3Config(BaseDepthConfig):
    """Configuration for Depth Anything v3.

    Seven variants:
        - small, base, large, giant (relative depth)
        - nested_giant_large (nested architecture)
        - metric_large (metric depth)
        - mono_large (monocular)

    Requires ``depth_anything_3`` package (optional dependency).
    """

    model_type = "depth-anything-v3"

    def __init__(
        self,
        backbone: str = "large",
        input_size: int = 518,
        patch_size: int = 14,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            **kwargs,
        )

    @property
    def hub_repo_id(self) -> str:
        return _DA3_REPO_IDS.get(self.backbone, self.backbone)

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthAnythingV3Config":
        backbone = _DA3_VARIANT_MAP.get(variant_id)
        if backbone is None:
            raise ValueError(
                f"Unknown DA v3 variant '{variant_id}'. "
                f"Available: {list(_DA3_VARIANT_MAP.keys())}"
            )
        return cls(backbone=backbone)
