"""
Apple DepthPro configuration.
"""

from ...configuration_utils import BaseDepthConfig


_DEPTHPRO_VARIANT_MAP = {
    "depth-pro": "depth-pro",
}


class DepthProConfig(BaseDepthConfig):
    """Configuration for Apple DepthPro.

    Sharp monocular metric depth in less than a second.
    Requires ``depth_pro`` package (optional dependency).
    """

    model_type = "depth-pro"

    def __init__(
        self,
        backbone: str = "depth-pro",
        input_size: int = 1536,
        patch_size: int = 16,
        is_metric: bool = True,
        hub_repo_id: str = "apple/DepthPro",
        hub_filename: str = "depth_pro.pt",
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            **kwargs,
        )
        self.hub_repo_id = hub_repo_id
        self.hub_filename = hub_filename

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthProConfig":
        if variant_id not in _DEPTHPRO_VARIANT_MAP:
            raise ValueError(
                f"Unknown DepthPro variant '{variant_id}'. "
                f"Available: {list(_DEPTHPRO_VARIANT_MAP.keys())}"
            )
        return cls()
