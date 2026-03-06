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
    Uses the bundled network implementation — no external ``depth_pro`` package needed.
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
        # Architecture params
        patch_encoder_preset: str = "dinov2l16_384",
        image_encoder_preset: str = "dinov2l16_384",
        fov_encoder_preset: str = "dinov2l16_384",
        decoder_features: int = 256,
        use_fov_head: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            **kwargs,
        )
        self.hub_repo_id = hub_repo_id
        self.hub_filename = hub_filename
        self.patch_encoder_preset = patch_encoder_preset
        self.image_encoder_preset = image_encoder_preset
        self.fov_encoder_preset = fov_encoder_preset
        self.decoder_features = decoder_features
        self.use_fov_head = use_fov_head

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthProConfig":
        if variant_id not in _DEPTHPRO_VARIANT_MAP:
            raise ValueError(
                f"Unknown DepthPro variant '{variant_id}'. "
                f"Available: {list(_DEPTHPRO_VARIANT_MAP.keys())}"
            )
        return cls()
