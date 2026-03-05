"""
Pixel-Perfect Depth configuration.
"""

from ...configuration_utils import BaseDepthConfig


_PPD_VARIANT_MAP = {
    "pixel-perfect-depth": "ppd",
}


class PixelPerfectDepthConfig(BaseDepthConfig):
    """Configuration for Pixel-Perfect Depth.

    Uses PPD for relative depth + MoGe for metric alignment.
    Requires ``ppd`` and ``moge`` packages (optional dependencies).
    """

    model_type = "pixel-perfect-depth"

    def __init__(
        self,
        backbone: str = "ppd",
        input_size: int = 518,
        patch_size: int = 14,
        is_metric: bool = True,
        sampling_steps: int = 20,
        ppd_hub_repo_id: str = "gangweix/Pixel-Perfect-Depth",
        ppd_hub_filename: str = "ppd.pth",
        moge_model_id: str = "Ruicheng/moge-2-vitl-normal",
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            **kwargs,
        )
        self.sampling_steps = sampling_steps
        self.ppd_hub_repo_id = ppd_hub_repo_id
        self.ppd_hub_filename = ppd_hub_filename
        self.moge_model_id = moge_model_id

    @classmethod
    def from_variant(cls, variant_id: str) -> "PixelPerfectDepthConfig":
        if variant_id not in _PPD_VARIANT_MAP:
            raise ValueError(
                f"Unknown PPD variant '{variant_id}'. "
                f"Available: {list(_PPD_VARIANT_MAP.keys())}"
            )
        return cls()
