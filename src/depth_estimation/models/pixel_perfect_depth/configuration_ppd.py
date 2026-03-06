"""
Pixel-Perfect Depth configuration.
"""

from ...configuration_utils import BaseDepthConfig


_PPD_VARIANT_MAP = {
    "pixel-perfect-depth": "pixel-perfect-depth",
}


class PixelPerfectDepthConfig(BaseDepthConfig):
    """Configuration for Pixel-Perfect Depth.

    Diffusion-based monocular depth estimation using a DiT backbone
    conditioned on DINOv2 ViT-L semantic features extracted from a
    Depth Anything V2 encoder.
    """

    model_type = "pixel-perfect-depth"

    def __init__(
        self,
        backbone: str = "vitl",
        input_size: int = 1024,
        patch_size: int = 16,
        is_metric: bool = False,
        # DiT architecture
        dit_in_channels: int = 4,
        dit_out_channels: int = 1,
        dit_hidden_size: int = 1024,
        dit_depth: int = 24,
        dit_num_heads: int = 16,
        dit_mlp_ratio: float = 4.0,
        # Sampling
        sampling_steps: int = 10,
        # HuggingFace Hub
        hub_repo_id: str = "gangweix/Pixel-Perfect-Depth",
        hub_filename: str = "ppd.pth",
        semantics_hub_repo_id: str = "depth-anything/Depth-Anything-V2-Large",
        semantics_hub_filename: str = "depth_anything_v2_vitl.pth",
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            **kwargs,
        )
        self.dit_in_channels = dit_in_channels
        self.dit_out_channels = dit_out_channels
        self.dit_hidden_size = dit_hidden_size
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.sampling_steps = sampling_steps
        self.hub_repo_id = hub_repo_id
        self.hub_filename = hub_filename
        self.semantics_hub_repo_id = semantics_hub_repo_id
        self.semantics_hub_filename = semantics_hub_filename

    @classmethod
    def from_variant(cls, variant_id: str) -> "PixelPerfectDepthConfig":
        if variant_id not in _PPD_VARIANT_MAP:
            raise ValueError(
                f"Unknown Pixel-Perfect-Depth variant '{variant_id}'. "
                f"Available: {list(_PPD_VARIANT_MAP.keys())}"
            )
        return cls()
