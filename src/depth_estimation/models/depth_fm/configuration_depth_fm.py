"""
DepthFM configuration.
"""

from ...configuration_utils import BaseDepthConfig


_DEPTHFM_VARIANT_MAP = {
    "depth-fm": "depth-fm",
}


class DepthFMConfig(BaseDepthConfig):
    """Configuration for DepthFM.

    DepthFM: Fast Monocular Depth Estimation with Flow Matching.
    Uses a UNet + Stable Diffusion VAE with ODE solving.
    Requires ``depthfm``, ``diffusers``, ``torchdiffeq``, ``einops`` packages.
    """

    model_type = "depth-fm"

    def __init__(
        self,
        backbone: str = "depth-fm",
        input_size: int = 512,
        patch_size: int = 64,
        num_steps: int = 2,
        ensemble_size: int = 4,
        hub_repo_id: str = "SharpAI/DepthFM",
        hub_filename: str = "depthfm-v1.ckpt",
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            **kwargs,
        )
        self.num_steps = num_steps
        self.ensemble_size = ensemble_size
        self.hub_repo_id = hub_repo_id
        self.hub_filename = hub_filename

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthFMConfig":
        if variant_id not in _DEPTHFM_VARIANT_MAP:
            raise ValueError(
                f"Unknown DepthFM variant '{variant_id}'. "
                f"Available: {list(_DEPTHFM_VARIANT_MAP.keys())}"
            )
        return cls()
