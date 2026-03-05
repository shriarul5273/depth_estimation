"""
Marigold-DC configuration.
"""

from ...configuration_utils import BaseDepthConfig


_MARIGOLD_DC_VARIANT_MAP = {
    "marigold-dc": "marigold-dc",
}


class MarigoldDCConfig(BaseDepthConfig):
    """Configuration for Marigold Depth Completion.

    Extends MarigoldDepthPipeline from diffusers for depth completion
    using sparse depth guidance with diffusion-based optimization.
    Requires ``diffusers`` package.
    """

    model_type = "marigold-dc"

    def __init__(
        self,
        backbone: str = "marigold-dc",
        input_size: int = 768,
        patch_size: int = 16,
        num_inference_steps: int = 50,
        ensemble_size: int = 1,
        processing_resolution: int = 768,
        seed: int = 2024,
        hub_model_id: str = "prs-eth/marigold-depth-v1-0",
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            **kwargs,
        )
        self.num_inference_steps = num_inference_steps
        self.ensemble_size = ensemble_size
        self.processing_resolution = processing_resolution
        self.seed = seed
        self.hub_model_id = hub_model_id

    @classmethod
    def from_variant(cls, variant_id: str) -> "MarigoldDCConfig":
        if variant_id not in _MARIGOLD_DC_VARIANT_MAP:
            raise ValueError(
                f"Unknown Marigold-DC variant '{variant_id}'. "
                f"Available: {list(_MARIGOLD_DC_VARIANT_MAP.keys())}"
            )
        return cls()
