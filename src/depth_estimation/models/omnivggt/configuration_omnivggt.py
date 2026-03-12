"""
OmniVGGT configuration.
"""

from ...configuration_utils import BaseDepthConfig


_OMNIVGGT_VARIANT_MAP = {
    "omnivggt": {
        "backbone": "vitl",
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "hub_repo_id": "Livioni/OmniVGGT",
        "is_metric": True,
    },
}


class OmniVGGTConfig(BaseDepthConfig):
    """Configuration for OmniVGGT.

    OmniVGGT is a spatial foundation model for 3D geometric understanding.
    It predicts metric depth from single or multi-frame images using a
    DINOv2 ViT-L backbone with alternating frame/global attention.
    Weights are loaded from HuggingFace Hub (``Livioni/OmniVGGT``).
    """

    model_type = "omnivggt"

    def __init__(
        self,
        backbone: str = "vitl",
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        hub_repo_id: str = "Livioni/OmniVGGT",
        is_metric: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            is_metric=is_metric,
            # OmniVGGT applies its own ImageNet normalization internally.
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            **kwargs,
        )
        self.img_size = img_size
        self.hub_repo_id = hub_repo_id

    @classmethod
    def from_variant(cls, variant_id: str) -> "OmniVGGTConfig":
        if variant_id not in _OMNIVGGT_VARIANT_MAP:
            raise ValueError(
                f"Unknown OmniVGGT variant '{variant_id}'. "
                f"Available: {list(_OMNIVGGT_VARIANT_MAP.keys())}"
            )
        return cls(**_OMNIVGGT_VARIANT_MAP[variant_id])
