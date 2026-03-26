"""
VGGT configuration.
"""

from ...configuration_utils import BaseDepthConfig


_VGGT_VARIANT_MAP = {
    "vggt": {
        "backbone": "vitl",
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "hub_repo_id": "facebook/VGGT-1B",
        "is_metric": True,
    },
    "vggt-commercial": {
        "backbone": "vitl",
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": 1024,
        "hub_repo_id": "facebook/VGGT-1B-Commercial",
        "is_metric": True,
    },
}


class VGGTConfig(BaseDepthConfig):
    """Configuration for VGGT.

    VGGT (Visual Geometry Grounded Transformer) is a spatial foundation model
    for 3D geometric understanding. It predicts metric depth from single or
    multi-frame images using a DINOv2 ViT-L backbone with alternating
    frame/global attention.
    Weights are loaded from HuggingFace Hub (``facebook/VGGT-1B``).
    """

    model_type = "vggt"

    def __init__(
        self,
        backbone: str = "vitl",
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        hub_repo_id: str = "facebook/VGGT-1B",
        is_metric: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            is_metric=is_metric,
            # VGGT applies its own ImageNet normalization internally.
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            **kwargs,
        )
        self.img_size = img_size
        self.hub_repo_id = hub_repo_id

    @classmethod
    def from_variant(cls, variant_id: str) -> "VGGTConfig":
        if variant_id not in _VGGT_VARIANT_MAP:
            raise ValueError(
                f"Unknown VGGT variant '{variant_id}'. "
                f"Available: {list(_VGGT_VARIANT_MAP.keys())}"
            )
        return cls(**_VGGT_VARIANT_MAP[variant_id])
