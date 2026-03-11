"""
MoGe configuration.
"""

from ...configuration_utils import BaseDepthConfig


_MOGE_VARIANT_MAP = {
    "moge-v1": {
        "backbone": "vitl",
        "version": "v1",
        "hub_repo_id": "Ruicheng/moge-vitl",
        "is_metric": False,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
    },
    "moge-v2-vitl": {
        "backbone": "vitl",
        "version": "v2",
        "hub_repo_id": "Ruicheng/moge-2-vitl",
        "is_metric": True,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
    },
    "moge-v2-vitl-normal": {
        "backbone": "vitl",
        "version": "v2",
        "hub_repo_id": "Ruicheng/moge-2-vitl-normal",
        "is_metric": True,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
    },
    "moge-v2-vitb-normal": {
        "backbone": "vitb",
        "version": "v2",
        "hub_repo_id": "Ruicheng/moge-2-vitb-normal",
        "is_metric": True,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
    },
    "moge-v2-vits-normal": {
        "backbone": "vits",
        "version": "v2",
        "hub_repo_id": "Ruicheng/moge-2-vits-normal",
        "is_metric": True,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
    },
}


class MoGeConfig(BaseDepthConfig):
    """Configuration for MoGe (Monocular Geometry Estimation).

    Supports both MoGe-1 (relative/affine depth) and MoGe-2 (metric depth,
    optional normal maps). Weights are loaded from HuggingFace Hub via the
    ``moge`` package.
    """

    model_type = "moge"

    def __init__(
        self,
        backbone: str = "vitl",
        version: str = "v2",
        hub_repo_id: str = "Ruicheng/moge-2-vitl",
        is_metric: bool = True,
        input_size: int = 518,
        patch_size: int = 14,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            # MoGe applies its own ImageNet normalization internally,
            # so we pass raw [0, 1] values from the DepthProcessor.
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            **kwargs,
        )
        self.version = version
        self.hub_repo_id = hub_repo_id

    @classmethod
    def from_variant(cls, variant_id: str) -> "MoGeConfig":
        if variant_id not in _MOGE_VARIANT_MAP:
            raise ValueError(
                f"Unknown MoGe variant '{variant_id}'. "
                f"Available: {list(_MOGE_VARIANT_MAP.keys())}"
            )
        return cls(**_MOGE_VARIANT_MAP[variant_id])
