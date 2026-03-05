"""
DepthAnythingV2Config — Configuration for Depth Anything v2 models.

Inherits from BaseDepthConfig and only overrides default values per variant.
"""

from typing import Any, Dict, List, Optional

from ...configuration_utils import BaseDepthConfig


# Backbone-specific parameters (different from v1)
_V2_BACKBONE_CONFIGS = {
    "vits": {
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "intermediate_layer_idx": [2, 5, 8, 11],
    },
    "vitb": {
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "intermediate_layer_idx": [2, 5, 8, 11],
    },
    "vitl": {
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "intermediate_layer_idx": [4, 11, 17, 23],
    },
}

# HuggingFace Hub repos for checkpoint download
_V2_HUB_REPOS = {
    "vits": "depth-anything/Depth-Anything-V2-Small",
    "vitb": "depth-anything/Depth-Anything-V2-Base",
    "vitl": "depth-anything/Depth-Anything-V2-Large",
}

# Variant ID → backbone mapping
_V2_VARIANT_MAP = {
    "depth-anything-v2-vits": "vits",
    "depth-anything-v2-vitb": "vitb",
    "depth-anything-v2-vitl": "vitl",
}


class DepthAnythingV2Config(BaseDepthConfig):
    """Configuration for Depth Anything v2 models.

    Supports three variants:
        - ``depth-anything-v2-vits``: ViT-S (Small, fastest)
        - ``depth-anything-v2-vitb``: ViT-B (Base, balanced)
        - ``depth-anything-v2-vitl``: ViT-L (Large, best quality)
    """

    model_type = "depth-anything-v2"

    def __init__(
        self,
        backbone: str = "vitl",
        input_size: int = 518,
        patch_size: int = 14,
        features: int = 256,
        out_channels: Optional[List[int]] = None,
        intermediate_layer_idx: Optional[List[int]] = None,
        use_clstoken: bool = False,
        use_bn: bool = False,
        **kwargs: Any,
    ):
        # Apply backbone-specific defaults if available
        if backbone in _V2_BACKBONE_CONFIGS:
            bc = _V2_BACKBONE_CONFIGS[backbone]
            kwargs.setdefault("embed_dim", bc["embed_dim"])
            kwargs.setdefault("num_heads", bc["num_heads"])
            kwargs.setdefault("num_layers", bc["num_layers"])
            features = kwargs.pop("features", bc["features"])
            out_channels = out_channels or bc["out_channels"]
            intermediate_layer_idx = intermediate_layer_idx or bc["intermediate_layer_idx"]

        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            features=features,
            out_channels=out_channels,
            **kwargs,
        )
        self.intermediate_layer_idx = intermediate_layer_idx or [4, 11, 17, 23]
        self.use_clstoken = use_clstoken
        self.use_bn = use_bn

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthAnythingV2Config":
        """Create a config from a variant identifier string."""
        if variant_id not in _V2_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_V2_VARIANT_MAP.keys())}"
            )
        backbone = _V2_VARIANT_MAP[variant_id]
        return cls(backbone=backbone)

    @property
    def hub_repo_id(self) -> str:
        """HuggingFace Hub repo ID for checkpoint download."""
        return _V2_HUB_REPOS.get(self.backbone, "")

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint filename on HuggingFace Hub."""
        return f"depth_anything_v2_{self.backbone}.pth"
