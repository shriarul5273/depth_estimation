"""
DepthAnythingV1Config — Configuration for Depth Anything v1 models.

Inherits from BaseDepthConfig and only overrides default values per variant.
"""

from typing import Any, Dict, List, Optional

from ...configuration_utils import BaseDepthConfig


# Backbone-specific parameters
_V1_BACKBONE_CONFIGS = {
    "vits": {
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}

# HuggingFace Hub model IDs for weight loading
_V1_HUB_MODELS = {
    "vits": "LiheYoung/depth_anything_vits14",
    "vitb": "LiheYoung/depth_anything_vitb14",
    "vitl": "LiheYoung/depth_anything_vitl14",
}

# Variant ID → backbone mapping
_V1_VARIANT_MAP = {
    "depth-anything-v1-vits": "vits",
    "depth-anything-v1-vitb": "vitb",
    "depth-anything-v1-vitl": "vitl",
}


class DepthAnythingV1Config(BaseDepthConfig):
    """Configuration for Depth Anything v1 models.

    Supports three variants:
        - ``depth-anything-v1-vits``: ViT-S (Small)
        - ``depth-anything-v1-vitb``: ViT-B (Base)
        - ``depth-anything-v1-vitl``: ViT-L (Large)
    """

    model_type = "depth-anything-v1"

    def __init__(
        self,
        backbone: str = "vitl",
        input_size: int = 518,
        patch_size: int = 14,
        features: int = 256,
        out_channels: Optional[List[int]] = None,
        use_clstoken: bool = False,
        use_bn: bool = False,
        **kwargs: Any,
    ):
        # Apply backbone-specific defaults if available
        if backbone in _V1_BACKBONE_CONFIGS:
            bc = _V1_BACKBONE_CONFIGS[backbone]
            kwargs.setdefault("embed_dim", bc["embed_dim"])
            kwargs.setdefault("num_heads", bc["num_heads"])
            kwargs.setdefault("num_layers", bc["num_layers"])
            features = kwargs.pop("features", bc["features"])
            out_channels = out_channels or bc["out_channels"]

        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            features=features,
            out_channels=out_channels,
            **kwargs,
        )
        self.use_clstoken = use_clstoken
        self.use_bn = use_bn

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthAnythingV1Config":
        """Create a config from a variant identifier string.

        Args:
            variant_id: One of "depth-anything-v1-vits", "depth-anything-v1-vitb",
                "depth-anything-v1-vitl".
        """
        if variant_id not in _V1_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_V1_VARIANT_MAP.keys())}"
            )
        backbone = _V1_VARIANT_MAP[variant_id]
        return cls(backbone=backbone)

    @property
    def hub_model_id(self) -> str:
        """HuggingFace Hub model ID for weight loading."""
        return _V1_HUB_MODELS.get(self.backbone, "")
