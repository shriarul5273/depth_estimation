"""
DepthAnythingV3Config — Configuration for Depth Anything v3 models.

Inherits from BaseDepthConfig and only overrides default values per variant.
"""

from typing import Any, List, Optional

from ...configuration_utils import BaseDepthConfig


# Backbone-specific parameters (read directly from da3-*.yaml configs)
_V3_BACKBONE_CONFIGS = {
    "small": {
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "head_dim_in": 768,           # 2 × embed_dim  (cat_token=True)
        "out_layers": [5, 7, 9, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "cat_token": True,
        "dinov2_name": "vits",
        "head_cls": "DualDPT",
        "output_dim": 2,
    },
    "base": {
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "head_dim_in": 1536,          # 2 × embed_dim
        "out_layers": [5, 7, 9, 11],
        "alt_start": 4,
        "qknorm_start": 4,
        "rope_start": 4,
        "cat_token": True,
        "dinov2_name": "vitb",
        "head_cls": "DualDPT",
        "output_dim": 2,
    },
    "large": {
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "head_dim_in": 2048,          # 2 × embed_dim
        "out_layers": [11, 15, 19, 23],
        "alt_start": 8,
        "qknorm_start": 8,
        "rope_start": 8,
        "cat_token": True,
        "dinov2_name": "vitl",
        "head_cls": "DualDPT",
        "output_dim": 2,
    },
    "giant": {
        "embed_dim": 1536,
        "num_heads": 24,
        "num_layers": 40,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "head_dim_in": 3072,          # 2 × embed_dim
        "out_layers": [19, 27, 33, 39],
        "alt_start": 13,
        "qknorm_start": 13,
        "rope_start": 13,
        "cat_token": True,
        "dinov2_name": "vitg",
        "head_cls": "DualDPT",
        "output_dim": 2,
    },
    "mono_large": {
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "head_dim_in": 1024,          # embed_dim  (cat_token=False)
        "out_layers": [4, 11, 17, 23],
        "alt_start": -1,
        "qknorm_start": -1,
        "rope_start": -1,
        "cat_token": False,
        "dinov2_name": "vitl",
        "head_cls": "DPT",
        "output_dim": 1,
    },
    "metric_large": {
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "head_dim_in": 1024,          # embed_dim  (cat_token=False)
        "out_layers": [4, 11, 17, 23],
        "alt_start": -1,
        "qknorm_start": -1,
        "rope_start": -1,
        "cat_token": False,
        "dinov2_name": "vitl",
        "head_cls": "DPT",
        "output_dim": 1,
        "is_metric": True,
    },
}

# HuggingFace Hub repos for checkpoint download
_V3_HUB_REPOS = {
    "small": "depth-anything/DA3-SMALL",
    "base": "depth-anything/DA3-BASE",
    "large": "depth-anything/DA3-LARGE",
    "giant": "depth-anything/DA3-GIANT",
    "mono_large": "depth-anything/DA3MONO-LARGE",
    "metric_large": "depth-anything/DA3METRIC-LARGE",
}

# Variant ID → backbone mapping
_V3_VARIANT_MAP = {
    "depth-anything-v3-small": "small",
    "depth-anything-v3-base": "base",
    "depth-anything-v3-large": "large",
    "depth-anything-v3-giant": "giant",
    "depth-anything-v3-mono-large": "mono_large",
    "depth-anything-v3-metric-large": "metric_large",
}

# Nested variant IDs (resolved by DepthAnythingV3NestedModel, not DepthAnythingV3Model)
_V3_NESTED_VARIANT_IDS = ["depth-anything-v3-nested-giant-large"]


class DepthAnythingV3Config(BaseDepthConfig):
    """Configuration for Depth Anything v3 models.

    Supports six variants:
        - ``depth-anything-v3-small``: ViT-S + DualDPT head, RoPE + global attn
        - ``depth-anything-v3-base``: ViT-B + DualDPT head, RoPE + global attn
        - ``depth-anything-v3-large``: ViT-L + DualDPT head, RoPE + global attn
        - ``depth-anything-v3-giant``: ViT-G + DualDPT head, RoPE + global attn
        - ``depth-anything-v3-mono-large``: ViT-L + DPT head, single-image only
        - ``depth-anything-v3-metric-large``: ViT-L + DPT head, metric depth
    """

    model_type = "depth-anything-v3"

    def __init__(
        self,
        backbone: str = "large",
        input_size: int = 518,
        patch_size: int = 14,
        features: int = 256,
        out_channels: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        if backbone in _V3_BACKBONE_CONFIGS:
            bc = _V3_BACKBONE_CONFIGS[backbone]
            kwargs.setdefault("embed_dim", bc["embed_dim"])
            kwargs.setdefault("num_heads", bc["num_heads"])
            kwargs.setdefault("num_layers", bc["num_layers"])
            kwargs.setdefault("is_metric", bc.get("is_metric", False))
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

        # V3-specific attributes set after super().__init__
        if backbone in _V3_BACKBONE_CONFIGS:
            bc = _V3_BACKBONE_CONFIGS[backbone]
            self.head_dim_in = bc["head_dim_in"]
            self.out_layers = bc["out_layers"]
            self.alt_start = bc["alt_start"]
            self.qknorm_start = bc["qknorm_start"]
            self.rope_start = bc["rope_start"]
            self.cat_token = bc["cat_token"]
            self.dinov2_name = bc["dinov2_name"]
            self.head_cls = bc["head_cls"]
            self.output_dim = bc["output_dim"]
        else:
            self.head_dim_in = kwargs.get("head_dim_in", 2048)
            self.out_layers = kwargs.get("out_layers", [11, 15, 19, 23])
            self.alt_start = kwargs.get("alt_start", 8)
            self.qknorm_start = kwargs.get("qknorm_start", 8)
            self.rope_start = kwargs.get("rope_start", 8)
            self.cat_token = kwargs.get("cat_token", True)
            self.dinov2_name = kwargs.get("dinov2_name", "vitl")
            self.head_cls = kwargs.get("head_cls", "DualDPT")
            self.output_dim = kwargs.get("output_dim", 2)

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthAnythingV3Config":
        """Create a config from a variant identifier string."""
        if variant_id in _V3_VARIANT_MAP:
            return cls(backbone=_V3_VARIANT_MAP[variant_id])
        if variant_id in _V3_NESTED_VARIANT_IDS:
            return cls(backbone="giant")
        raise ValueError(
            f"Unknown variant '{variant_id}'. "
            f"Available: {list(_V3_VARIANT_MAP.keys()) + _V3_NESTED_VARIANT_IDS}"
        )

    @property
    def hub_repo_id(self) -> str:
        """HuggingFace Hub repo ID for checkpoint download."""
        return _V3_HUB_REPOS.get(self.backbone, "")

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint filename on HuggingFace Hub."""
        return "model.safetensors"
