"""
BaseDepthConfig — Base configuration class for all depth estimation models.

Stores model-level metadata needed to instantiate a model and its processor.
Subclasses override default values; no new logic needed for simple configs.
"""

import copy
import json
from typing import Any, Dict, List, Optional


class BaseDepthConfig:
    """Base configuration for depth estimation models.

    Every field has a sensible default so users can override only what they need.
    Subclasses typically only change default values (Transformers modular pattern).

    Attributes:
        model_type: Unique identifier for the model family.
        backbone: Encoder backbone name (e.g. "vits", "vitb", "vitl").
        input_size: Expected input resolution (square).
        patch_size: Vision Transformer patch size.
        embed_dim: Embedding dimension of the backbone.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        features: Number of features in the DPT head.
        out_channels: Channel dimensions for each DPT stage.
        is_metric: Whether the model produces metric (absolute) depth.
        max_depth: Maximum depth value (for metric models).
        min_depth: Minimum depth value (for metric models).
        mean: Per-channel normalization mean (default: ImageNet).
        std: Per-channel normalization std (default: ImageNet).
    """

    model_type: str = "base"

    def __init__(
        self,
        backbone: str = "vitl",
        input_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        features: int = 256,
        out_channels: Optional[List[int]] = None,
        is_metric: bool = False,
        max_depth: Optional[float] = None,
        min_depth: Optional[float] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        **kwargs: Any,
    ):
        self.backbone = backbone
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.features = features
        self.out_channels = out_channels or [256, 512, 1024, 1024]
        self.is_metric = is_metric
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        # Store any extra kwargs for forward compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-round-trippable dictionary."""
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.model_type
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseDepthConfig":
        """Instantiate a config from a dictionary."""
        config_dict = copy.deepcopy(config_dict)
        config_dict.pop("model_type", None)
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str) -> None:
        """Save config to a JSON file."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "BaseDepthConfig":
        """Load config from a directory containing config.json."""
        import os
        config_path = os.path.join(pretrained_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({json.dumps(self.to_dict(), indent=2)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseDepthConfig):
            return False
        return self.to_dict() == other.to_dict()
