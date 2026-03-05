"""
BaseDepthModel — Base model class for all depth estimation models.

Provides weight loading, device management, and inference context.
Subclasses implement forward() with their specific network architecture.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .configuration_utils import BaseDepthConfig

logger = logging.getLogger(__name__)


class BaseDepthModel(nn.Module):
    """Base wrapper around depth estimation neural networks.

    Subclasses must implement:
        - ``forward(pixel_values) → torch.Tensor``  (raw depth tensor)
        - ``_build_model(config)``  (construct the network architecture)
        - ``_load_pretrained_weights(model_id, device)`` (load checkpoint)

    Provides:
        - ``from_pretrained(model_id)`` for weight loading.
        - Device management via ``to(device)``, ``half()``, ``float()``.
        - Automatic ``torch.no_grad()`` context during inference.
    """

    config_class: type = BaseDepthConfig

    def __init__(self, config: BaseDepthConfig):
        super().__init__()
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            pixel_values: Input tensor of shape (B, 3, H, W), normalized.

        Returns:
            Depth tensor of shape (B, H, W) or (B, 1, H, W).
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "BaseDepthModel":
        """Load a pretrained model.

        Args:
            model_id: Model identifier string (e.g. "depth-anything-v1-vitb")
                or a local path to a checkpoint.
            device: Device to load the model to. Auto-detected if None.
            **kwargs: Additional arguments passed to the model constructor.

        Returns:
            Instantiated model with pretrained weights, in eval mode.
        """
        if device is None:
            device = _auto_detect_device()

        model = cls._load_pretrained_weights(model_id, device=device, **kwargs)
        model = model.to(device).eval()
        logger.info(f"Loaded {cls.__name__} from '{model_id}' on {device}")
        return model

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "BaseDepthModel":
        """Load weights from a pretrained checkpoint.

        Subclasses must override this to implement their specific loading logic
        (HuggingFace Hub, local .pth, etc.).
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _load_pretrained_weights()"
        )

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run inference with no_grad context.

        Args:
            pixel_values: Input tensor of shape (B, 3, H, W).

        Returns:
            Depth tensor.
        """
        return self.forward(pixel_values)


def _auto_detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
