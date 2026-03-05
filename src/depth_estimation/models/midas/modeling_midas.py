"""
MiDaS — Single-file model implementation.

Wraps HuggingFace transformers depth-estimation pipeline for Intel MiDaS variants.
Produces high-quality relative depth maps.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_midas import MiDaSConfig, _MIDAS_VARIANT_MAP

logger = logging.getLogger(__name__)


class MiDaSModel(BaseDepthModel):
    """Intel MiDaS depth estimation model.

    Uses HuggingFace transformers pipeline internally.
    Three variants: DPT-Large, DPT-Hybrid, BEiT-Large.

    Usage::

        model = MiDaSModel.from_pretrained("midas-dpt-large")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = MiDaSConfig

    def __init__(self, config: MiDaSConfig):
        super().__init__(config)
        self._pipeline = None
        self._device_str = "cpu"

    def _ensure_pipeline(self):
        """Lazy-load the HF pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "MiDaS requires the `transformers` package. "
                "Install with: pip install transformers"
            )
        device_id = 0 if self._device_str == "cuda" else -1
        self._pipeline = hf_pipeline(
            task="depth-estimation",
            model=self.config.hf_model_id,
            device=device_id,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W).
        """
        self._ensure_pipeline()
        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        depths = []
        for i in range(batch_size):
            # Convert tensor to PIL for the HF pipeline
            img_tensor = pixel_values[i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
            img_tensor = img_tensor * std + mean
            img_tensor = img_tensor.clamp(0, 1)
            img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Run HF pipeline
            outputs = self._pipeline(pil_image)
            depth_map = np.array(outputs["depth"])

            # Resize to match input
            depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)
            if depth_tensor.shape[2:] != (h, w):
                depth_tensor = F.interpolate(
                    depth_tensor, size=(h, w), mode="bilinear", align_corners=False
                )
            depths.append(depth_tensor.squeeze(0).squeeze(0))

        return torch.stack(depths)

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "MiDaSModel":
        """Load MiDaS model."""
        backbone = _MIDAS_VARIANT_MAP.get(model_id)
        if backbone is None:
            backbone = model_id

        config = MiDaSConfig(backbone=backbone)
        model = cls(config)
        model._device_str = device
        model._ensure_pipeline()

        logger.info(f"Loaded MiDaS ({config.display_name}) from {config.hf_model_id}")
        return model
