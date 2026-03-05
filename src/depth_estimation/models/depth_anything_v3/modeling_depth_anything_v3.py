"""
Depth Anything v3 — Single-file model implementation.

Wraps the ``depth_anything_3`` package's DepthAnything3 API.
Requires ``depth_anything_3`` to be installed separately.

Weight loading via HuggingFace Hub ``from_pretrained``.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_depth_anything_v3 import DepthAnythingV3Config, _DA3_VARIANT_MAP

logger = logging.getLogger(__name__)


def _check_da3_available():
    try:
        from depth_anything_3.api import DepthAnything3
        return True
    except ImportError:
        return False


class DepthAnythingV3Model(BaseDepthModel):
    """Depth Anything v3 model.

    Wraps the ``depth_anything_3`` package's DepthAnything3 API.

    Usage::

        model = DepthAnythingV3Model.from_pretrained("depth-anything-v3-large")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = DepthAnythingV3Config

    def __init__(self, config: DepthAnythingV3Config):
        super().__init__(config)
        self._da3_model = None

    def _ensure_model(self):
        """Lazy-load the DA3 model."""
        if self._da3_model is not None:
            return
        if not _check_da3_available():
            raise ImportError(
                "Depth Anything v3 requires the `depth_anything_3` package. "
                "Install with: pip install depth-anything-3"
            )
        from depth_anything_3.api import DepthAnything3
        self._da3_model = DepthAnything3.from_pretrained(self.config.hub_repo_id)
        device = _auto_detect_device()
        self._da3_model = self._da3_model.to(device=torch.device(device))
        self._da3_model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W).
        """
        self._ensure_model()
        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        # Convert tensor batch to list of PIL images for DA3 API
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(pixel_values.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(pixel_values.device)

        pil_images = []
        for i in range(batch_size):
            img = pixel_values[i] * std + mean
            img = img.clamp(0, 1)
            img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        # Run DA3 inference
        prediction = self._da3_model.inference(
            image=pil_images,
            process_res=None,
            process_res_method="keep",
        )

        # Extract depth maps
        depth_maps = prediction.depth  # numpy array
        if isinstance(depth_maps, np.ndarray):
            depth_tensor = torch.from_numpy(depth_maps).float()
        else:
            depth_tensor = torch.tensor(depth_maps).float()

        # Ensure correct shape (B, H, W)
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        # Resize to input dimensions if needed
        if depth_tensor.shape[1:] != (h, w):
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(1), size=(h, w),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        return depth_tensor

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthAnythingV3Model":
        """Load DA v3 model from HuggingFace Hub."""
        backbone = _DA3_VARIANT_MAP.get(model_id)
        if backbone is None:
            backbone = model_id

        config = DepthAnythingV3Config(backbone=backbone)
        model = cls(config)
        model._ensure_model()

        logger.info(f"Loaded Depth Anything v3 ({backbone}) from {config.hub_repo_id}")
        return model
