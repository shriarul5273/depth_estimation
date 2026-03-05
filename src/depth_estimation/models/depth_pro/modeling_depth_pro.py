"""
Apple DepthPro — Single-file model implementation.

Wraps Apple's ``depth_pro`` package for sharp monocular metric depth.
Requires ``depth_pro`` to be installed separately.

Weight loading via HuggingFace Hub ``apple/DepthPro``.
"""

import logging
import os
import tempfile
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_depth_pro import DepthProConfig, _DEPTHPRO_VARIANT_MAP

logger = logging.getLogger(__name__)


def _check_depth_pro_available():
    try:
        import depth_pro
        return True
    except ImportError:
        return False


class DepthProModel(BaseDepthModel):
    """Apple DepthPro model.

    Sharp monocular metric depth estimation.

    Usage::

        model = DepthProModel.from_pretrained("depth-pro")
        depth = model(pixel_values)  # (B, H, W) tensor in meters
    """

    config_class = DepthProConfig

    def __init__(self, config: DepthProConfig):
        super().__init__(config)
        self._model = None
        self._transform = None

    def _ensure_model(self):
        """Lazy-load DepthPro model and transforms."""
        if self._model is not None:
            return
        if not _check_depth_pro_available():
            raise ImportError(
                "DepthPro requires the `depth_pro` package. "
                "Install from: https://github.com/apple/ml-depth-pro"
            )
        import depth_pro
        from huggingface_hub import hf_hub_download

        # Download checkpoint from HF Hub
        checkpoint_path = hf_hub_download(
            repo_id=self.config.hub_repo_id,
            filename=self.config.hub_filename,
            repo_type="model",
        )

        # Create symlink for depth_pro's expected path
        local_dir = "./checkpoints"
        local_path = os.path.join(local_dir, "depth_pro.pt")
        if not os.path.exists(local_path):
            os.makedirs(local_dir, exist_ok=True)
            os.symlink(checkpoint_path, local_path)

        device = torch.device(_auto_detect_device())
        model, transform = depth_pro.create_model_and_transforms(device=device)
        model = model.to(device).eval()
        self._model = model
        self._transform = transform

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W) in meters.
        """
        self._ensure_model()
        import depth_pro

        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(pixel_values.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(pixel_values.device)

        depths = []
        for i in range(batch_size):
            img = pixel_values[i] * std + mean
            img = img.clamp(0, 1)
            img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Save temporarily for depth_pro.load_rgb
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = f.name
                Image.fromarray(img_np).save(temp_path)

            try:
                image, _, f_px = depth_pro.load_rgb(temp_path)
                image = self._transform(image)
                image = image.to(next(self._model.parameters()).device)

                with torch.no_grad():
                    prediction = self._model.infer(image, f_px=f_px)

                depth = prediction["depth"].cpu()

                # Resize to match input
                if depth.shape != (h, w):
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0).unsqueeze(0).float(),
                        size=(h, w), mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
                depths.append(depth)
            finally:
                os.unlink(temp_path)

        return torch.stack(depths)

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthProModel":
        """Load DepthPro model."""
        config = DepthProConfig()
        model = cls(config)
        model._ensure_model()

        logger.info("Loaded Apple DepthPro from apple/DepthPro")
        return model
