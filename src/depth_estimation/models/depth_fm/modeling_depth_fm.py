"""
DepthFM — Single-file model implementation.

Wraps the ``depthfm`` package's DepthFM model for fast monocular depth
estimation using flow matching with ODE solving.

Requires: ``depthfm``, ``diffusers``, ``torchdiffeq``, ``einops``
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_depth_fm import DepthFMConfig, _DEPTHFM_VARIANT_MAP

logger = logging.getLogger(__name__)


def _check_depthfm_available():
    try:
        from depthfm import DepthFM
        return True
    except ImportError:
        return False


class DepthFMModel(BaseDepthModel):
    """DepthFM: Fast Monocular Depth Estimation with Flow Matching.

    Uses a UNet + Stable Diffusion VAE with ODE solving for fast inference.

    Usage::

        model = DepthFMModel.from_pretrained("depth-fm")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = DepthFMConfig

    def __init__(self, config: DepthFMConfig):
        super().__init__(config)
        self._model = None

    def _ensure_model(self):
        """Lazy-load the DepthFM model."""
        if self._model is not None:
            return
        if not _check_depthfm_available():
            raise ImportError(
                "DepthFM requires the `depthfm` package. "
                "Install from: https://github.com/CompVis/depth-fm"
            )
        from depthfm import DepthFM
        from huggingface_hub import hf_hub_download

        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id=self.config.hub_repo_id,
            filename=self.config.hub_filename,
            repo_type="model",
        )

        device = _auto_detect_device()
        model = DepthFM(ckpt_path)
        model = model.to(device).eval()
        self._model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized with
                ImageNet mean/std.

        Returns:
            Depth tensor (B, H, W) in [0, 1].
        """
        self._ensure_model()
        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        # DepthFM expects input in [-1, 1] range, not ImageNet normalized
        # Denormalize from ImageNet → [0, 1] → [-1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(pixel_values.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(pixel_values.device)
        imgs = pixel_values * std + mean  # [0, 1]
        imgs = imgs * 2.0 - 1.0  # [-1, 1]

        # Resize to be divisible by 64
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode="reflect")

        device = next(self._model.parameters()).device
        imgs = imgs.to(device)

        with torch.no_grad():
            depth = self._model.predict_depth(
                imgs,
                num_steps=self.config.num_steps,
                ensemble_size=self.config.ensemble_size,
            )

        # depth shape: (B, 1, H_padded, W_padded)
        depth = depth[:, :, :h, :w]  # remove padding
        depth = depth.squeeze(1)  # (B, H, W)

        return depth

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthFMModel":
        """Load DepthFM model."""
        config = DepthFMConfig()
        model = cls(config)
        model._ensure_model()

        logger.info(f"Loaded DepthFM from {config.hub_repo_id}")
        return model
