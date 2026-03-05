"""
Pixel-Perfect Depth — Single-file model implementation.

Wraps Pixel-Perfect Depth (PPD) + MoGe for metric depth estimation.
Requires ``ppd`` and ``moge`` packages (optional dependencies).
"""

import logging
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_ppd import PixelPerfectDepthConfig, _PPD_VARIANT_MAP

logger = logging.getLogger(__name__)


def _check_ppd_available():
    try:
        from ppd.models.ppd import PixelPerfectDepth
        from moge.model.v2 import MoGeModel
        return True
    except ImportError:
        return False


class PixelPerfectDepthModel(BaseDepthModel):
    """Pixel-Perfect Depth model.

    Combines PPD (relative depth via diffusion) with MoGe (metric alignment).

    Usage::

        model = PixelPerfectDepthModel.from_pretrained("pixel-perfect-depth")
        depth = model(pixel_values)  # (B, H, W) tensor in meters
    """

    config_class = PixelPerfectDepthConfig

    def __init__(self, config: PixelPerfectDepthConfig):
        super().__init__(config)
        self._ppd_model = None
        self._moge_model = None

    def _ensure_models(self):
        """Lazy-load PPD and MoGe models."""
        if self._ppd_model is not None:
            return
        if not _check_ppd_available():
            raise ImportError(
                "Pixel-Perfect Depth requires `ppd` and `moge` packages. "
                "See: https://github.com/gangweix/Pixel-Perfect-Depth"
            )
        from ppd.models.ppd import PixelPerfectDepth
        from moge.model.v2 import MoGeModel
        from huggingface_hub import hf_hub_download

        device = torch.device(_auto_detect_device())

        # Load PPD
        ppd_model = PixelPerfectDepth(sampling_steps=self.config.sampling_steps)
        ckpt_path = hf_hub_download(
            repo_id=self.config.ppd_hub_repo_id,
            filename=self.config.ppd_hub_filename,
            repo_type="model",
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        ppd_model.load_state_dict(state_dict, strict=False)
        ppd_model = ppd_model.to(device).eval()
        self._ppd_model = ppd_model

        # Load MoGe
        moge_model = MoGeModel.from_pretrained(self.config.moge_model_id).eval()
        moge_model = moge_model.to(device)
        self._moge_model = moge_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W) in meters.
        """
        self._ensure_models()
        from ppd.utils.align_depth_func import recover_metric_depth_ransac

        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(pixel_values.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(pixel_values.device)

        depths = []
        device = next(self._ppd_model.parameters()).device

        for i in range(batch_size):
            img = pixel_values[i] * std + mean
            img = img.clamp(0, 1)
            img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # PPD: relative depth
            with torch.no_grad():
                depth_rel, resize_image = self._ppd_model.infer_image(
                    img_bgr, sampling_steps=self.config.sampling_steps
                )

            # MoGe: metric depth
            rgb_tensor = torch.tensor(
                cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB) / 255,
                dtype=torch.float32, device=device,
            ).permute(2, 0, 1)

            with torch.no_grad():
                metric_depth, mask, intrinsics = self._moge_model.infer(rgb_tensor)

            metric_depth[~mask] = metric_depth[mask].max()
            metric_depth_aligned = recover_metric_depth_ransac(
                depth_rel, metric_depth, mask
            )

            # Resize to input dimensions
            depth_full = cv2.resize(
                metric_depth_aligned, (w, h), interpolation=cv2.INTER_LINEAR
            )
            depths.append(torch.from_numpy(depth_full).float())

        return torch.stack(depths)

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "PixelPerfectDepthModel":
        """Load PPD model."""
        config = PixelPerfectDepthConfig()
        model = cls(config)
        model._ensure_models()

        logger.info("Loaded Pixel-Perfect Depth")
        return model
