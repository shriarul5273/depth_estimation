"""
MiDaS — Single-file model implementation.

Loads DPTForDepthEstimation directly as an nn.Module so all parameters
participate correctly in .to(device) / .eval() / .half() calls.
Produces high-quality relative depth maps.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F

from ...modeling_utils import BaseDepthModel
from .configuration_midas import MiDaSConfig, _MIDAS_VARIANT_MAP

logger = logging.getLogger(__name__)


class MiDaSModel(BaseDepthModel):
    """Intel MiDaS depth estimation model.

    Wraps DPTForDepthEstimation directly — weights participate in
    .to(device) / .eval() / .half() without any device mismatch.

    Three variants: DPT-Large, DPT-Hybrid, BEiT-Large.

    Usage::

        model = MiDaSModel.from_pretrained("midas-dpt-large")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = MiDaSConfig

    def __init__(self, config: MiDaSConfig):
        super().__init__(config)
        self.dpt = None  # DPTForDepthEstimation, set in _load_pretrained_weights

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: ``(B, 3, H, W)`` float32, ImageNet-normalised.

        Returns:
            Depth tensor ``(B, H, W)``.
        """
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        outputs = self.dpt(pixel_values=pixel_values)
        pred = outputs.predicted_depth  # (B, H_out, W_out)

        if pred.shape[-2:] != (h, w):
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return pred  # (B, H, W)

    def _backbone_module(self):
        """Return the HF DPT ViT encoder (dpt.dpt.encoder)."""
        if self.dpt is None:
            raise RuntimeError(
                "MiDaSModel is not yet loaded. Call from_pretrained() first."
            )
        return self.dpt.dpt.encoder

    def unfreeze_top_k_backbone_layers(self, k: int) -> None:
        """Unfreeze the last k ViT encoder layers of the HF DPT model.

        Overrides the DINOv2-specific base implementation.
        The HF DPT encoder layers are at ``dpt.dpt.encoder.layer``.

        Args:
            k: Number of encoder layers to unfreeze from the top.
        """
        encoder = self._backbone_module()
        layers = list(encoder.layer)
        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info(
            f"Unfroze top {k} MiDaS encoder layers. "
            f"Trainable params: {self._count_trainable():,}"
        )

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "MiDaSModel":
        """Load MiDaS model weights from HuggingFace Hub."""
        try:
            from transformers import DPTForDepthEstimation
        except ImportError:
            raise ImportError(
                "MiDaS requires the `transformers` package. "
                "Install with: pip install transformers"
            )

        backbone = _MIDAS_VARIANT_MAP.get(model_id)
        if backbone is None:
            backbone = model_id

        config = MiDaSConfig(backbone=backbone)

        logger.info("Loading MiDaS (%s) from %s …", config.display_name, config.hf_model_id)

        # low_cpu_mem_usage=False avoids accelerate's meta-device placement,
        # which (when CUDA is available) can scatter tensors across devices and
        # cause "indices not on same device" errors during the DPT forward pass.
        dpt = DPTForDepthEstimation.from_pretrained(
            config.hf_model_id,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        )

        model = cls(config)
        model.dpt = dpt
        # Move every parameter and buffer to the target device in one shot.
        model = model.to(device)
        model.eval()

        logger.info("Loaded MiDaS (%s) on %s", config.display_name, device)
        return model
