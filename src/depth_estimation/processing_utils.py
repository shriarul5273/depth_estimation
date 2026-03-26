"""
DepthProcessor — Shared processor for all depth estimation models.

Handles image-to-tensor (preprocess) and tensor-to-output (postprocess)
transformations. NOT duplicated per model — reads parameters from the config.
"""

import io
import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import matplotlib
from PIL import Image

from .configuration_utils import BaseDepthConfig
from .output import DepthOutput

logger = logging.getLogger(__name__)


class DepthProcessor:
    """Shared image processor for depth estimation models.

    Preprocessing pipeline:
        1. Load image (path, URL, PIL Image, or NumPy array).
        2. Convert to RGB.
        3. Resize to model's expected input_size (preserving aspect ratio,
           ensuring dimensions are multiples of patch_size).
        4. Normalize pixel values (ImageNet mean/std by default, overridable).
        5. Convert to torch.Tensor and add batch dimension.

    Postprocessing pipeline:
        1. Squeeze batch dimension.
        2. Resize depth map back to original image resolution.
        3. Normalize depth values to [0, 1] range.
        4. Optionally apply colormap (default: Spectral_r).
        5. Return DepthOutput.
    """

    def __init__(self, config: Optional[BaseDepthConfig] = None):
        if config is None:
            config = BaseDepthConfig()

        self.input_size = config.input_size
        self.patch_size = config.patch_size
        self.mean = config.mean
        self.std = config.std

    @classmethod
    def from_config(cls, config: BaseDepthConfig) -> "DepthProcessor":
        """Create a processor from a model config."""
        return cls(config=config)

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        images: Union[str, Image.Image, np.ndarray, List],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Preprocess one or more images for model inference.

        Args:
            images: A single image or list of images. Each image can be:
                - A file path (str)
                - A URL (str starting with http)
                - A PIL Image
                - A NumPy array (H, W, 3), BGR or RGB
            return_tensors: Return format, currently only "pt" (PyTorch).

        Returns:
            Dictionary with:
                - "pixel_values": Tensor of shape (B, 3, H, W)
                - "original_sizes": List of (H, W) tuples
        """
        if not isinstance(images, list):
            images = [images]

        pixel_values_list = []
        original_sizes = []

        for image in images:
            img = self._load_image(image)
            original_sizes.append((img.shape[0], img.shape[1]))
            tensor = self._transform(img)
            pixel_values_list.append(tensor)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        return {
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
        }

    def __call__(
        self,
        images: Union[str, Image.Image, np.ndarray, List],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Alias for preprocess()."""
        return self.preprocess(images, return_tensors=return_tensors)

    # ------------------------------------------------------------------ #
    #  Postprocessing
    # ------------------------------------------------------------------ #

    def postprocess(
        self,
        depth_tensor: torch.Tensor,
        original_sizes: List[Tuple[int, int]],
        colorize: bool = True,
        colormap: str = "Spectral_r",
    ) -> Union[DepthOutput, List[DepthOutput]]:
        """Convert raw model output to DepthOutput(s).

        Args:
            depth_tensor: Raw depth from model, shape (B, H, W) or (B, 1, H, W).
            original_sizes: Original (H, W) for each image in the batch.
            colorize: Whether to produce a colored depth visualization.
            colormap: Matplotlib colormap name.

        Returns:
            A single DepthOutput if batch size is 1, otherwise a list.
        """
        if depth_tensor.dim() == 4:
            depth_tensor = depth_tensor.squeeze(1)

        outputs = []
        batch_size = depth_tensor.shape[0]

        for i in range(batch_size):
            depth = depth_tensor[i]
            h, w = original_sizes[i]

            # Resize to original resolution
            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            depth_np = depth_resized.cpu().numpy().astype(np.float32)

            # Normalize to [0, 1]
            d_min, d_max = depth_np.min(), depth_np.max()
            if d_max - d_min > 1e-8:
                depth_norm = (depth_np - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_np)

            # Colorize
            colored = None
            if colorize:
                colored = self._colorize(depth_norm, colormap)

            outputs.append(
                DepthOutput(
                    depth=depth_norm,
                    colored_depth=colored,
                    metadata={},
                )
            )

        return outputs[0] if len(outputs) == 1 else outputs

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Load an image from various sources into a NumPy RGB array."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            raise ValueError(f"Expected (H, W, 3) array, got shape {image.shape}")

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                return self._load_from_url(image)
            return self._load_from_path(image)

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _load_from_path(path: str) -> np.ndarray:
        """Load image from a local file path."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from '{path}'")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_from_url(url: str) -> np.ndarray:
        """Load image from a URL."""
        import urllib.request
        with urllib.request.urlopen(url) as response:
            data = response.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)

    def _transform(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Apply resize + normalize + convert to tensor.

        Matches the Depth Anything preprocessing:
            - Resize so the shorter side >= input_size, keep aspect ratio,
              ensure both dims are multiples of patch_size.
            - Normalize with ImageNet mean/std.
            - Return tensor of shape (1, 3, H, W).
        """
        h, w = image_rgb.shape[:2]
        target = self.input_size

        # Compute scale so shorter side >= target
        scale = target / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Ensure dimensions are multiples of patch_size
        new_h = new_h - (new_h % self.patch_size)
        new_w = new_w - (new_w % self.patch_size)
        new_h = max(new_h, self.patch_size)
        new_w = max(new_w, self.patch_size)

        resized = cv2.resize(
            image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )

        # Normalize to [0, 1] then apply mean/std
        img = resized.astype(np.float32) / 255.0
        img = (img - np.array(self.mean)) / np.array(self.std)

        # HWC → CHW, add batch dim
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    @staticmethod
    def _colorize(depth_norm: np.ndarray, colormap: str = "Spectral_r") -> np.ndarray:
        """Apply a matplotlib colormap to a [0, 1] depth map.

        Returns an (H, W, 3) uint8 RGB array.
        """
        cmap = matplotlib.colormaps.get_cmap(colormap)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        colored = (cmap(depth_uint8)[:, :, :3] * 255).astype(np.uint8)
        return colored
