"""
Paired depth augmentation transforms.

All transforms accept and return a three-tuple::

    (pixel_values, depth_map, valid_mask)

where:
    pixel_values: ``torch.Tensor`` float32 ``(3, H, W)`` — RGB image
    depth_map:    ``torch.Tensor`` float32 ``(1, H, W)`` — depth in metres or relative
    valid_mask:   ``torch.Tensor`` bool    ``(1, H, W)`` — True where depth is valid

Spatial transforms (flip, crop, scale) apply identically to all three tensors.
Photometric transforms (color jitter, normalize) apply to ``pixel_values`` only.
"""

import random
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter


class Compose:
    """Chain multiple paired transforms together.

    Args:
        transforms: Sequence of callables, each accepting and returning
            ``(pixel_values, depth_map, valid_mask)``.
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            pixel_values, depth_map, valid_mask = t(pixel_values, depth_map, valid_mask)
        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append(")")
        return "\n".join(lines)


class PairedRandomHorizontalFlip:
    """Randomly flip all three tensors horizontally.

    Args:
        p: Probability of flipping. Default 0.5.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            pixel_values = torch.flip(pixel_values, dims=[-1])
            depth_map = torch.flip(depth_map, dims=[-1])
            valid_mask = torch.flip(valid_mask, dims=[-1])
        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class PairedRandomCrop:
    """Randomly crop all three tensors to the given size.

    The same crop box is applied to all three tensors.

    Args:
        size: Output ``(H, W)`` or a single int for a square crop.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        else:
            self.crop_h, self.crop_w = size

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h, w = pixel_values.shape
        if h < self.crop_h or w < self.crop_w:
            raise ValueError(
                f"Image size ({h}×{w}) is smaller than crop size "
                f"({self.crop_h}×{self.crop_w}). "
                "Use PairedRandomScale to enlarge the image first."
            )
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        pixel_values = pixel_values[:, top:top + self.crop_h, left:left + self.crop_w]
        depth_map = depth_map[:, top:top + self.crop_h, left:left + self.crop_w]
        valid_mask = valid_mask[:, top:top + self.crop_h, left:left + self.crop_w]
        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size=({self.crop_h}, {self.crop_w}))"


class PairedCenterCrop:
    """Deterministic center crop of all three tensors.

    Args:
        size: Output ``(H, W)`` or a single int for a square crop.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        else:
            self.crop_h, self.crop_w = size

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h, w = pixel_values.shape
        top = max(0, (h - self.crop_h) // 2)
        left = max(0, (w - self.crop_w) // 2)
        pixel_values = pixel_values[:, top:top + self.crop_h, left:left + self.crop_w]
        depth_map = depth_map[:, top:top + self.crop_h, left:left + self.crop_w]
        valid_mask = valid_mask[:, top:top + self.crop_h, left:left + self.crop_w]
        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size=({self.crop_h}, {self.crop_w}))"


class PairedRandomScale:
    """Randomly scale all three tensors by a factor drawn from ``scale_range``.

    - Image is resized with bilinear interpolation.
    - Depth map and mask are resized with nearest-neighbour interpolation
      (to preserve hole patterns in sparse ground truth).
    - Depth values are multiplied by the scale factor when ``scale_depth=True``
      (correct for metric datasets where scene geometry scales with the image).

    Args:
        scale_range: ``(min_scale, max_scale)``. Default ``(1.0, 2.0)``.
            Use values >= 1.0 when followed by a random crop to avoid
            producing images smaller than the crop size.
        scale_depth: If True, multiply depth values by the scale factor.
            Default True.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (1.0, 2.0),
        scale_depth: bool = True,
    ):
        self.scale_range = scale_range
        self.scale_depth = scale_depth

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        _, h, w = pixel_values.shape
        new_h = max(1, int(round(h * s)))
        new_w = max(1, int(round(w * s)))

        # F.interpolate requires 4D input (B, C, H, W)
        pixel_values = F.interpolate(
            pixel_values.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        depth_map = F.interpolate(
            depth_map.unsqueeze(0),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0)

        valid_mask = F.interpolate(
            valid_mask.float().unsqueeze(0),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0).bool()

        if self.scale_depth:
            depth_map = depth_map * s

        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale_range={self.scale_range}, scale_depth={self.scale_depth})"
        )


class PairedResize:
    """Resize all three tensors so the shorter spatial side equals ``size``.

    Useful as a pre-processing step before center-cropping to ensure the image
    is at least as large as the crop. Image uses bilinear interpolation; depth
    and mask use nearest-neighbour.

    Args:
        size: Target size for the shorter side (pixels).
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h, w = pixel_values.shape
        if min(h, w) == self.size:
            return pixel_values, depth_map, valid_mask

        if h <= w:
            new_h = self.size
            new_w = int(round(w * self.size / h))
        else:
            new_w = self.size
            new_h = int(round(h * self.size / w))

        pixel_values = F.interpolate(
            pixel_values.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        depth_map = F.interpolate(
            depth_map.unsqueeze(0),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0)

        valid_mask = F.interpolate(
            valid_mask.float().unsqueeze(0),
            size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0).bool()

        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class PairedColorJitter:
    """Apply random color jitter to the image only. Depth and mask pass through.

    Args:
        brightness: Brightness jitter range (float or (min, max)).
        contrast:   Contrast jitter range.
        saturation: Saturation jitter range.
        hue:        Hue jitter range.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self._jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_values = self._jitter(pixel_values)
        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._jitter})"


class PairedNormalize:
    """Normalize the image with mean and std. Depth and mask pass through.

    Args:
        mean: Per-channel mean. Default ImageNet mean.
        std:  Per-channel std.  Default ImageNet std.
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std:  Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(
        self,
        pixel_values: torch.Tensor,
        depth_map: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_values = TF.normalize(pixel_values, mean=self.mean, std=self.std)
        return pixel_values, depth_map, valid_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# ---------------------------------------------------------------------------
# Preset transform pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(input_size: int = 518) -> Compose:
    """Standard training transform pipeline.

    Applies: resize (shorter side ≥ input_size) → random scale → random crop
    → random horizontal flip → color jitter → ImageNet normalization.

    Args:
        input_size: Spatial size of the training crops. Default 518
            (Depth Anything v2 standard).

    Returns:
        :class:`Compose` transform.
    """
    return Compose([
        PairedResize(size=input_size),
        PairedRandomScale(scale_range=(1.0, 2.0), scale_depth=True),
        PairedRandomCrop(size=input_size),
        PairedRandomHorizontalFlip(p=0.5),
        PairedColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        PairedNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(input_size: int = 518) -> Compose:
    """Standard validation transform pipeline.

    Applies: resize (shorter side = input_size) → center crop →
    ImageNet normalization.

    Args:
        input_size: Spatial size of the validation crops. Default 518.

    Returns:
        :class:`Compose` transform.
    """
    return Compose([
        PairedResize(size=input_size),
        PairedCenterCrop(size=input_size),
        PairedNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
