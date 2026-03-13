"""Abstract base class for all depth estimation datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseDepthDataset(ABC):
    """Abstract base for all depth datasets.

    Subclasses must implement:
        ``__len__() -> int``
        ``_load_sample(index) -> (rgb uint8 HxWx3, depth float32 HxW)``

    ``__getitem__`` returns a dict with:
        pixel_values  – ``(3, H, W)`` float32 tensor, normalised to ``[0, 1]``
        depth_map     – ``(1, H, W)`` float32 tensor (metres or relative)
        valid_mask    – ``(1, H, W)`` bool tensor

    Args:
        transform:  Callable ``(pixel_values, depth_map, valid_mask) → ...``
                    applied after tensor conversion.
        min_depth:  Pixels with depth below this value are masked out.
        max_depth:  Pixels with depth above this value are masked out.
    """

    def __init__(
        self,
        transform=None,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
    ) -> None:
        self.transform = transform
        self.min_depth = min_depth
        self.max_depth = max_depth

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(rgb, depth)`` for *index*.

        Returns:
            rgb:   ``uint8`` array of shape ``(H, W, 3)``.
            depth: ``float32`` array of shape ``(H, W)``, in metres or
                   relative units depending on the dataset.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        image, depth = self._load_sample(index)

        pixel_values = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        depth_map = torch.from_numpy(depth).unsqueeze(0).float()
        valid_mask = (depth_map > self.min_depth) & (depth_map < self.max_depth)

        if self.transform is not None:
            pixel_values, depth_map, valid_mask = self.transform(
                pixel_values, depth_map, valid_mask
            )

        return {
            "pixel_values": pixel_values,
            "depth_map": depth_map,
            "valid_mask": valid_mask,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={len(self)}, "
            f"min_depth={self.min_depth}, "
            f"max_depth={self.max_depth})"
        )
