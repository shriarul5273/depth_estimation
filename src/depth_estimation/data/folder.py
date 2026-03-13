"""Generic folder dataset — load paired RGB + depth from two directories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base_dataset import BaseDepthDataset

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_DEPTH_EXTS = {".npy", ".npz", ".png", ".tiff", ".tif", ".exr"}


class FolderDataset(BaseDepthDataset):
    """Load RGB + depth pairs from two parallel directories.

    Directory layout (files matched by stem)::

        image_dir/
          001.jpg
          002.jpg
        depth_dir/
          001.npy   ← float32 metres or relative
          002.png   ← 16-bit PNG (divided by depth_scale to get metres)

    Supports depth formats:
        - ``.npy`` / ``.npz``: loaded directly with ``np.load``.
        - ``.png`` / ``.tiff``: loaded as uint16 and divided by *depth_scale*.
        - ``.exr``: loaded via OpenCV (requires ``opencv-python``).

    Args:
        image_dir:   Directory of RGB images.
        depth_dir:   Directory of depth maps.  If ``None``, only RGB is
                     returned (depth_map will be all-zero, valid_mask all-False).
        depth_scale: Divisor applied to integer depth files (PNG/TIFF) to
                     convert to metres.  Default ``256.0`` (KITTI convention).
        transform:   Paired callable.
        min_depth:   Minimum valid depth. Default ``1e-3``.
        max_depth:   Maximum valid depth. Default ``1000.0``.

    Example::

        ds = FolderDataset(
            image_dir="data/rgb",
            depth_dir="data/depth",
            depth_scale=1000.0,  # millimetres → metres
        )
        sample = ds[0]
    """

    def __init__(
        self,
        image_dir: str | Path,
        depth_dir: Optional[str | Path] = None,
        depth_scale: float = 256.0,
        transform=None,
        min_depth: float = 1e-3,
        max_depth: float = 1000.0,
    ) -> None:
        super().__init__(transform=transform, min_depth=min_depth, max_depth=max_depth)
        self.image_dir = Path(image_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.depth_scale = depth_scale

        if not self.image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")
        if self.depth_dir is not None and not self.depth_dir.exists():
            raise FileNotFoundError(f"depth_dir not found: {self.depth_dir}")

        self._samples = self._collect_samples()

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def _collect_samples(self) -> List[dict]:
        image_paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )

        if not image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        samples: List[dict] = []
        missing_depth = 0

        for rgb_path in image_paths:
            depth_path: Optional[Path] = None
            if self.depth_dir is not None:
                # Try matching by stem with every supported depth extension
                for ext in _DEPTH_EXTS:
                    candidate = self.depth_dir / (rgb_path.stem + ext)
                    if candidate.exists():
                        depth_path = candidate
                        break
                if depth_path is None:
                    missing_depth += 1

            samples.append({"rgb": rgb_path, "depth": depth_path})

        if missing_depth > 0:
            logger.warning(
                "%d / %d images have no matching depth file in %s",
                missing_depth, len(samples), self.depth_dir,
            )

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        s = self._samples[index]
        image = np.array(Image.open(s["rgb"]).convert("RGB"), dtype=np.uint8)
        H, W = image.shape[:2]

        depth_path: Optional[Path] = s["depth"]
        if depth_path is None:
            depth = np.zeros((H, W), dtype=np.float32)
            return image, depth

        ext = depth_path.suffix.lower()

        if ext == ".npy":
            depth = np.load(depth_path).squeeze().astype(np.float32)

        elif ext == ".npz":
            data = np.load(depth_path)
            key = "depth" if "depth" in data else next(iter(data))
            depth = data[key].squeeze().astype(np.float32)

        elif ext == ".exr":
            import cv2
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH).astype(np.float32)

        else:
            # PNG / TIFF — typically 16-bit integers
            depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
            depth = depth_raw / self.depth_scale

        return image, depth

    def __repr__(self) -> str:
        return (
            f"FolderDataset("
            f"n={len(self)}, "
            f"image_dir={self.image_dir}, "
            f"depth_dir={self.depth_dir})"
        )
