"""NYU Depth V2 — labeled set (1 449 images, 480 × 640).

Reference:
    Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images"
    ECCV 2012.  http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

Files downloaded automatically on first use (~2.8 GB + tiny split file):
    Dataset:  http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    Splits:   http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat

Requires:  h5py  (``pip install h5py``)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .base_dataset import BaseDepthDataset
from .hub import download_file, get_cache_dir

logger = logging.getLogger(__name__)

_MAT_URL = (
    "http://horatio.cs.nyu.edu/mit/silberman/"
    "nyu_depth_v2/nyu_depth_v2_labeled.mat"
)
_SPLIT_URL = (
    "http://horatio.cs.nyu.edu/mit/silberman/"
    "indoor_seg_sup/splits.mat"
)
# Fallback split sizes when splits.mat is unavailable
_N_TRAIN = 795
_N_TOTAL = 1449


class NYUDepthV2Dataset(BaseDepthDataset):
    """NYU Depth V2 labeled set.

    Depth values are in **metres** (from a Microsoft Kinect sensor).
    Train / test split follows Eigen et al.: 795 train, 654 test.

    Args:
        root:       Directory containing (or to receive) the ``.mat`` files.
                    Defaults to ``~/.cache/depth_estimation/datasets/nyu_depth_v2/``.
        split:      ``"train"`` or ``"test"``.
        transform:  Paired callable ``(pixel_values, depth_map, valid_mask) → ...``
        min_depth:  Minimum valid depth in metres. Default ``1e-3``.
        max_depth:  Maximum valid depth in metres. Default ``10.0``.
        download:   Auto-download when files are missing. Default ``True``.

    Example::

        ds = NYUDepthV2Dataset(split="test")
        sample = ds[0]
        # sample["pixel_values"]  → (3, 480, 640) float32
        # sample["depth_map"]     → (1, 480, 640) float32  (metres)
        # sample["valid_mask"]    → (1, 480, 640) bool
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: str = "train",
        transform=None,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        download: bool = True,
    ) -> None:
        super().__init__(transform=transform, min_depth=min_depth, max_depth=max_depth)
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.split = split
        self.root = Path(root) if root else get_cache_dir("nyu_depth_v2")

        if download:
            self._download()

        self._mat_path = self.root / "nyu_depth_v2_labeled.mat"
        self._split_path = self.root / "splits.mat"

        if not self._mat_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self._mat_path}.\n"
                "Pass download=True or download manually from:\n"
                f"  {_MAT_URL}"
            )

        self._indices = self._load_split_indices()
        # h5py handle opened lazily so DataLoader workers can pickle this object
        self._h5 = None

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self) -> None:
        mat_dest = self.root / "nyu_depth_v2_labeled.mat"
        split_dest = self.root / "splits.mat"

        if mat_dest.exists():
            logger.info("NYU Depth V2: already downloaded at %s", self.root)
        else:
            logger.info("NYU Depth V2: downloading labeled set (~2.8 GB) …")
            download_file(_MAT_URL, mat_dest, desc="nyu_depth_v2_labeled.mat")

        if not split_dest.exists():
            download_file(_SPLIT_URL, split_dest, desc="splits.mat")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_split_indices(self) -> np.ndarray:
        """Return 0-based sample indices for the requested split."""
        try:
            import h5py

            with h5py.File(self._split_path, "r") as f:
                key = "testNdxs" if self.split == "test" else "trainNdxs"
                # MATLAB indices are 1-based
                indices = np.array(f[key]).flatten().astype(int) - 1
        except Exception as exc:
            logger.warning(
                "Could not load splits.mat (%s); falling back to default split.", exc
            )
            all_idx = np.arange(_N_TOTAL)
            indices = all_idx[_N_TRAIN:] if self.split == "test" else all_idx[:_N_TRAIN]

        return np.sort(indices)

    def _open_h5(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self._mat_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._indices)

    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        f = self._open_h5()
        idx = self._indices[index]

        # The .mat (MATLAB v7.3 / HDF5) stores arrays in column-major order.
        # h5py reads them transposed: images[i] → (C=3, W=640, H=480).
        # We transpose to standard (H, W, C) / (H, W).
        image = np.array(f["images"][idx]).transpose(2, 1, 0)  # (H=480, W=640, C=3)
        depth = np.array(f["depths"][idx]).transpose(1, 0)      # (H=480, W=640)

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image, depth.astype(np.float32)

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"NYUDepthV2Dataset("
            f"split={self.split!r}, "
            f"n={len(self)}, "
            f"root={self.root})"
        )
