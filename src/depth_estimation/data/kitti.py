"""KITTI Eigen split for monocular depth estimation.

KITTI requires registration before download — files cannot be fetched
automatically.  See download instructions below.

Download instructions
---------------------
1. Register at https://www.cvlibs.net/datasets/kitti/index.php
2. Download the raw data (city / residential / road / campus / person):
   https://www.cvlibs.net/datasets/kitti/raw_data.php
3. Download the improved ground-truth depth (Garg/Eigen dense GT):
   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
4. Download the Eigen split file lists (train / val / test):
   https://github.com/cleinc/bts/tree/master/utils/filenames

Expected directory layout
--------------------------
::

    root/
      2011_09_26/
        2011_09_26_drive_0001_sync/
          image_02/data/
            0000000000.png
            ...
          velodyne_points/data/
            0000000000.bin
      ...
      data_depth_annotated/
        train/
          2011_09_26_drive_0001_sync/
            proj_depth/groundtruth/image_02/
              0000000005.png
              ...
        val/
          ...

    eigen_train_files.txt
    eigen_val_files.txt
    eigen_test_files.txt

Each line in the split files has the format::

    2011_09_26/2011_09_26_drive_0001_sync 0000000005 l

Depth maps from ``data_depth_annotated`` are 16-bit PNG files where
the integer value divided by 256.0 gives depth in **metres**.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base_dataset import BaseDepthDataset

logger = logging.getLogger(__name__)

_SPLIT_URLS = {
    "train": "https://raw.githubusercontent.com/cleinc/bts/master/pytorch/filenames/eigen_train_files_with_gt.txt",
    "val":   "https://raw.githubusercontent.com/cleinc/bts/master/pytorch/filenames/eigen_val_files_with_gt.txt",
    "test":  "https://raw.githubusercontent.com/cleinc/bts/master/pytorch/filenames/eigen_test_files_with_gt.txt",
}


class KITTIEigenDataset(BaseDepthDataset):
    """KITTI Eigen split — monocular depth estimation benchmark.

    Depth values are in **metres**.  Ground-truth is sparse projected
    LiDAR (~5 % of pixels); ``valid_mask`` covers only those pixels.

    Args:
        root:         Path to the KITTI raw data root (see module docstring).
        split:        ``"train"``, ``"val"``, or ``"test"``.
        filenames:    Path to the split file list (downloaded automatically
                      from the BTS repo if not provided).
        transform:    Paired callable ``(pixel_values, depth_map, valid_mask) → ...``
        min_depth:    Minimum valid depth in metres. Default ``1e-3``.
        max_depth:    Maximum valid depth in metres. Default ``80.0``.
        download_split: Download the split .txt file automatically.
                        The raw data itself must be downloaded manually.

    Example::

        ds = KITTIEigenDataset(
            root="/data/kitti",
            split="test",
        )
        sample = ds[0]
        # sample["depth_map"] values are in metres; most are 0 (invalid)
        # sample["valid_mask"] is True only where LiDAR returns exist
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        filenames: Optional[str | Path] = None,
        transform=None,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        download_split: bool = True,
    ) -> None:
        super().__init__(transform=transform, min_depth=min_depth, max_depth=max_depth)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.root = Path(root)
        self.split = split

        if not self.root.exists():
            raise FileNotFoundError(
                f"KITTI root not found: {self.root}\n"
                "KITTI requires manual registration and download.\n"
                "See the module docstring or docs/datasets.md for instructions."
            )

        if filenames is not None:
            self._filenames_path = Path(filenames)
        else:
            self._filenames_path = self.root / f"eigen_{split}_files_with_gt.txt"
            if not self._filenames_path.exists() and download_split:
                self._download_split_file()

        self._samples = self._parse_filenames()

    # ------------------------------------------------------------------
    # Download split file
    # ------------------------------------------------------------------

    def _download_split_file(self) -> None:
        from .hub import download_file

        url = _SPLIT_URLS[self.split]
        logger.info("Downloading KITTI %s split file from %s", self.split, url)
        download_file(url, self._filenames_path, desc=self._filenames_path.name)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_filenames(self) -> List[dict]:
        """Parse the split file into a list of sample dicts."""
        if not self._filenames_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {self._filenames_path}\n"
                "Pass download_split=True or provide filenames= manually."
            )

        samples = []
        with open(self._filenames_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Format: "date/sequence  frame_id  side  [gt_path]"
                seq_dir = parts[0]       # e.g. 2011_09_26/2011_09_26_drive_0001_sync
                frame_id = parts[1]      # e.g. 0000000005
                side = parts[2] if len(parts) > 2 else "l"
                gt_rel = parts[3] if len(parts) > 3 else None

                cam = "image_02" if side == "l" else "image_03"
                rgb_path = self.root / seq_dir / cam / "data" / f"{frame_id}.png"

                if gt_rel and gt_rel != "None":
                    gt_path = self.root / gt_rel
                else:
                    # Construct default GT path from data_depth_annotated layout
                    seq_name = Path(seq_dir).name
                    gt_path = (
                        self.root
                        / "data_depth_annotated"
                        / ("train" if self.split != "test" else "val")
                        / seq_name
                        / "proj_depth"
                        / "groundtruth"
                        / cam
                        / f"{frame_id}.png"
                    )

                samples.append({"rgb": rgb_path, "depth": gt_path, "frame": frame_id})

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        sample = self._samples[index]

        image = np.array(Image.open(sample["rgb"]).convert("RGB"), dtype=np.uint8)

        depth_path: Path = sample["depth"]
        if depth_path.exists():
            # 16-bit PNG: value / 256.0 = metres (KITTI convention)
            depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
            depth = depth_raw / 256.0
        else:
            # No GT (common for some Eigen test frames)
            H, W = image.shape[:2]
            depth = np.zeros((H, W), dtype=np.float32)

        return image, depth

    def __repr__(self) -> str:
        return (
            f"KITTIEigenDataset("
            f"split={self.split!r}, "
            f"n={len(self)}, "
            f"root={self.root})"
        )
