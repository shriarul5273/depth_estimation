"""DIODE — Dense Indoor and Outdoor DEpth dataset.

Reference:
    Vasiljevic et al., "DIODE: A Dense Indoor and Outdoor DEpth Dataset"
    arXiv 1908.00463, 2019.  https://diode-dataset.org

Files downloaded automatically on first use:
    val   set (~2.6 GB):  https://diode-dataset.s3.amazonaws.com/val.tar.gz
    train set (~81 GB):   https://diode-dataset.s3.amazonaws.com/train.tar.gz

File layout after extraction::

    val/
      indoors/
        scene_00001/
          scan_00001/
            00001.png            ← RGB (768 × 1024 × 3, uint8)
            00001_depth.npy      ← depth in metres (768 × 1024, float32)
            00001_depth_mask.npy ← valid mask (768 × 1024, uint8, 0/1)
      outdoors/
        ...
    train/
      indoors/  ...
      outdoors/ ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base_dataset import BaseDepthDataset
from .hub import download_file, extract_archive, get_cache_dir

logger = logging.getLogger(__name__)

_URLS = {
    "train": "https://diode-dataset.s3.amazonaws.com/train.tar.gz",
    "val":   "https://diode-dataset.s3.amazonaws.com/val.tar.gz",
}
# "test" uses the val archive (no public GT for the test set)
_ARCHIVE_MAP = {"train": "train", "val": "val", "test": "val"}

_SCENE_TYPES = ("indoors", "outdoors")


class DIODEDataset(BaseDepthDataset):
    """DIODE depth dataset — indoors and/or outdoors scenes.

    Depth values are in **metres** (laser scanner, dense GT).

    Args:
        root:        Directory that contains (or will contain) the extracted
                     ``train/`` and ``val/`` subdirectories.
                     Defaults to ``~/.cache/depth_estimation/datasets/diode/``.
        split:       ``"train"``, ``"val"``, or ``"test"``.
                     ``"test"`` loads the val images (no public GT exists for test).
        scene_type:  ``"indoors"``, ``"outdoors"``, or ``"all"`` (default).
        transform:   Paired callable ``(pixel_values, depth_map, valid_mask) → ...``
        min_depth:   Minimum valid depth in metres. Default ``1e-3``.
        max_depth:   Maximum valid depth in metres. Default ``350.0``
                     (outdoor scenes reach ~300 m).
        download:    Auto-download when the split archive is missing.
                     The val set is ~2.6 GB; the train set is ~81 GB.

    Example::

        # Download val set (~2.6 GB) on first run, then reuse cache
        ds = DIODEDataset(split="val", scene_type="indoors")
        sample = ds[0]
        # sample["depth_map"]  → (1, 768, 1024) float32  (metres)
        # sample["valid_mask"] → (1, 768, 1024) bool
    """

    def __init__(
        self,
        root: Optional[str | Path] = None,
        split: str = "val",
        scene_type: str = "all",
        transform=None,
        min_depth: float = 1e-3,
        max_depth: float = 350.0,
        download: bool = True,
    ) -> None:
        super().__init__(transform=transform, min_depth=min_depth, max_depth=max_depth)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        if scene_type not in ("indoors", "outdoors", "all"):
            raise ValueError(
                f"scene_type must be 'indoors', 'outdoors', or 'all', got {scene_type!r}"
            )

        self.split = split
        self.scene_type = scene_type
        self.root = Path(root) if root else get_cache_dir("diode")

        if download:
            self._download()

        self._samples = self._collect_samples()

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self) -> None:
        archive_key = _ARCHIVE_MAP[self.split]
        split_dir = self.root / archive_key

        if split_dir.exists():
            logger.info("DIODE %s: already extracted at %s", archive_key, split_dir)
            return

        url = _URLS[archive_key]
        archive_name = f"{archive_key}.tar.gz"
        archive_path = self.root / archive_name

        size_hint = "~2.6 GB" if archive_key == "val" else "~81 GB"
        logger.info("DIODE: downloading %s (%s) …", archive_name, size_hint)
        download_file(url, archive_path, desc=archive_name)

        logger.info("DIODE: extracting %s …", archive_name)
        extract_archive(archive_path, self.root)
        # Remove archive to save space (optional — keep if re-extraction may be needed)
        # archive_path.unlink()

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def _collect_samples(self) -> List[dict]:
        archive_key = _ARCHIVE_MAP[self.split]
        split_dir = self.root / archive_key

        if not split_dir.exists():
            raise FileNotFoundError(
                f"DIODE {archive_key} split not found at {split_dir}.\n"
                "Pass download=True or extract the archive manually."
            )

        scene_types = (
            _SCENE_TYPES if self.scene_type == "all" else (self.scene_type,)
        )

        samples: List[dict] = []
        for stype in scene_types:
            stype_dir = split_dir / stype
            if not stype_dir.exists():
                continue
            for scene_dir in sorted(stype_dir.iterdir()):
                for scan_dir in sorted(scene_dir.iterdir()):
                    for rgb_path in sorted(scan_dir.glob("*.png")):
                        stem = rgb_path.stem
                        depth_path = scan_dir / f"{stem}_depth.npy"
                        mask_path = scan_dir / f"{stem}_depth_mask.npy"
                        if depth_path.exists() and mask_path.exists():
                            samples.append(
                                {
                                    "rgb": rgb_path,
                                    "depth": depth_path,
                                    "mask": mask_path,
                                }
                            )

        if not samples:
            raise RuntimeError(
                f"No samples found in {split_dir} for scene_type={self.scene_type!r}."
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
        depth = np.load(s["depth"]).astype(np.float32).squeeze()
        mask = np.load(s["mask"]).astype(bool).squeeze()

        # Zero out invalid pixels so valid_mask generation in BaseDepthDataset works
        depth[~mask] = 0.0

        return image, depth

    def __repr__(self) -> str:
        return (
            f"DIODEDataset("
            f"split={self.split!r}, "
            f"scene_type={self.scene_type!r}, "
            f"n={len(self)}, "
            f"root={self.root})"
        )
