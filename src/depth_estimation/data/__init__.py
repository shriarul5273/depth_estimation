"""depth_estimation.data — dataset loading and download utilities.

Quick start
-----------
Use :func:`load_dataset` to load any supported dataset by name::

    from depth_estimation.data import load_dataset

    # NYU Depth V2 — auto-download ~2.8 GB on first use
    ds = load_dataset("nyu_depth_v2", split="test")

    # DIODE val set — auto-download ~2.6 GB on first use
    ds = load_dataset("diode", split="val", scene_type="indoors")

    # KITTI — path required (manual download; see KITTIEigenDataset docs)
    ds = load_dataset("kitti_eigen", split="test", root="/data/kitti")

    # Generic folder of RGB + depth pairs
    ds = load_dataset("folder", image_dir="rgb/", depth_dir="depth/")

Each dataset returns dicts with keys:
    pixel_values  – ``(3, H, W)`` float32 tensor, normalised to ``[0, 1]``
    depth_map     – ``(1, H, W)`` float32 tensor (metres or relative)
    valid_mask    – ``(1, H, W)`` bool tensor

Supported dataset names
-----------------------
=============  ==============  ==============================================
Name           Auto-download   Notes
=============  ==============  ==============================================
nyu_depth_v2   Yes (~2.8 GB)   Indoor. Eigen split: 795 train / 654 test.
diode          Yes (~2.6 GB)   Indoor + outdoor. Val set auto-downloads.
                               Train set is ~81 GB.
kitti_eigen    No              Outdoor. Requires registration. Split file
                               auto-downloaded from BTS repo.
folder         N/A             Generic paired RGB + depth directories.
=============  ==============  ==============================================
"""

from __future__ import annotations

from typing import Any, Optional

from .base_dataset import BaseDepthDataset
from .folder import FolderDataset
from .nyu_depth_v2 import NYUDepthV2Dataset
from .kitti import KITTIEigenDataset
from .diode import DIODEDataset
from . import transforms
from .transforms import get_train_transforms, get_val_transforms, Compose, PairedResize

__all__ = [
    "load_dataset",
    "BaseDepthDataset",
    "NYUDepthV2Dataset",
    "KITTIEigenDataset",
    "DIODEDataset",
    "FolderDataset",
    "transforms",
    "get_train_transforms",
    "get_val_transforms",
    "Compose",
    "PairedResize",
]

_REGISTRY = {
    "nyu_depth_v2": NYUDepthV2Dataset,
    "kitti_eigen":  KITTIEigenDataset,
    "diode":        DIODEDataset,
    "folder":       FolderDataset,
}


def load_dataset(
    name: str,
    split: str = "train",
    root: Optional[str] = None,
    download: bool = True,
    transform=None,
    **kwargs: Any,
) -> BaseDepthDataset:
    """Load a depth dataset by name.

    Args:
        name:      Dataset name — one of ``"nyu_depth_v2"``, ``"kitti_eigen"``,
                   ``"diode"``, ``"folder"``.
        split:     ``"train"``, ``"val"``, or ``"test"``.  Not all datasets
                   expose every split.
        root:      Local root directory.  Defaults to
                   ``~/.cache/depth_estimation/datasets/<name>/``.
        download:  Whether to auto-download (where supported).
        transform: Paired callable applied after tensor conversion.
        **kwargs:  Extra keyword arguments forwarded to the dataset class
                   (e.g. ``scene_type="indoors"`` for DIODE,
                   ``image_dir=`` for ``folder``).

    Returns:
        A :class:`BaseDepthDataset` subclass instance.

    Example::

        from depth_estimation.data import load_dataset

        ds = load_dataset("nyu_depth_v2", split="test")
        print(ds)          # NYUDepthV2Dataset(split='test', n=654, ...)
        sample = ds[0]
        print(sample["depth_map"].shape)  # torch.Size([1, 480, 640])
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )

    cls = _REGISTRY[name]

    # Build kwargs for each dataset class
    init_kwargs: dict[str, Any] = {"transform": transform}

    if name == "folder":
        # FolderDataset uses image_dir / depth_dir, not root / split / download
        init_kwargs.update(kwargs)
    else:
        init_kwargs["split"] = split
        if root is not None:
            init_kwargs["root"] = root
        if name != "kitti_eigen":
            init_kwargs["download"] = download
        init_kwargs.update(kwargs)

    return cls(**init_kwargs)
