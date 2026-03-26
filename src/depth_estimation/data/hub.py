"""Download utilities — fetch files and extract archives to a local cache."""

from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path.home() / ".cache" / "depth_estimation" / "datasets"


def get_cache_dir(name: str = "") -> Path:
    """Return (and create) the cache directory for *name*."""
    base = Path(os.environ.get("DEPTH_CACHE_DIR", _DEFAULT_CACHE))
    d = base / name if name else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_file(url: str, dest: Path, desc: Optional[str] = None) -> Path:
    """Download *url* to *dest*, skipping if already present.

    Shows a tqdm progress bar when tqdm is installed; falls back to a
    plain print otherwise.
    """
    dest = Path(dest)
    if dest.exists():
        logger.debug("Already downloaded: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    label = desc or dest.name

    try:
        from tqdm import tqdm

        class _Hook:
            def __init__(self) -> None:
                self.pbar: Optional[tqdm] = None

            def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
                if self.pbar is None:
                    self.pbar = tqdm(
                        total=total_size, unit="B", unit_scale=True, desc=label
                    )
                downloaded = block_num * block_size
                self.pbar.update(min(downloaded, total_size) - self.pbar.n)
                if downloaded >= total_size and self.pbar:
                    self.pbar.close()

        urlretrieve(url, tmp, reporthook=_Hook())
    except ImportError:
        print(f"Downloading {label} ...")
        urlretrieve(url, tmp)

    tmp.rename(dest)
    logger.info("Saved to %s", dest)
    return dest


def extract_archive(src: Path, dest: Path) -> Path:
    """Extract a .tar.gz / .tgz / .tar.bz2 / .tar / .zip archive to *dest*."""
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = src.name.lower()

    logger.info("Extracting %s → %s", src.name, dest)
    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(src, "r:*") as tf:
            tf.extractall(dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {src}")

    return dest
