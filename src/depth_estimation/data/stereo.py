"""Stereo datasets and augmentations, for stereo models such as S2M2.

Stereo samples are rectified ``(left, right)`` pairs with ground-truth
**disparity** for the left view (in pixels), rather than an RGB image with
depth. Each dataset's ``__getitem__`` returns::

    pixel_values  – (6, H, W) float32 in [0, 1]: left RGB then right RGB
    disparity     – (1, H, W) float32, 0 where invalid
    valid_mask    – (1, H, W) bool

Datasets
--------
=================  ==============  ===========================================
Name               Auto-download   Notes
=================  ==============  ===========================================
middlebury_2014    Yes (Q: 44 MB)  MiddEval3 training scenes (the 15 with
                                   public GT), at Q / H / F resolution.
kitti_2015         No              KITTI 2015 stereo, 200 training pairs.
stereo_folder      N/A             left_dir / right_dir / disparity_dir.
=================  ==============  ===========================================

Augmentations follow the paired-transform protocol of
:mod:`depth_estimation.data.transforms` (``(pixel_values, disparity,
valid_mask) -> same``) and keep the two views consistent: spatial ops are
identical for both views and rescale disparity with the image width. There is
no horizontal flip, which would swap the roles of the two cameras.
"""

from __future__ import annotations

import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

from .transforms import Compose

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_DISP_EXTS = (".pfm", ".png", ".npy", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_pfm(path: Union[str, Path]) -> np.ndarray:
    """Read a PFM file (Middlebury / SceneFlow disparity) as ``(H, W)`` float32."""
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        channels = 3 if header == "PF" else 1
        dims = f.readline().decode("latin-1")
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1")
        match = re.match(r"^\s*(\d+)\s+(\d+)\s*$", dims)
        if not match:
            raise ValueError(f"Malformed PFM header in {path}")
        width, height = int(match.group(1)), int(match.group(2))
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=endian + "f4")
    shape = (height, width, channels) if channels == 3 else (height, width)
    data = np.flipud(data[: int(np.prod(shape))].reshape(shape))
    if channels == 3:
        data = data[..., 0]
    return np.ascontiguousarray(data, dtype=np.float32)


def read_disparity(
    path: Union[str, Path], scale: float = 256.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a disparity map and its validity mask.

    - ``.pfm``: float disparity; ``inf``/``nan`` are invalid.
    - ``.png`` / ``.tif``: 16-bit integers divided by ``scale`` (KITTI uses
      256); 0 is invalid.
    - ``.npy``: float disparity; non-finite or <= 0 is invalid.

    Returns:
        ``(disparity, valid)``: ``(H, W)`` float32 (0 where invalid) and bool.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pfm":
        disp = read_pfm(path)
        valid = np.isfinite(disp)
    elif ext == ".npy":
        disp = np.load(path).squeeze().astype(np.float32)
        valid = np.isfinite(disp) & (disp > 0)
    elif ext in (".png", ".tif", ".tiff"):
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise FileNotFoundError(f"Could not read disparity: {path}")
        valid = raw > 0
        disp = raw.astype(np.float32) / scale
    else:
        raise ValueError(f"Unsupported disparity format {ext!r}: {path}")
    disp = np.where(valid, disp, 0.0).astype(np.float32)
    return disp, valid


def _read_rgb(path: Union[str, Path]) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseStereoDataset(ABC):
    """Abstract base for stereo datasets.

    Subclasses implement ``__len__`` and ``_load_sample(index)`` returning
    ``(left, right, disparity, valid)``: two ``(H, W, 3)`` uint8 RGB arrays,
    ``(H, W)`` float32 left-view disparity in pixels and an ``(H, W)`` bool
    mask.

    Args:
        transform: Paired callable ``(pixel_values, disparity, valid_mask)``
            applied after tensor conversion (see :func:`get_stereo_train_transforms`).
        max_disparity: Pixels with ground truth above this are masked out.
    """

    #: Marks the dataset as stereo for evaluate() / DepthTrainer.
    is_stereo = True

    def __init__(self, transform=None, max_disparity: Optional[float] = None) -> None:
        self.transform = transform
        self.max_disparity = max_disparity

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _load_sample(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _masked_sample(self, index: int):
        left, right, disp, valid = self._load_sample(index)
        if left.shape != right.shape:
            raise ValueError(
                f"Sample {index}: left {left.shape} and right {right.shape} differ in size."
            )
        if self.max_disparity is not None:
            valid = valid & (disp <= self.max_disparity)
        return left, right, disp, valid

    def __getitem__(self, index: int) -> Dict[str, Any]:
        left, right, disp, valid = self._masked_sample(index)
        pixel_values = (
            torch.from_numpy(np.concatenate([left, right], axis=2)).permute(2, 0, 1).float()
            / 255.0
        )
        disparity = torch.from_numpy(np.where(valid, disp, 0.0)).float().unsqueeze(0)
        valid_mask = torch.from_numpy(valid).unsqueeze(0)
        if self.transform is not None:
            pixel_values, disparity, valid_mask = self.transform(
                pixel_values, disparity, valid_mask
            )
        return {
            "pixel_values": pixel_values,
            "disparity": disparity,
            "valid_mask": valid_mask,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"


# ---------------------------------------------------------------------------
# Generic folder
# ---------------------------------------------------------------------------


class StereoFolderDataset(BaseStereoDataset):
    """Stereo pairs from parallel directories, matched by file stem.

    ::

        left_dir/000.png    right_dir/000.png    disparity_dir/000.pfm|png|npy

    Args:
        left_dir: Left (reference) images.
        right_dir: Right images, same stems as ``left_dir``.
        disparity_dir: Left-view disparity (see :func:`read_disparity`). If
            ``None``, disparity is all zero and invalid (inference only).
        disparity_scale: Divisor for 16-bit PNG/TIFF disparity. Default 256.
        transform: Paired stereo transform.
        max_disparity: Mask ground truth above this.
    """

    def __init__(
        self,
        left_dir: Union[str, Path],
        right_dir: Union[str, Path],
        disparity_dir: Optional[Union[str, Path]] = None,
        disparity_scale: float = 256.0,
        transform=None,
        max_disparity: Optional[float] = None,
    ) -> None:
        super().__init__(transform=transform, max_disparity=max_disparity)
        self.left_dir = Path(left_dir)
        self.right_dir = Path(right_dir)
        self.disparity_dir = Path(disparity_dir) if disparity_dir else None
        self.disparity_scale = disparity_scale
        for d in (self.left_dir, self.right_dir, self.disparity_dir):
            if d is not None and not d.is_dir():
                raise FileNotFoundError(f"Directory not found: {d}")
        self._samples = self._collect()

    def _collect(self) -> List[dict]:
        rights = {
            p.stem: p for p in self.right_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
        }
        disps: Dict[str, Path] = {}
        if self.disparity_dir is not None:
            for p in self.disparity_dir.iterdir():
                if p.suffix.lower() in _DISP_EXTS:
                    disps.setdefault(p.stem, p)
        samples = []
        for left in sorted(self.left_dir.iterdir()):
            if left.suffix.lower() not in _IMAGE_EXTS:
                continue
            if left.stem not in rights:
                logger.warning("No right image for %s — skipping.", left.name)
                continue
            if self.disparity_dir is not None and left.stem not in disps:
                logger.warning("No disparity for %s — skipping.", left.name)
                continue
            samples.append(
                {"left": left, "right": rights[left.stem], "disp": disps.get(left.stem)}
            )
        if not samples:
            raise RuntimeError(f"No stereo pairs found in {self.left_dir} / {self.right_dir}")
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_sample(self, index: int):
        s = self._samples[index]
        left, right = _read_rgb(s["left"]), _read_rgb(s["right"])
        if s["disp"] is None:
            disp = np.zeros(left.shape[:2], dtype=np.float32)
            valid = np.zeros(left.shape[:2], dtype=bool)
        else:
            disp, valid = read_disparity(s["disp"], self.disparity_scale)
        return left, right, disp, valid


# ---------------------------------------------------------------------------
# Middlebury 2014 (MiddEval3)
# ---------------------------------------------------------------------------

_MIDDLEBURY_URL = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-{kind}-{res}.zip"


class Middlebury2014Dataset(BaseStereoDataset):
    """Middlebury 2014 stereo (MiddEval3 training set).

    The 15 MiddEval3 training scenes are the only ones with public ground
    truth, so every ``split`` returns them. Disparity is in pixels of the
    chosen resolution.

    Args:
        root: Download / extraction directory. Defaults to
            ``~/.cache/depth_estimation/datasets/middlebury_2014``.
        split: Accepted for API compatibility; see above.
        resolution: ``"Q"`` (quarter, ~750x500, 44 MB), ``"H"`` (half,
            163 MB) or ``"F"`` (full, ~3000x2000).
        download: Download and extract the data + GT zips if missing.
        occlusion: ``"all"`` evaluates every pixel with ground truth;
            ``"noc"`` only non-occluded pixels (``mask0nocc.png``).
        transform: Paired stereo transform.
        max_disparity: Mask ground truth above this.

    ``calibration(index)`` returns ``focal_length_px``, ``baseline`` (metres)
    and ``doffs`` for metric depth via ``S2M2Model.infer``.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        split: str = "train",
        resolution: str = "Q",
        download: bool = True,
        occlusion: str = "all",
        transform=None,
        max_disparity: Optional[float] = None,
    ) -> None:
        super().__init__(transform=transform, max_disparity=max_disparity)
        if resolution not in ("Q", "H", "F"):
            raise ValueError(f"resolution must be 'Q', 'H' or 'F', got {resolution!r}")
        if occlusion not in ("all", "noc"):
            raise ValueError(f"occlusion must be 'all' or 'noc', got {occlusion!r}")
        from .hub import get_cache_dir

        self.root = Path(root) if root else get_cache_dir("middlebury_2014")
        self.split = split
        self.resolution = resolution
        self.occlusion = occlusion
        self._scene_root = self.root / "MiddEval3" / f"training{resolution}"
        if download and not self._has_gt():
            self._download()
        if not self._has_gt():
            raise FileNotFoundError(
                f"Middlebury 2014 ({resolution}) not found under {self.root}. "
                "Pass download=True, or extract MiddEval3-data-"
                f"{resolution}.zip and MiddEval3-GT0-{resolution}.zip there."
            )
        self._scenes = sorted(
            p for p in self._scene_root.iterdir() if (p / "disp0GT.pfm").exists()
        )

    def _has_gt(self) -> bool:
        return self._scene_root.is_dir() and any(self._scene_root.glob("*/disp0GT.pfm"))

    def _download(self) -> None:
        from .hub import download_file, extract_archive

        self.root.mkdir(parents=True, exist_ok=True)
        for kind in ("data", "GT0"):
            name = f"MiddEval3-{kind}-{self.resolution}.zip"
            archive = self.root / name
            if not archive.exists():
                logger.info("Middlebury 2014: downloading %s …", name)
                download_file(
                    _MIDDLEBURY_URL.format(kind=kind, res=self.resolution), archive, desc=name
                )
            extract_archive(archive, self.root)

    def __len__(self) -> int:
        return len(self._scenes)

    @property
    def scene_names(self) -> List[str]:
        return [p.name for p in self._scenes]

    def _load_sample(self, index: int):
        import cv2

        scene = self._scenes[index]
        left, right = _read_rgb(scene / "im0.png"), _read_rgb(scene / "im1.png")
        disp, valid = read_disparity(scene / "disp0GT.pfm")
        if self.occlusion == "noc":
            nocc = cv2.imread(str(scene / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
            valid = valid & (nocc == 255)
        return left, right, disp, valid

    def calibration(self, index: int) -> Dict[str, float]:
        """Rectified focal length (px), baseline (metres) and doffs for a scene."""
        text = (self._scenes[index] / "calib.txt").read_text()
        values = dict(line.split("=", 1) for line in text.split() if "=" in line)
        focal = float(values["cam0"].strip("[").split()[0])
        return {
            "focal_length_px": focal,
            "baseline": float(values["baseline"]) / 1000.0,
            "doffs": float(values["doffs"]),
        }

    def __repr__(self) -> str:
        return (
            f"Middlebury2014Dataset(n={len(self)}, resolution={self.resolution!r}, "
            f"occlusion={self.occlusion!r})"
        )


# ---------------------------------------------------------------------------
# KITTI 2015
# ---------------------------------------------------------------------------


class KITTI2015Dataset(BaseStereoDataset):
    """KITTI 2015 stereo benchmark (200 training pairs with ground truth).

    KITTI requires registration, so the data cannot be downloaded
    automatically. Get ``data_scene_flow.zip`` from
    https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
    and extract it so that ``root/training/image_2``, ``image_3`` and
    ``disp_occ_0`` exist. Only the training split has public ground truth, so
    every ``split`` returns the 200 training pairs.

    Args:
        root: Directory containing ``training/``.
        split: Accepted for API compatibility; see above.
        occlusion: ``"all"`` (``disp_occ_0``, the official default) or
            ``"noc"`` (``disp_noc_0``, non-occluded pixels only).
        transform: Paired stereo transform.
        max_disparity: Mask ground truth above this.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        occlusion: str = "all",
        transform=None,
        max_disparity: Optional[float] = None,
    ) -> None:
        super().__init__(transform=transform, max_disparity=max_disparity)
        if occlusion not in ("all", "noc"):
            raise ValueError(f"occlusion must be 'all' or 'noc', got {occlusion!r}")
        self.root = Path(root)
        self.split = split
        self.occlusion = occlusion
        base = self.root / "training"
        disp_dir = base / ("disp_occ_0" if occlusion == "all" else "disp_noc_0")
        for d in (base / "image_2", base / "image_3", disp_dir):
            if not d.is_dir():
                raise FileNotFoundError(
                    f"{d} not found. Extract KITTI 2015 data_scene_flow.zip so that "
                    f"{self.root}/training/image_2, image_3 and disp_occ_0 exist "
                    "(registration required, see the class docstring)."
                )
        self._samples = [
            (base / "image_2" / p.name, base / "image_3" / p.name, p)
            for p in sorted(disp_dir.glob("*_10.png"))
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def _load_sample(self, index: int):
        left_path, right_path, disp_path = self._samples[index]
        disp, valid = read_disparity(disp_path, scale=256.0)
        return _read_rgb(left_path), _read_rgb(right_path), disp, valid


# ---------------------------------------------------------------------------
# Stereo augmentations
# ---------------------------------------------------------------------------


def _split_views(pixel_values: torch.Tensor) -> torch.Tensor:
    """``(6, H, W)`` -> ``(2, 3, H, W)`` (left, right)."""
    return pixel_values.reshape(2, 3, *pixel_values.shape[-2:])


class StereoRandomScale:
    """Rescale both views by one random factor; disparity scales with width.

    Args:
        scale_range: ``(min, max)`` scale factor. Default ``(0.8, 1.2)``.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, pixel_values, disparity, valid_mask):
        s = random.uniform(*self.scale_range)
        h, w = pixel_values.shape[-2:]
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        if (nh, nw) == (h, w):
            return pixel_values, disparity, valid_mask
        pixel_values = F.interpolate(
            pixel_values[None], size=(nh, nw), mode="bilinear", align_corners=False
        )[0]
        disparity = F.interpolate(disparity[None], size=(nh, nw), mode="nearest")[0] * (nw / w)
        valid_mask = F.interpolate(valid_mask[None].float(), size=(nh, nw), mode="nearest")[0] > 0.5
        return pixel_values, disparity, valid_mask

    def __repr__(self) -> str:
        return f"StereoRandomScale(scale_range={self.scale_range})"


class StereoRandomCrop:
    """Crop both views and the disparity at the same random location.

    Inputs smaller than ``size`` are zero-padded (padding is marked invalid).

    Args:
        size: ``(height, width)`` or an int for a square crop.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def _pad(self, pixel_values, disparity, valid_mask):
        ch, cw = self.size
        h, w = pixel_values.shape[-2:]
        ph, pw = max(0, ch - h), max(0, cw - w)
        if ph or pw:
            pad = (0, pw, 0, ph)
            pixel_values = F.pad(pixel_values, pad)
            disparity = F.pad(disparity, pad)
            valid_mask = F.pad(valid_mask, pad)
        return pixel_values, disparity, valid_mask

    def _offsets(self, h: int, w: int) -> Tuple[int, int]:
        ch, cw = self.size
        return random.randint(0, h - ch), random.randint(0, w - cw)

    def __call__(self, pixel_values, disparity, valid_mask):
        pixel_values, disparity, valid_mask = self._pad(pixel_values, disparity, valid_mask)
        ch, cw = self.size
        top, left = self._offsets(*pixel_values.shape[-2:])
        sl = (..., slice(top, top + ch), slice(left, left + cw))
        return pixel_values[sl], disparity[sl], valid_mask[sl]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class StereoCenterCrop(StereoRandomCrop):
    """Deterministic centre crop of both views and the disparity."""

    def _offsets(self, h: int, w: int) -> Tuple[int, int]:
        ch, cw = self.size
        return (h - ch) // 2, (w - cw) // 2


class StereoColorJitter:
    """Photometric jitter of both views.

    The same jitter is applied to both views, except with probability
    ``asymmetric_prob`` each view gets its own (real stereo cameras differ
    slightly in exposure and colour).
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.05,
        asymmetric_prob: float = 0.2,
    ) -> None:
        self._jitter = ColorJitter(brightness, contrast, saturation, hue)
        self.asymmetric_prob = asymmetric_prob

    def __call__(self, pixel_values, disparity, valid_mask):
        views = _split_views(pixel_values)
        if random.random() < self.asymmetric_prob:
            views = torch.stack([self._jitter(v) for v in views])
        else:
            # A batched call samples one set of factors for both views.
            views = self._jitter(views)
        return views.reshape(6, *views.shape[-2:]), disparity, valid_mask

    def __repr__(self) -> str:
        return f"StereoColorJitter({self._jitter}, asymmetric_prob={self.asymmetric_prob})"


class StereoNormalize:
    """Normalise each view with the model's mean/std (S2M2: 0.5 / 0.5)."""

    def __init__(
        self,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, pixel_values, disparity, valid_mask):
        pixel_values = TF.normalize(pixel_values, mean=self.mean * 2, std=self.std * 2)
        return pixel_values, disparity, valid_mask

    def __repr__(self) -> str:
        return f"StereoNormalize(mean={self.mean}, std={self.std})"


def get_stereo_train_transforms(
    crop_size: Union[int, Tuple[int, int]] = (320, 640),
    scale_range: Tuple[float, float] = (0.8, 1.2),
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> Compose:
    """Training pipeline for stereo fine-tuning.

    Random scale → random crop → colour jitter → normalise. Use crop sides
    that are multiples of 32 for S2M2. ``mean``/``std`` default to S2M2's
    ``[-1, 1]`` normalisation.
    """
    return Compose(
        [
            StereoRandomScale(scale_range),
            StereoRandomCrop(crop_size),
            StereoColorJitter(),
            StereoNormalize(mean, std),
        ]
    )


def get_stereo_val_transforms(
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> Compose:
    """Validation pipeline: optional centre crop, then normalise.

    Without ``crop_size`` samples keep their native size, so use
    ``batch_size=1`` when their sizes differ (e.g. Middlebury).
    """
    steps = [StereoCenterCrop(crop_size)] if crop_size is not None else []
    return Compose(steps + [StereoNormalize(mean, std)])
