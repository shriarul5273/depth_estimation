"""Tests for the depth_estimation.data module.

All tests use synthetic in-memory data and tmp_path fixtures.
No network calls are made (auto-download tests are skipped in CI).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from depth_estimation.data import (
    BaseDepthDataset,
    FolderDataset,
    load_dataset,
)
from depth_estimation.data.hub import get_cache_dir


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_rgb(path: Path, h: int = 32, w: int = 32) -> np.ndarray:
    """Save a random uint8 RGB image and return the array."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return arr


def _write_depth_npy(path: Path, h: int = 32, w: int = 32) -> np.ndarray:
    """Save a float32 depth array as .npy and return it."""
    arr = np.random.uniform(0.5, 5.0, (h, w)).astype(np.float32)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# BaseDepthDataset — contract tests via a minimal concrete subclass
# ---------------------------------------------------------------------------

class _SyntheticDataset(BaseDepthDataset):
    """Minimal concrete dataset: returns fixed synthetic data."""

    def __init__(self, n: int = 4, h: int = 32, w: int = 32, **kwargs):
        super().__init__(**kwargs)
        self._n = n
        self._h = h
        self._w = w

    def __len__(self) -> int:
        return self._n

    def _load_sample(self, index: int):
        rgb = np.random.randint(0, 255, (self._h, self._w, 3), dtype=np.uint8)
        depth = np.full((self._h, self._w), fill_value=2.0, dtype=np.float32)
        return rgb, depth


class TestBaseDepthDataset:
    def test_len(self):
        ds = _SyntheticDataset(n=6)
        assert len(ds) == 6

    def test_getitem_keys(self):
        ds = _SyntheticDataset()
        sample = ds[0]
        assert set(sample.keys()) == {"pixel_values", "depth_map", "valid_mask"}

    def test_pixel_values_shape(self):
        ds = _SyntheticDataset(h=32, w=48)
        pv = ds[0]["pixel_values"]
        assert pv.shape == (3, 32, 48)

    def test_pixel_values_range(self):
        ds = _SyntheticDataset()
        pv = ds[0]["pixel_values"]
        assert pv.dtype == torch.float32
        assert pv.min() >= 0.0
        assert pv.max() <= 1.0

    def test_depth_map_shape(self):
        ds = _SyntheticDataset(h=32, w=48)
        dm = ds[0]["depth_map"]
        assert dm.shape == (1, 32, 48)
        assert dm.dtype == torch.float32

    def test_valid_mask_shape_and_dtype(self):
        ds = _SyntheticDataset(h=32, w=48)
        vm = ds[0]["valid_mask"]
        assert vm.shape == (1, 32, 48)
        assert vm.dtype == torch.bool

    def test_valid_mask_respects_depth_bounds(self):
        # depth is uniformly 2.0 — within default min/max (1e-3, 10.0)
        ds = _SyntheticDataset(min_depth=1e-3, max_depth=10.0)
        vm = ds[0]["valid_mask"]
        assert vm.all()

    def test_valid_mask_excludes_out_of_range(self):
        # max_depth=1.0 → depth of 2.0 is invalid
        ds = _SyntheticDataset(min_depth=1e-3, max_depth=1.0)
        vm = ds[0]["valid_mask"]
        assert not vm.any()

    def test_transform_called(self):
        called = []

        def _t(pv, dm, vm):
            called.append(True)
            return pv, dm, vm

        ds = _SyntheticDataset(transform=_t)
        _ = ds[0]
        assert called

    def test_repr(self):
        ds = _SyntheticDataset(n=3)
        r = repr(ds)
        assert "_SyntheticDataset" in r
        assert "n=3" in r


# ---------------------------------------------------------------------------
# FolderDataset
# ---------------------------------------------------------------------------

class TestFolderDataset:
    def test_basic_load(self, tmp_path):
        img_dir = tmp_path / "rgb"
        dep_dir = tmp_path / "depth"
        img_dir.mkdir()
        dep_dir.mkdir()

        for i in range(3):
            _write_rgb(img_dir / f"img{i:03d}.png")
            _write_depth_npy(dep_dir / f"img{i:03d}.npy")

        ds = FolderDataset(image_dir=img_dir, depth_dir=dep_dir)
        assert len(ds) == 3
        sample = ds[0]
        assert set(sample.keys()) == {"pixel_values", "depth_map", "valid_mask"}

    def test_npy_depth_values(self, tmp_path):
        img_dir = tmp_path / "rgb"
        dep_dir = tmp_path / "depth"
        img_dir.mkdir()
        dep_dir.mkdir()

        _write_rgb(img_dir / "a.png", h=16, w=16)
        arr = np.full((16, 16), 3.5, dtype=np.float32)
        np.save(dep_dir / "a.npy", arr)

        ds = FolderDataset(image_dir=img_dir, depth_dir=dep_dir, max_depth=10.0)
        sample = ds[0]
        # All pixels should be valid (depth 3.5 within [1e-3, 10])
        assert sample["valid_mask"].all()

    def test_png_depth_scale(self, tmp_path):
        img_dir = tmp_path / "rgb"
        dep_dir = tmp_path / "depth"
        img_dir.mkdir()
        dep_dir.mkdir()

        _write_rgb(img_dir / "a.png", h=8, w=8)
        # 16-bit PNG depth: value 1024 → 1024/256.0 = 4.0 m
        depth_uint16 = np.full((8, 8), 1024, dtype=np.uint16)
        Image.fromarray(depth_uint16).save(dep_dir / "a.png")

        ds = FolderDataset(image_dir=img_dir, depth_dir=dep_dir, depth_scale=256.0, max_depth=10.0)
        sample = ds[0]
        depth_vals = sample["depth_map"][0]
        assert torch.allclose(depth_vals, torch.full_like(depth_vals, 4.0), atol=1e-3)

    def test_npz_depth(self, tmp_path):
        img_dir = tmp_path / "rgb"
        dep_dir = tmp_path / "depth"
        img_dir.mkdir()
        dep_dir.mkdir()

        _write_rgb(img_dir / "a.png", h=8, w=8)
        arr = np.full((8, 8), 2.0, dtype=np.float32)
        np.savez(dep_dir / "a.npz", depth=arr)

        ds = FolderDataset(image_dir=img_dir, depth_dir=dep_dir, max_depth=10.0)
        sample = ds[0]
        assert sample["valid_mask"].all()

    def test_missing_depth_returns_zero_mask(self, tmp_path):
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        _write_rgb(img_dir / "a.png")

        # No depth_dir → all depth = 0 → valid_mask all False
        ds = FolderDataset(image_dir=img_dir)
        sample = ds[0]
        assert not sample["valid_mask"].any()

    def test_image_dir_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FolderDataset(image_dir=tmp_path / "nonexistent")

    def test_depth_dir_not_found(self, tmp_path):
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        _write_rgb(img_dir / "a.png")
        with pytest.raises(FileNotFoundError):
            FolderDataset(image_dir=img_dir, depth_dir=tmp_path / "no_depth")

    def test_empty_image_dir(self, tmp_path):
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        with pytest.raises(RuntimeError, match="No images found"):
            FolderDataset(image_dir=img_dir)

    def test_repr(self, tmp_path):
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        _write_rgb(img_dir / "a.png")
        ds = FolderDataset(image_dir=img_dir)
        r = repr(ds)
        assert "FolderDataset" in r
        assert "n=1" in r


# ---------------------------------------------------------------------------
# load_dataset dispatcher
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("totally_fake_dataset")

    def test_folder_dispatch(self, tmp_path):
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        _write_rgb(img_dir / "a.png")
        ds = load_dataset("folder", image_dir=str(img_dir))
        assert isinstance(ds, FolderDataset)
        assert len(ds) == 1

    def test_folder_accepts_transform(self, tmp_path):
        img_dir = tmp_path / "rgb"
        img_dir.mkdir()
        _write_rgb(img_dir / "a.png")

        identity = lambda pv, dm, vm: (pv, dm, vm)
        ds = load_dataset("folder", image_dir=str(img_dir), transform=identity)
        assert ds.transform is identity

    @pytest.mark.parametrize("name", ["nyu_depth_v2", "kitti_eigen", "diode"])
    def test_download_datasets_require_network(self, name):
        """These datasets require a network call; just verify load_dataset dispatches
        without error when download=False and no root is given (will raise on access)."""
        # We just check the function accepts the name without ValueError
        # Actual instantiation may raise other errors if root is missing — that's fine.
        try:
            load_dataset(name, download=False)
        except ValueError as e:
            pytest.fail(f"load_dataset raised ValueError for '{name}': {e}")
        except Exception:
            pass  # Other errors (FileNotFoundError, etc.) are expected without data


# ---------------------------------------------------------------------------
# hub.get_cache_dir
# ---------------------------------------------------------------------------

class TestGetCacheDir:
    def test_default_cache_dir(self):
        cache = get_cache_dir("nyu_depth_v2")
        assert "depth_estimation" in str(cache)
        assert "nyu_depth_v2" in str(cache)

    def test_env_var_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEPTH_CACHE_DIR", str(tmp_path))
        cache = get_cache_dir("diode")
        assert str(tmp_path) in str(cache)
        assert "diode" in str(cache)

    def test_returns_path_object(self):
        cache = get_cache_dir("kitti_eigen")
        assert isinstance(cache, Path)
