"""Tests for depth_estimation.data.transforms (paired depth augmentations).

All tests use synthetic tensors — no dataset files or network calls.
"""

from __future__ import annotations

import pytest
import torch

from depth_estimation.data.transforms import (
    Compose,
    PairedCenterCrop,
    PairedColorJitter,
    PairedNormalize,
    PairedRandomCrop,
    PairedRandomHorizontalFlip,
    PairedRandomScale,
    PairedResize,
    get_train_transforms,
    get_val_transforms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(h=64, w=80, seed=0):
    """Return a (pixel_values, depth_map, valid_mask) tuple."""
    g = torch.Generator()
    g.manual_seed(seed)
    pv = torch.rand(3, h, w, generator=g)
    dm = torch.rand(1, h, w, generator=g).clamp(min=0.1)
    vm = (dm > 0.2).bool()
    return pv, dm, vm


def _shapes(pv, dm, vm):
    return pv.shape, dm.shape, vm.shape


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

class TestCompose:
    def test_identity_compose(self):
        pv, dm, vm = _sample()
        t = Compose([])
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert torch.equal(pv, pv2)
        assert torch.equal(dm, dm2)

    def test_chaining_order(self):
        """Transforms should be applied left-to-right."""
        calls = []
        def make_t(n):
            def t(pv, dm, vm):
                calls.append(n)
                return pv, dm, vm
            return t

        pv, dm, vm = _sample()
        Compose([make_t(1), make_t(2), make_t(3)])(pv, dm, vm)
        assert calls == [1, 2, 3]

    def test_repr(self):
        t = Compose([PairedResize(32)])
        assert "Compose" in repr(t)


# ---------------------------------------------------------------------------
# PairedResize
# ---------------------------------------------------------------------------

class TestPairedResize:
    def test_shorter_side_becomes_target(self):
        pv, dm, vm = _sample(h=40, w=80)   # shorter = 40
        t = PairedResize(size=32)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert min(pv2.shape[-2], pv2.shape[-1]) == 32

    def test_aspect_ratio_preserved(self):
        pv, dm, vm = _sample(h=40, w=80)   # 1:2 ratio
        t = PairedResize(size=32)
        pv2, dm2, vm2 = t(pv, dm, vm)
        h2, w2 = pv2.shape[-2], pv2.shape[-1]
        assert abs(w2 / h2 - 2.0) < 0.1

    def test_all_three_same_spatial_size(self):
        pv, dm, vm = _sample(h=64, w=48)
        t = PairedResize(size=32)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape[-2:] == dm2.shape[-2:] == vm2.shape[-2:]

    def test_no_op_when_already_correct(self):
        pv, dm, vm = _sample(h=32, w=48)
        t = PairedResize(size=32)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert torch.equal(pv, pv2)

    def test_depth_dtype_preserved(self):
        pv, dm, vm = _sample(h=64, w=80)
        t = PairedResize(size=32)
        _, dm2, vm2 = t(pv, dm, vm)
        assert dm2.dtype == torch.float32
        assert vm2.dtype == torch.bool


# ---------------------------------------------------------------------------
# PairedRandomCrop
# ---------------------------------------------------------------------------

class TestPairedRandomCrop:
    def test_output_size_int(self):
        pv, dm, vm = _sample(h=64, w=64)
        pv2, dm2, vm2 = PairedRandomCrop(32)(pv, dm, vm)
        assert pv2.shape == (3, 32, 32)
        assert dm2.shape == (1, 32, 32)
        assert vm2.shape == (1, 32, 32)

    def test_output_size_tuple(self):
        pv, dm, vm = _sample(h=64, w=80)
        pv2, dm2, vm2 = PairedRandomCrop((32, 48))(pv, dm, vm)
        assert pv2.shape == (3, 32, 48)

    def test_all_three_same_crop(self):
        """depth and mask should come from the same crop as pixel_values."""
        # Use all-identical tensors so we can verify consistency
        pv = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 64, 64).expand(3, -1, -1).clone()
        dm = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 64, 64)
        vm = torch.ones(1, 64, 64, dtype=torch.bool)

        pv2, dm2, vm2 = PairedRandomCrop(32)(pv, dm, vm)
        # First channel of pv2 should equal dm2 (same spatial crop)
        assert torch.equal(pv2[0].unsqueeze(0), dm2)

    def test_raises_when_image_too_small(self):
        pv, dm, vm = _sample(h=16, w=16)
        with pytest.raises(ValueError, match="smaller than crop size"):
            PairedRandomCrop(32)(pv, dm, vm)


# ---------------------------------------------------------------------------
# PairedCenterCrop
# ---------------------------------------------------------------------------

class TestPairedCenterCrop:
    def test_output_size(self):
        pv, dm, vm = _sample(h=64, w=80)
        pv2, dm2, vm2 = PairedCenterCrop(32)(pv, dm, vm)
        assert pv2.shape == (3, 32, 32)
        assert dm2.shape == (1, 32, 32)

    def test_deterministic(self):
        pv, dm, vm = _sample(h=64, w=80)
        t = PairedCenterCrop(32)
        r1 = t(pv, dm, vm)
        r2 = t(pv, dm, vm)
        assert torch.equal(r1[0], r2[0])

    def test_tuple_size(self):
        pv, dm, vm = _sample(h=64, w=80)
        pv2, dm2, vm2 = PairedCenterCrop((20, 40))(pv, dm, vm)
        assert pv2.shape == (3, 20, 40)


# ---------------------------------------------------------------------------
# PairedRandomHorizontalFlip
# ---------------------------------------------------------------------------

class TestPairedRandomHorizontalFlip:
    def test_p0_never_flips(self):
        pv, dm, vm = _sample()
        t = PairedRandomHorizontalFlip(p=0.0)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert torch.equal(pv, pv2)
        assert torch.equal(dm, dm2)

    def test_p1_always_flips(self):
        pv, dm, vm = _sample()
        t = PairedRandomHorizontalFlip(p=1.0)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert torch.equal(pv2, torch.flip(pv, dims=[-1]))
        assert torch.equal(dm2, torch.flip(dm, dims=[-1]))
        assert torch.equal(vm2, torch.flip(vm, dims=[-1]))

    def test_shape_preserved(self):
        pv, dm, vm = _sample()
        t = PairedRandomHorizontalFlip(p=1.0)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape == pv.shape
        assert dm2.shape == dm.shape
        assert vm2.shape == vm.shape


# ---------------------------------------------------------------------------
# PairedRandomScale
# ---------------------------------------------------------------------------

class TestPairedRandomScale:
    def test_output_shape_within_range(self):
        pv, dm, vm = _sample(h=64, w=64)
        t = PairedRandomScale(scale_range=(1.0, 2.0))
        pv2, dm2, vm2 = t(pv, dm, vm)
        h2, w2 = pv2.shape[-2], pv2.shape[-1]
        assert 64 <= h2 <= 130
        assert 64 <= w2 <= 130

    def test_all_three_same_spatial_size(self):
        pv, dm, vm = _sample(h=64, w=64)
        t = PairedRandomScale(scale_range=(1.0, 2.0))
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape[-2:] == dm2.shape[-2:] == vm2.shape[-2:]

    def test_scale_depth_true(self):
        """With scale_depth=True, constant depth values should be multiplied by the scale."""
        # Constant depth map so nearest-neighbour resize preserves exact values
        pv = torch.rand(3, 32, 32)
        dm = torch.full((1, 32, 32), 3.0)
        vm = torch.ones(1, 32, 32, dtype=torch.bool)
        # scale_range=(2.0,2.0) always draws scale=2.0
        t = PairedRandomScale(scale_range=(2.0, 2.0), scale_depth=True)
        _, dm2, _ = t(pv, dm, vm)
        # After 2× resize, all values should equal 3.0 * 2.0 = 6.0
        assert torch.allclose(dm2, torch.full_like(dm2, 6.0), atol=1e-4)

    def test_scale_depth_false_does_not_scale_values(self):
        """With scale_depth=False, depth values should not be multiplied."""
        pv = torch.rand(3, 32, 32)
        dm = torch.full((1, 32, 32), 3.0)
        vm = torch.ones(1, 32, 32, dtype=torch.bool)
        t = PairedRandomScale(scale_range=(2.0, 2.0), scale_depth=False)
        _, dm2, _ = t(pv, dm, vm)
        # Values unchanged — only spatial size changes
        assert torch.allclose(dm2, torch.full_like(dm2, 3.0), atol=1e-4)

    def test_mask_stays_bool(self):
        pv, dm, vm = _sample()
        t = PairedRandomScale(scale_range=(1.0, 1.5))
        _, _, vm2 = t(pv, dm, vm)
        assert vm2.dtype == torch.bool


# ---------------------------------------------------------------------------
# PairedColorJitter
# ---------------------------------------------------------------------------

class TestPairedColorJitter:
    def test_depth_and_mask_unchanged(self):
        pv, dm, vm = _sample()
        t = PairedColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert torch.equal(dm, dm2)
        assert torch.equal(vm, vm2)

    def test_image_modified(self):
        torch.manual_seed(42)
        pv, dm, vm = _sample()
        t = PairedColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5)
        pv2, _, _ = t(pv, dm, vm)
        # Color jitter should change the image (with high probability for large params)
        assert not torch.equal(pv, pv2)

    def test_shape_preserved(self):
        pv, dm, vm = _sample()
        t = PairedColorJitter()
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape == pv.shape


# ---------------------------------------------------------------------------
# PairedNormalize
# ---------------------------------------------------------------------------

class TestPairedNormalize:
    def test_depth_and_mask_unchanged(self):
        pv, dm, vm = _sample()
        t = PairedNormalize()
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert torch.equal(dm, dm2)
        assert torch.equal(vm, vm2)

    def test_imagenet_normalization(self):
        """After ImageNet normalization, mean should be near 0."""
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        # Create a 1×1 image with exact ImageNet mean — result should be ~0
        pv = torch.tensor(mean).reshape(3, 1, 1)
        dm = torch.ones(1, 1, 1)
        vm = torch.ones(1, 1, 1, dtype=torch.bool)
        t = PairedNormalize(mean=mean, std=std)
        pv2, _, _ = t(pv, dm, vm)
        assert pv2.abs().max() < 1e-5


# ---------------------------------------------------------------------------
# get_train_transforms / get_val_transforms
# ---------------------------------------------------------------------------

class TestTransformPresets:
    @pytest.mark.parametrize("input_size", [224, 384, 518])
    def test_train_output_shape(self, input_size):
        pv, dm, vm = _sample(h=input_size + 32, w=input_size + 64)
        t = get_train_transforms(input_size=input_size)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape == (3, input_size, input_size)
        assert dm2.shape == (1, input_size, input_size)
        assert vm2.shape == (1, input_size, input_size)

    @pytest.mark.parametrize("input_size", [224, 384, 518])
    def test_val_output_shape(self, input_size):
        pv, dm, vm = _sample(h=input_size + 32, w=input_size + 64)
        t = get_val_transforms(input_size=input_size)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape == (3, input_size, input_size)
        assert dm2.shape == (1, input_size, input_size)
        assert vm2.shape == (1, input_size, input_size)

    def test_val_is_deterministic(self):
        pv, dm, vm = _sample(h=600, w=800)
        t = get_val_transforms(input_size=518)
        r1 = t(pv, dm, vm)
        r2 = t(pv, dm, vm)
        assert torch.equal(r1[0], r2[0])
        assert torch.equal(r1[1], r2[1])

    def test_train_handles_small_input(self):
        """Input whose shorter side equals input_size should still work."""
        input_size = 256
        pv, dm, vm = _sample(h=input_size, w=input_size + 100)
        t = get_train_transforms(input_size=input_size)
        pv2, dm2, vm2 = t(pv, dm, vm)
        assert pv2.shape == (3, input_size, input_size)

    def test_mask_dtype_preserved_through_pipeline(self):
        pv, dm, vm = _sample(h=600, w=800)
        for t in [get_train_transforms(518), get_val_transforms(518)]:
            _, _, vm2 = t(pv, dm, vm)
            assert vm2.dtype == torch.bool

    def test_train_returns_compose(self):
        from depth_estimation.data.transforms import Compose
        assert isinstance(get_train_transforms(), Compose)
        assert isinstance(get_val_transforms(), Compose)
