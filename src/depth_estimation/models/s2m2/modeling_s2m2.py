"""
S2M2 — Self-contained single-file stereo model implementation.

S2M2: Scalable Stereo Matching Model for Reliable Depth Estimation
(Min et al., ICCV 2025). All network components (CNN encoder, U-Net feature
pyramid, stacked multi-resolution transformer, optimal-transport disparity
initialisation, global and iterative local refiners, learned upsampling) are
inlined here, ported from https://github.com/junhong-3dv/s2m2. Loads weights
from HuggingFace Hub (``minimok/s2m2``).

Code and weights are released under CC BY-NC 4.0 (non-commercial use only).
Modifications from upstream: inputs arrive already normalised to ``[-1, 1]``
by the shared processor, attention is only gradient-checkpointed when
gradients are enabled, and checkpoint loading is strict.
"""

# Network code adapted from junhong-3dv/s2m2 (Copyright the S2M2 authors),
# licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from ...modeling_utils import BaseDepthModel
from .configuration_s2m2 import S2M2Config

logger = logging.getLogger(__name__)

#: Input sides are padded to a multiple of this (1/32 is the coarsest level).
INPUT_MULTIPLE = 32

_PRECISIONS = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Keys in the released checkpoints that belong to training-only modules the
# inference network does not build.
_UNUSED_CHECKPOINT_PREFIXES = (
    "feature_flow_attn.",
    "upsample_mask_4x.",
    "disp_init.layer_norm1.",
)


def _resolve_precision(precision: str, device: torch.device) -> torch.dtype:
    """Map a config precision string to an autocast dtype for ``device``."""
    if precision == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if precision not in _PRECISIONS:
        raise ValueError(
            f"Unknown precision '{precision}'. "
            f"Choose 'auto' or one of {list(_PRECISIONS)}."
        )
    return _PRECISIONS[precision]


def _autocast(device: torch.device, dtype: torch.dtype):
    """Autocast on ``device`` at ``dtype``; a no-op for float32."""
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


# ======================================================================
# Utilities
# ======================================================================


def _custom_sinc(x: Tensor) -> Tensor:
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        (torch.sin(3.1415 * x) / (3.1415 * x)).to(x.dtype),
    )


def _custom_unfold(x: Tensor, kernel_size: int = 3, padding: int = 1) -> Tensor:
    """3x3 neighbourhood unfold as channel concat (ONNX-friendly)."""
    _, _, h, w = x.shape
    x_pad = F.pad(x, (padding, padding, padding, padding), "replicate")
    x_list = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            x_list.append(x_pad[:, :, i : i + h, j : j + w])
    return torch.cat(x_list, dim=1)


def _get_pe(h: int, w: int, pe_dim: int, dtype: torch.dtype, device) -> Tensor:
    """Relative positional encoding for the 1/32-resolution attention."""
    with torch.no_grad():
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h - 1, h).to(device).to(dtype),
            torch.linspace(0, w - 1, w).to(device).to(dtype),
            indexing="ij",
        )
        rel_x_pos = (grid_x.reshape(-1, 1) - grid_x.reshape(1, -1)).long()
        rel_y_pos = (grid_y.reshape(-1, 1) - grid_y.reshape(1, -1)).long()

        sig = 5 / pe_dim
        dim_t = torch.linspace(-1, 1, pe_dim // 2).to(device).to(dtype)

        x_pos = torch.linspace(-3, 3, 2 * w + 1).to(device).to(dtype).tanh()
        pe_x = F.normalize(
            _custom_sinc((dim_t[None, :] - x_pos[:, None]) / sig), p=2, dim=-1
        )
        rel_pe_x = pe_x[rel_x_pos + w - 1].reshape(h * w, h * w, pe_dim // 2).to(dtype)

        y_pos = torch.linspace(-3, 3, 2 * h + 1).to(device).to(dtype).tanh()
        pe_y = F.normalize(
            _custom_sinc((dim_t[None, :] - y_pos[:, None]) / sig), p=2, dim=-1
        )
        rel_pe_y = pe_y[rel_y_pos + h - 1].reshape(h * w, h * w, pe_dim // 2).to(dtype)

        pe = 0.5 * torch.cat([rel_pe_x, rel_pe_y], dim=2)
    return pe.clone()


def _image_pad(img: Tensor, fill: Tensor, factor: int = INPUT_MULTIPLE) -> Tensor:
    """Centre-pad ``(B, C, H, W)`` to a multiple of ``factor``.

    The border is filled with a blurred copy of the image (upstream's
    ``image_pad``), not zeros, so the padding does not create false edges.
    ``fill`` is the per-channel ``(1, C, 1, 1)`` value of a black pixel in
    ``img``'s normalisation: upstream zero-pads raw ``[0, 255]`` pixels
    before blurring, and the released weights expect that border.
    """
    h, w = img.shape[-2:]
    h_new = math.ceil(h / factor) * factor
    w_new = math.ceil(w / factor) * factor
    pad_h, pad_w = h_new - h, w_new - w
    if pad_h == 0 and pad_w == 0:
        return img

    fill = fill.to(device=img.device, dtype=img.dtype)
    img_pad = F.pad(
        img - fill,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        "constant",
        0,
    ) + fill
    blurred = F.adaptive_avg_pool2d(
        img_pad.float(), output_size=[max(h // factor, 1), max(w // factor, 1)]
    )
    img_pad = F.interpolate(blurred, size=[h_new, w_new], mode="bilinear").to(
        img.dtype
    )
    top, left = pad_h // 2, pad_w // 2
    img_pad[:, :, top : top + h, left : left + w] = img
    return img_pad


def _image_crop(img: Tensor, size: Tuple[int, int]) -> Tensor:
    """Undo :func:`_image_pad`: centre-crop ``(B, C, H, W)`` back to ``size``."""
    h, w = img.shape[-2:]
    h_new, w_new = size
    top, left = (h - h_new) // 2, (w - w_new) // 2
    return img[:, :, top : top + h_new, left : left + w_new]


# ======================================================================
# Attention blocks
# ======================================================================


class SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_pe = use_pe
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)
        if use_pe:
            self.pe_proj = nn.Linear(32, self.head_dim)

    def forward(self, x: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        b, n, _ = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_pe:
            score = torch.einsum("...ic, ...jc -> ...ij", self.scale * q, k)
            attn = score.reshape(b, self.num_heads, n, n).softmax(dim=-1)
            out = torch.einsum("...ij, ...jc -> ...ic", attn, v)
            # contextual relative positional encoding
            pe_sum = torch.einsum("...nij, ijc -> ...nic", attn, pe)
            out = out + self.pe_proj(pe_sum)
        elif torch.is_grad_enabled():
            out = cp.checkpoint(
                F.scaled_dot_product_attention, q, k, v, use_reentrant=False
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        return self.proj(out.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim))


class CrossAttn(nn.Module):
    """Symmetric cross attention between the left and right views."""

    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // num_heads
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)

    def _heads(self, t: Tensor) -> Tensor:
        b, n, _ = t.shape
        return t.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, t: Tensor) -> Tensor:
        b, _, n, _ = t.shape
        return self.proj(t.transpose(1, 2).reshape(b, n, self.num_heads * self.head_dim))

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        x_out = F.scaled_dot_product_attention(
            self._heads(self.q(x)), self._heads(self.k(y)), self._heads(self.v(y))
        )
        y_out = F.scaled_dot_product_attention(
            self._heads(self.q(y)), self._heads(self.k(x)), self._heads(self.v(x))
        )
        return self._merge(x_out), self._merge(y_out)


class SelfAttnBlock1D(nn.Module):
    """Row-wise (epipolar) self attention, pre-norm."""

    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.attn = SelfAttn(dim, num_heads, dim_expansion, use_pe)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        b, h, w, c = z.shape
        z = z.reshape(b * h, w, c)
        z = self.attn(self.norm_pre(z), pe) + z
        return z.reshape(b, h, w, c)


class CrossAttnBlock1D(nn.Module):
    """Row-wise (epipolar) left/right cross attention, pre-norm."""

    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.attn = CrossAttn(dim, num_heads, dim_expansion)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        x, y = self.norm_pre(z).chunk(2, dim=0)
        b, h, w, c = x.shape
        x, y = self.attn(x.reshape(b * h, w, c), y.reshape(b * h, w, c))
        return torch.cat([x.reshape(b, h, w, c), y.reshape(b, h, w, c)], dim=0) + z


class SelfAttnBlock2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.attn = SelfAttn(dim, num_heads, dim_expansion, use_pe)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        b, h, w, c = z.shape
        z = z.reshape(b, h * w, c).contiguous()
        z = self.attn(self.norm_pre(z), pe) + z
        return z.reshape(b, h, w, c).contiguous()


class CrossAttnBlock2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.attn = CrossAttn(dim, num_heads, dim_expansion)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        x, y = self.norm_pre(z).chunk(2, dim=0)
        b, h, w, c = x.shape
        x, y = self.attn(x.reshape(b, h * w, c), y.reshape(b, h * w, c))
        return torch.cat([x.reshape(b, h, w, c), y.reshape(b, h, w, c)], dim=0) + z


class FFN(nn.Module):
    def __init__(self, dim: int, dim_expansion: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_expansion * dim),
            nn.GELU(),
            nn.Linear(dim_expansion * dim, dim),
        )
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.ffn(self.norm_pre(z)) + z


class ConvBlock2D(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dim_expansion: int):
        super().__init__()
        pad = kernel_size // 2
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim_expansion * dim, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(dim_expansion * dim, dim, kernel_size, padding=pad),
        )
        self.convs_1x = nn.Sequential(
            nn.Conv2d(dim, dim_expansion * dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim_expansion * dim, dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.convs(z) + self.convs_1x(z)


class GlobalAttnBlock(nn.Module):
    """Global 2D attention block (optional cross attention first)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_expansion: int,
        use_cross_attn: bool = False,
        use_pe: bool = False,
    ):
        super().__init__()
        self.self_attn = SelfAttnBlock2D(dim, num_heads, dim_expansion, use_pe)
        if use_cross_attn:
            self.cross_attn = CrossAttnBlock2D(dim, num_heads, dim_expansion)
            self.ffn_c = FFN(dim, dim_expansion)
        else:
            self.cross_attn = None
        self.ffn = FFN(dim, dim_expansion)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        z = z.permute(0, 2, 3, 1)
        if self.cross_attn is not None:
            z = self.ffn_c(self.cross_attn(z))
        z = self.ffn(self.self_attn(z, pe))
        return z.permute(0, 3, 1, 2).contiguous()


class BasicAttnBlock(nn.Module):
    """Row-wise cross attention + self attention block."""

    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool = False):
        super().__init__()
        self.cross_attn = CrossAttnBlock1D(dim, num_heads, dim_expansion)
        self.self_attn = SelfAttnBlock1D(dim, num_heads, dim_expansion, use_pe)
        self.ffn_c = FFN(dim, dim_expansion)
        self.ffn = FFN(dim, dim_expansion)

    def forward(self, z: Tensor, pe: Optional[Tensor] = None) -> Tensor:
        z = z.permute(0, 2, 3, 1)
        z = self.ffn_c(self.cross_attn(z))
        z = self.ffn(self.self_attn(z, pe))
        return z.permute(0, 3, 1, 2)


# ======================================================================
# Feature fusion, U-Net and multi-resolution transformer
# ======================================================================


class FeatureFusion(nn.Module):
    def __init__(self, dim: int, kernel_size: int, use_gate: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.use_gate = use_gate
        if use_gate:
            self.feature_gate = nn.Sequential(
                nn.Conv2d(2 * dim, dim, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )

    def forward(self, z0: Tensor, z1: Tensor) -> Tensor:
        z = torch.cat([z0, z1], dim=1)
        if not self.use_gate:
            return self.feature_fusion(z)
        eps = 0.01
        w = self.feature_gate(z).clamp(min=eps, max=1 - eps)
        return self.feature_fusion(z) + w * z0 + (1 - w) * z1


def _down(dim_in: int, dim_out: int) -> nn.Sequential:
    return nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(dim_in, dim_out, kernel_size=1))


def _up(dim_in: int, dim_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(dim_in, dim_out, kernel_size=1),
    )


class Unet(nn.Module):
    """4-level U-Net with global attention at 1/8 of its input resolution."""

    def __init__(
        self,
        dims: List[int],
        dim_expansion: int,
        use_pe: bool,
        n_attn: int = 1,
        use_gate_fusion: bool = True,
    ):
        super().__init__()
        self.use_pe = use_pe

        self.down_conv0 = _down(dims[0], dims[1])
        self.down_conv1 = _down(dims[1], dims[2])
        self.down_conv2 = _down(dims[2], dims[2])
        self.up_conv0 = _up(dims[1], dims[0])
        self.up_conv1 = _up(dims[2], dims[1])
        self.up_conv2 = _up(dims[2], dims[2])

        self.concat_conv0 = FeatureFusion(dims[0], 1, use_gate_fusion)
        self.concat_conv1 = FeatureFusion(dims[1], 1, use_gate_fusion)
        self.concat_conv2 = FeatureFusion(dims[2], 1, use_gate_fusion)

        self.enc0 = ConvBlock2D(dims[0], 3, dim_expansion)
        self.enc1 = ConvBlock2D(dims[1], 3, dim_expansion)
        self.enc2 = ConvBlock2D(dims[2], 3, dim_expansion)
        self.enc3s = nn.ModuleList(
            GlobalAttnBlock(dims[2], 8, dim_expansion, use_cross_attn=False, use_pe=use_pe)
            for _ in range(n_attn)
        )

        self.dec0 = ConvBlock2D(dims[0], 3, dim_expansion)
        self.dec1 = ConvBlock2D(dims[1], 3, dim_expansion)
        self.dec2 = ConvBlock2D(dims[2], 3, dim_expansion)
        self.dec3s = nn.ModuleList(
            GlobalAttnBlock(dims[2], 8, dim_expansion, use_cross_attn=False, use_pe=False)
            for _ in range(n_attn)
        )

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pe = None
        if self.use_pe:
            h, w = z.shape[-2:]
            pe = _get_pe(h // 8, w // 8, 32, z.dtype, z.device)

        z0 = self.enc0(z)
        z1 = self.enc1(self.down_conv0(z0))
        z2 = self.enc2(self.down_conv1(z1))
        z3 = self.down_conv2(z2)
        for block in self.enc3s:
            z3 = block(z3, pe)
        for block in self.dec3s:
            z3 = block(z3, pe)

        z2_new = self.dec2(self.concat_conv2(z2, self.up_conv2(z3)))
        z1_new = self.dec1(self.concat_conv1(z1, self.up_conv1(z2_new)))
        z0_new = self.dec0(self.concat_conv0(z0, self.up_conv0(z1_new)))
        return z0_new, z1_new, z2_new, z3


class MRT(nn.Module):
    """Multi-Resolution Transformer."""

    def __init__(
        self, dims: List[int], num_heads: int, dim_expansion: int, use_gate_fusion: bool
    ):
        super().__init__()
        self.down_conv0 = _down(dims[0], dims[1])
        self.down_conv1 = _down(dims[1], dims[2])
        self.down_conv2 = _down(dims[2], dims[2])
        self.up_conv0 = _up(dims[1], dims[0])
        self.up_conv1 = _up(dims[2], dims[1])
        self.up_conv2 = _up(dims[2], dims[2])

        self.down_concat1 = FeatureFusion(dims[1], 1, use_gate_fusion)
        self.down_concat2 = FeatureFusion(dims[2], 1, use_gate_fusion)
        self.down_concat3 = FeatureFusion(dims[2], 1, use_gate_fusion)
        self.up_concat0 = FeatureFusion(dims[0], 1, use_gate_fusion)
        self.up_concat1 = FeatureFusion(dims[1], 1, use_gate_fusion)
        self.up_concat2 = FeatureFusion(dims[2], 1, use_gate_fusion)

        self.enc_attn0 = BasicAttnBlock(dims[0], 1 * num_heads, dim_expansion)
        self.enc_attn1 = BasicAttnBlock(dims[1], 2 * num_heads, dim_expansion)
        self.enc_attn2 = BasicAttnBlock(dims[2], 4 * num_heads, dim_expansion)
        self.enc_attn3s = nn.ModuleList(
            GlobalAttnBlock(dims[2], 8 * num_heads, dim_expansion, use_cross_attn=True)
            for _ in range(2)
        )
        self.dec_attn0 = BasicAttnBlock(dims[0], 1 * num_heads, dim_expansion)
        self.dec_attn1 = BasicAttnBlock(dims[1], 2 * num_heads, dim_expansion)
        self.dec_attn2 = BasicAttnBlock(dims[2], 4 * num_heads, dim_expansion)
        self.dec_attn3s = nn.ModuleList(
            GlobalAttnBlock(dims[2], 8 * num_heads, dim_expansion, use_cross_attn=True)
            for _ in range(2)
        )

    def forward(self, z0: Tensor, z1: Tensor, z2: Tensor, z3: Tensor):
        z0 = self.enc_attn0(z0)
        z1 = self.enc_attn1(self.down_concat1(z1, self.down_conv0(z0)))
        z2 = self.enc_attn2(self.down_concat2(z2, self.down_conv1(z1)))
        z3 = self.down_concat3(z3, self.down_conv2(z2))
        for block in self.enc_attn3s:
            z3 = block(z3)
        for block in self.dec_attn3s:
            z3 = block(z3)

        z2 = self.dec_attn2(self.up_concat2(z2, self.up_conv2(z3)))
        z1 = self.dec_attn1(self.up_concat1(z1, self.up_conv1(z2)))
        z0 = self.dec_attn0(self.up_concat0(z0, self.up_conv0(z1)))
        return z0, z1, z2, z3


class StackedMRT(nn.Module):
    def __init__(
        self,
        num_transformer: int,
        dims: List[int],
        num_heads: int,
        dim_expansion: int,
        use_gate_fusion: bool,
    ):
        super().__init__()
        self.uformer_list = nn.ModuleList(
            MRT(dims, num_heads, dim_expansion, use_gate_fusion)
            for _ in range(num_transformer)
        )

    def forward(self, z0: Tensor, z1: Tensor, z2: Tensor, z3: Tensor) -> Tensor:
        for uformer in self.uformer_list:
            z0, z1, z2, z3 = uformer(z0, z1, z2, z3)
        return z0.contiguous()


# ======================================================================
# Matching, cost volume and upsampling
# ======================================================================


def _bilinear_sampler(img: Tensor, coords: Tensor) -> Tensor:
    """``grid_sample`` with pixel coordinates.

    The grid is built in float32: in float16 the normalised coordinates are
    off by up to ~0.1 px per row at w ~ 370. ``x * W / (W - 1) - 0.5`` is the
    lookup the released weights were trained with.
    """
    W = torch.tensor(img.shape[-1], dtype=torch.float32, device=img.device)
    H = torch.tensor(img.shape[-2], dtype=torch.float32, device=img.device)
    xgrid, ygrid = coords.float().split([1, 1], dim=-1)
    xgrid = xgrid * W / (W - 1) - 0.5
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    if not torch.is_autocast_enabled(img.device.type):
        # grid_sample needs matching dtypes; under autocast it runs in fp32.
        grid = grid.to(img.dtype)
    return F.grid_sample(img, grid, mode="bilinear", align_corners=True)


class CostVolume:
    """Two-level 1D correlation lookup for the iterative refinement."""

    def __init__(self, cv: Tensor, coords: Tensor, radius: int):
        self.radius = radius
        r = radius
        self.dx = torch.linspace(-r, r, 2 * r + 1, device=cv.device, dtype=cv.dtype).reshape(
            1, 1, 2 * r + 1, 1
        )

        b, h, w, w2 = cv.shape
        cv = cv.reshape(b * h * w, 1, 1, w2)
        # avg_pool2d(kernel=[1, 2]) without a pooling layer (TensorRT limit).
        n, c, hh, _ = cv.shape
        cv_2x = cv.reshape(n, c, hh, w2 // 2, 2).mean(dim=-1)

        self.cv = cv.reshape(b * h, 1, w, w2)
        self.cv_2x = cv_2x.reshape(b * h, 1, w, w2 // 2)
        self.coords = coords.reshape(b * h * w, 1, 1, 1)

    def _lookup(self, volume: Tensor, x0: Tensor, shape) -> Tensor:
        b, h, w = shape
        x0 = x0.reshape(b * h, w, -1, 1)
        y0 = (self.coords + 0 * self.dx).reshape(b * h, w, -1, 1)
        corrs = _bilinear_sampler(volume, torch.cat([x0, y0], dim=-1))
        return corrs.reshape(b, h, w, 2 * self.radius + 1).permute(0, 3, 1, 2)

    def __call__(self, disp: Tensor) -> Tuple[Tensor, Tensor]:
        b, _, h, w = disp.shape
        d = disp.reshape(b * h * w, 1, 1, 1)
        corrs = self._lookup(self.cv, self.coords - d + self.dx, (b, h, w))
        corrs_2x = self._lookup(self.cv_2x, self.coords / 2 - d / 2 + self.dx, (b, h, w))
        return corrs, corrs_2x


class CNNEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1), nn.GELU(), nn.Conv2d(16, 16, kernel_size=1)
        )
        self.conv1_down = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1),
        )
        self.norm1 = nn.GroupNorm(8, output_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv2_down = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_2x = self.norm1(self.conv1_down(self.conv0(x)))
        x_2x = self.conv2(x_2x) + x_2x
        return self.conv2_down(x_2x), x_2x


class UpsampleMask4x(nn.Module):
    """Convex-upsampling weights for 1/4 -> full resolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv_x = nn.ConvTranspose2d(dim, 64, kernel_size=2, stride=2)
        self.conv_y = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 9, kernel_size=2, stride=2),
        )

    def forward(self, feat_x: Tensor, feat_y: Tensor) -> Tensor:
        return self.conv_concat(torch.cat([self.conv_x(feat_x), self.conv_y(feat_y)], dim=1))


class UpsampleMask1x(nn.Module):
    """Edge-guided full-resolution filter weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv_disp = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=False)
        )
        self.conv_rgb = nn.Sequential(
            nn.ConvTranspose2d(3, 16, kernel_size=3, padding=1), nn.ReLU(inplace=False)
        )
        self.conv_ctx = nn.ConvTranspose2d(dim, 16, kernel_size=2, stride=2)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(48, 9, kernel_size=1),
        )

    def forward(self, disp: Tensor, rgb: Tensor, ctx: Tensor) -> Tensor:
        feat = torch.cat([self.conv_disp(disp), self.conv_rgb(rgb), self.conv_ctx(ctx)], dim=1)
        return self.conv_concat(feat)


def _logsumexp_stable(x: Tensor, dim: int, keepdim: bool = False, eps: float = 1e-30):
    m, _ = x.max(dim=dim, keepdim=True)
    y = m + torch.log(torch.clamp((x - m).exp().sum(dim=dim, keepdim=True), min=eps))
    return y if keepdim else y.squeeze(dim)


class DispInit(nn.Module):
    """Initial disparity, confidence and occlusion via optimal transport."""

    def __init__(self, dim: int, ot_iter: int, use_positivity: bool):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.ot_iter = ot_iter
        self.use_positivity = use_positivity

    def _sinkhorn(self, attn: Tensor, log_mu: Tensor, log_nu: Tensor) -> Tensor:
        v = log_nu - _logsumexp_stable(attn, dim=2)
        u = log_mu - _logsumexp_stable(attn + v.unsqueeze(2), dim=3)
        for _ in range(self.ot_iter - 1):
            v = log_nu - _logsumexp_stable(attn + u.unsqueeze(3), dim=2)
            u = log_mu - _logsumexp_stable(attn + v.unsqueeze(2), dim=3)
        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn: Tensor) -> Tensor:
        w = attn.shape[-1]
        # uniform marginals plus a dustbin
        marginal = torch.cat(
            [torch.ones([w], device=attn.device), torch.tensor([w], device=attn.device)]
        ) / (2 * w)
        log_mu = marginal.log().reshape(1, 1, w + 1)
        log_nu = marginal.log().reshape(1, 1, w + 1)
        attn = F.pad(attn, (0, 1, 0, 1), "constant", 0)
        attn = self._sinkhorn(attn, log_mu, log_nu)
        log_const = torch.log(2 * torch.tensor(w, dtype=torch.float32, device=attn.device))
        return (attn[:, :, :-1, :-1].float() + log_const).exp()

    def forward(self, feature: Tensor):
        device = feature.device
        w = feature.shape[-1]
        x_grid = torch.linspace(0, w - 1, w, device=device, dtype=torch.float32)
        if self.use_positivity:
            mask = torch.triu(torch.ones((w, w), dtype=torch.bool, device=device), diagonal=1)
        else:
            mask = torch.zeros((w, w), dtype=torch.bool, device=device)

        feature0, feature1 = self.layer_norm(feature.permute(0, 2, 3, 1)).chunk(2, dim=0)
        cv = torch.einsum("...hic,...hjc -> ...hij", feature0, feature1)

        prob = self._optimal_transport(cv.masked_fill(mask, -1e4))
        masked_prob = prob.masked_fill(mask, 0)

        # sub-pixel disparity from a 5-tap window around the argmax
        prob_max_ind = masked_prob.argmax(dim=3).unsqueeze(3)
        prob_l = 2
        masked_prob_pad = F.pad(masked_prob, (prob_l, prob_l), "constant", 0)
        conf = 0
        correspondence_left = 0
        for idx in range(2 * prob_l + 1):
            weight = torch.gather(masked_prob_pad, index=prob_max_ind + idx, dim=3)
            conf = conf + weight
            correspondence_left = correspondence_left + weight * (prob_max_ind + idx - prob_l)
        eps = 1e-4
        correspondence_left = (correspondence_left + eps) / (conf + eps)
        disparity = (x_grid.reshape(1, 1, w) - correspondence_left.squeeze(3)).unsqueeze(1)
        conf = conf.unsqueeze(1).squeeze(-1)
        occ = masked_prob.sum(dim=3).unsqueeze(1)
        return disparity, conf, occ, cv


# ======================================================================
# Refiners
# ======================================================================


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        c = hidden_dim + input_dim
        p = kernel_size // 2
        self.convz1 = nn.Conv2d(c, hidden_dim, [kernel_size, 1], padding=[p, 0])
        self.convr1 = nn.Conv2d(c, hidden_dim, [kernel_size, 1], padding=[p, 0])
        self.convq1 = nn.Conv2d(c, hidden_dim, [kernel_size, 1], padding=[p, 0])
        self.convz2 = nn.Conv2d(c, hidden_dim, [1, kernel_size], padding=[0, p])
        self.convr2 = nn.Conv2d(c, hidden_dim, [1, kernel_size], padding=[0, p])
        self.convq2 = nn.Conv2d(c, hidden_dim, [1, kernel_size], padding=[0, p])

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        for convz, convr, convq in (
            (self.convz1, self.convr1, self.convq1),
            (self.convz2, self.convr2, self.convq2),
        ):
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(convz(hx))
            r = torch.sigmoid(convr(hx))
            q = torch.tanh(convq(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q
        return h.to(x.dtype)


class GlobalRefiner(nn.Module):
    """Fill in low-confidence disparity without local correlations."""

    def __init__(self, feature_channels: int):
        super().__init__()
        c = feature_channels
        self.init_feat = nn.Sequential(
            nn.Conv2d(2 + c, c, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(c, c, kernel_size=1)
        )
        self.refine_unet = Unet(dims=[c, c, c], dim_expansion=1, use_pe=False, n_attn=1)
        self.out_feat = nn.Sequential(nn.Conv2d(c, 1, kernel_size=3, padding=1))

    def forward(self, ctx: Tensor, disp: Tensor, conf: Tensor) -> Tensor:
        mask = 1.0 * (conf > 0.2)
        conf_logit = (mask * conf).logit(eps=1e-1)
        feat = self.init_feat(
            torch.cat([disp / 1e2 * mask, conf_logit, ctx], dim=1).to(disp.dtype)
        )
        disp_update = self.out_feat(self.refine_unet(feat)[0]) * 1e2
        return (mask * disp + (1 - mask) * disp_update).to(disp.dtype)


class LocalRefiner(nn.Module):
    """Iterative local refinement using cost-volume lookups."""

    def __init__(
        self, feature_channels: int, dim_expansion: int, radius: int, use_gate_fusion: bool
    ):
        super().__init__()
        c = feature_channels
        k = 2 * radius + 1
        self.disp_feat = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(96, 96, kernel_size=3, padding=1)
        )
        self.corr_feat1 = nn.Sequential(
            nn.Conv2d(k, 96, kernel_size=1), nn.GELU(), nn.Conv2d(96, 64, kernel_size=1)
        )
        self.corr_feat2 = nn.Sequential(
            nn.Conv2d(k, 96, kernel_size=1), nn.GELU(), nn.Conv2d(96, 64, kernel_size=1)
        )
        self.conf_occ_feat = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(64, 32, kernel_size=1)
        )
        self.disp_corr_ctx_cat = nn.Sequential(
            nn.Conv2d(256 + c, 2 * c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * c, c, kernel_size=3, padding=1),
        )
        self.refine_unet = Unet(
            dims=[c, c, 2 * c],
            dim_expansion=dim_expansion,
            use_pe=False,
            use_gate_fusion=use_gate_fusion,
            n_attn=1,
        )
        self.disp_update = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, 1, kernel_size=3, padding=1, bias=False),
        )
        self.conf_occ_update = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=False),
        )
        self.gru = ConvGRU(c, c, 3)

    def forward(
        self,
        hidden: Tensor,
        ctx: Tensor,
        disp: Tensor,
        conf: Tensor,
        occ: Tensor,
        cv_fn: Callable,
    ):
        dtype = disp.dtype
        conf_logit = conf.logit(eps=1e-2)
        occ_logit = occ.logit(eps=1e-2)

        corr1, corr2 = cv_fn(disp)
        feat = torch.cat(
            [
                self.disp_feat(disp / 1e2),
                self.corr_feat1(corr1 / 16),
                self.corr_feat2(corr2 / 16),
                ctx,
                self.conf_occ_feat(torch.cat([conf_logit, occ_logit], dim=1).to(dtype)),
            ],
            dim=1,
        ).to(dtype)
        refine_feat = self.refine_unet(self.disp_corr_ctx_cat(feat))[0]
        hidden_new = self.gru(hidden, refine_feat)
        disp_update = self.disp_update(hidden_new)
        conf_update, occ_update = self.conf_occ_update(hidden_new).chunk(2, dim=1)

        conf_new = torch.sigmoid(conf_update + conf_logit).to(dtype)
        occ_new = torch.sigmoid(occ_update + occ_logit).to(dtype)
        disp_new = (disp + disp_update).to(dtype)
        return hidden_new.to(dtype), disp_new, conf_new, occ_new


# ======================================================================
# Full network
# ======================================================================


class S2M2Network(nn.Module):
    """The S2M2 network. Attribute names match the released checkpoints."""

    def __init__(
        self,
        feature_channels: int,
        dim_expansion: int,
        num_transformer: int,
        use_positivity: bool = True,
        refine_iter: int = 3,
    ):
        super().__init__()
        c = feature_channels
        self.use_positivity = use_positivity
        self.refine_iter = refine_iter

        self.cnn_backbone = CNNEncoder(output_dim=c)
        self.feat_pyramid = Unet(
            dims=[c, c, 2 * c],
            dim_expansion=dim_expansion,
            use_gate_fusion=True,
            use_pe=True,
            n_attn=num_transformer * 2,
        )
        self.transformer = StackedMRT(
            num_transformer=num_transformer,
            dims=[c, c, 2 * c],
            num_heads=1,
            dim_expansion=dim_expansion,
            use_gate_fusion=True,
        )
        self.disp_init = DispInit(dim=c, ot_iter=3, use_positivity=use_positivity)
        self.upsample_mask_1x = UpsampleMask1x(c)
        self.upsample_mask_4x_refine = UpsampleMask4x(c)
        self.global_refiner = GlobalRefiner(feature_channels=c)
        self.feat_fusion_layer = FeatureFusion(dim=c, kernel_size=3, use_gate=True)
        self.refiner = LocalRefiner(
            feature_channels=c, dim_expansion=dim_expansion, radius=4, use_gate_fusion=True
        )
        self.ctx_feat = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1), nn.GELU(), nn.Conv2d(c, c, kernel_size=1)
        )

    @staticmethod
    def _upsample4x(x: Tensor, up_weights: Tensor) -> Tensor:
        b, _, h, w = x.shape
        x_unfold = _custom_unfold(x, 3, 1)
        x_unfold = F.interpolate(x_unfold, (h * 4, w * 4), mode="nearest").reshape(
            b, 9, h * 4, w * 4
        )
        return (x_unfold * up_weights.softmax(dim=1)).sum(1, keepdim=True)

    @staticmethod
    def _upsample1x(x: Tensor, filter_weights: Tensor) -> Tensor:
        return (_custom_unfold(x, 3, 1) * filter_weights.softmax(dim=1)).sum(1, keepdim=True)

    def forward(self, img0: Tensor, img1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict left-view disparity, occlusion and confidence.

        Args:
            img0, img1: Left and right images ``(B, 3, H, W)`` normalised to
                ``[-1, 1]``, with ``H`` and ``W`` multiples of 32.

        Returns:
            ``(disparity, occlusion, confidence)``, each ``(B, 1, H, W)``.
        """
        feature_4x, feature_2x = self.cnn_backbone(torch.cat([img0, img1], dim=0))
        feature0_2x, _ = feature_2x.chunk(2, dim=0)

        feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x = self.feat_pyramid(
            feature_4x
        )
        feature_tr_4x = self.transformer(
            feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x
        )

        disp, conf, occ, cv = self.disp_init(feature_tr_4x)

        feature0_tr_4x, _ = feature_tr_4x.chunk(2, dim=0)
        feature0_py_4x, _ = feature_py_4x.chunk(2, dim=0)

        disp = self.global_refiner(feature0_tr_4x.contiguous(), disp.detach(), conf.detach())
        if self.use_positivity:
            disp = disp.clamp(min=0)

        feature0_fusion_4x = self.feat_fusion_layer(feature0_tr_4x, feature0_py_4x)
        ctx0 = self.ctx_feat(feature0_fusion_4x)
        hidden = torch.tanh(ctx0)

        b, _, h, w = feature0_fusion_4x.shape
        # float32: FP16 cannot represent every integer above 2048.
        coords_4x = torch.arange(w, device=feature0_fusion_4x.device, dtype=torch.float32)
        cv_fn = CostVolume(cv, coords_4x.reshape(1, 1, w, 1).repeat(b, h, 1, 1), radius=4)

        for _ in range(self.refine_iter):
            hidden, disp, conf, occ = self.refiner(hidden, ctx0, disp, conf, occ, cv_fn)
            if self.use_positivity:
                disp = disp.clamp(min=0)
            occ = occ * torch.ge(coords_4x.reshape(1, 1, 1, -1) - disp, 0)

        upsample_mask = self.upsample_mask_4x_refine(hidden, feature0_2x)
        disp_up = self._upsample4x(disp * 4, upsample_mask)
        occ_up = self._upsample4x(occ, upsample_mask)
        conf_up = self._upsample4x(conf, upsample_mask)

        filter_weights = self.upsample_mask_1x(disp_up, img0, feature0_2x)
        disp_up = self._upsample1x(disp_up, filter_weights)
        occ_up = self._upsample1x(occ_up, filter_weights)
        conf_up = self._upsample1x(conf_up, filter_weights)
        return disp_up, occ_up, conf_up


# ======================================================================
# Library wrapper
# ======================================================================

ImageLike = Union[str, np.ndarray, "Image.Image", Tensor]  # noqa: F821


class S2M2Model(BaseDepthModel):
    """S2M2 stereo matching model.

    Takes a rectified stereo pair and predicts the left view's disparity
    (plus occlusion and confidence). ``forward()`` takes the pair stacked on
    the channel axis, ``(B, 6, H, W)`` = left RGB then right RGB, which is
    what the shared processor produces for a ``(left, right)`` tuple.

    Usage::

        from depth_estimation import pipeline

        pipe = pipeline("depth-estimation", model="s2m2-l")
        result = pipe(("left.png", "right.png"))  # relative depth (disparity)

        model = pipe.model
        out = model.infer("left.png", "right.png",
                          focal_length_px=1000.0, baseline=0.12)
        out["disparity"]   # (H, W) pixels, native resolution
        out["depth"]       # (H, W) in the baseline's unit
    """

    config_class = S2M2Config
    # forward() takes a 6-channel stereo pair; the generic exporter traces a
    # 3-channel single image.
    _onnx_exportable = False

    def __init__(self, config: S2M2Config):
        super().__init__(config)
        self.net = S2M2Network(
            feature_channels=config.feature_channels,
            dim_expansion=config.dim_expansion,
            num_transformer=config.num_transformer,
            use_positivity=not config.allow_negative,
            refine_iter=config.refine_iter,
        )

    @property
    def precision(self) -> torch.dtype:
        return _resolve_precision(self.config.precision, self.device)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_stereo(self, left: Tensor, right: Tensor) -> Dict[str, Tensor]:
        """Disparity, occlusion and confidence for normalised ``(B, 3, H, W)`` pairs.

        Any ``H`` and ``W`` work: inputs are padded to a multiple of 32 and
        the outputs cropped back. Outputs are float32 ``(B, H, W)``.
        """
        if left.shape != right.shape:
            raise ValueError(
                f"Left and right images must have the same shape, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}."
            )
        height, width = left.shape[-2:]
        mean = torch.tensor(self.config.mean).reshape(1, 3, 1, 1)
        std = torch.tensor(self.config.std).reshape(1, 3, 1, 1)
        black = -mean / std
        left = _image_pad(left.to(self.device), black)
        right = _image_pad(right.to(self.device), black)
        with _autocast(left.device, self.precision):
            disp, occ, conf = self.net(left, right)
        return {
            "disparity": _image_crop(disp, (height, width))[:, 0].float(),
            "occlusion": _image_crop(occ, (height, width))[:, 0].float(),
            "confidence": _image_crop(conf, (height, width))[:, 0].float(),
        }

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            pixel_values: Stereo pairs ``(B, 6, H, W)`` — left RGB in channels
                0-2, right RGB in 3-5 — normalised to ``[-1, 1]``.

        Returns:
            Left-view disparity ``(B, H, W)`` in pixels of the input
            resolution. Disparity is inverse depth (larger = closer).
        """
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 6:
            raise ValueError(
                "S2M2 is a stereo model: forward() takes a (B, 6, H, W) tensor "
                "(left and right RGB stacked on channels), got "
                f"{tuple(pixel_values.shape)}. With the pipeline, pass a "
                "(left, right) tuple: pipe((left_image, right_image))."
            )
        left, right = pixel_values.split(3, dim=1)
        return self.predict_stereo(left, right)["disparity"]

    def _to_tensor(self, image: ImageLike) -> Tensor:
        """Load and normalise one image to ``(1, 3, H, W)`` at native resolution."""
        if isinstance(image, Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(
                    f"Expected a (3, H, W) or (B, 3, H, W) tensor, got {tuple(image.shape)}."
                )
            return image.float()

        from ...processing_utils import DepthProcessor

        rgb = DepthProcessor(self.config)._load_image(image).astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.config.mean, dtype=np.float32)) / np.array(
            self.config.std, dtype=np.float32
        )
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    @torch.no_grad()
    def infer(
        self,
        left: ImageLike,
        right: ImageLike,
        focal_length_px: Optional[float] = None,
        baseline: Optional[float] = None,
        doffs: float = 0.0,
    ) -> Dict[str, Optional[Tensor]]:
        """Stereo inference at native resolution, with optional metric depth.

        Args:
            left, right: The rectified pair. Each is a path, URL, PIL image or
                ``(H, W, 3)`` uint8 RGB array, or a tensor ``(3, H, W)`` /
                ``(B, 3, H, W)`` already normalised to ``[-1, 1]``.
            focal_length_px: Rectified focal length in pixels at this
                resolution.
            baseline: Distance between the camera centres. ``depth`` is in
                this unit (e.g. metres).
            doffs: Principal-point x offset between the cameras
                (``cx_right - cx_left``), as in Middlebury ``calib.txt``.

        Returns:
            Dict with ``disparity`` (pixels), ``occlusion`` (0 = occluded,
            1 = visible in both views), ``confidence`` (probability the
            disparity error is under 4 px) and ``depth``
            (``baseline * focal_length_px / (disparity + doffs)``, ``inf``
            where that denominator is <= 0; ``None`` unless both
            ``focal_length_px`` and ``baseline`` are given). Each is
            ``(H, W)``, or ``(B, H, W)`` for batched tensor input.
        """
        left_t, right_t = self._to_tensor(left), self._to_tensor(right)
        batched = isinstance(left, Tensor) and left.ndim == 4
        out: Dict[str, Optional[Tensor]] = dict(self.predict_stereo(left_t, right_t))

        depth = None
        if focal_length_px is not None and baseline is not None:
            denom = out["disparity"] + doffs
            depth = torch.where(
                denom > 0,
                baseline * focal_length_px / denom.clamp(min=1e-6),
                torch.full_like(denom, float("inf")),
            )
        out["depth"] = depth

        if not batched:
            out = {k: (v[0] if v is not None else None) for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _backbone_module(self) -> nn.Module:
        raise NotImplementedError(
            "S2M2 is a stereo model and DepthTrainer trains monocular models on "
            "single-image datasets, so S2M2 fine-tuning is not supported. The "
            "network (self.net) is a plain nn.Module if you want to train it "
            "with your own stereo loop."
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "S2M2Model":
        """Build the architecture and load the checkpoint from HF Hub."""
        from huggingface_hub import hf_hub_download

        config = S2M2Config.from_variant(model_id)
        for key, value in kwargs.items():
            setattr(config, key, value)

        torch_device = torch.device(device)
        model = cls(config)

        path = hf_hub_download(repo_id=config.hub_repo_id, filename=config.hub_filename)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith(_UNUSED_CHECKPOINT_PREFIXES)
        }
        model.net.load_state_dict(state_dict, strict=True)

        model = model.to(torch_device).eval()
        logger.info(
            "Loaded S2M2 %s from %s/%s",
            config.backbone.upper(),
            config.hub_repo_id,
            config.hub_filename,
        )
        return model
