"""
MoGe — Monocular Geometry Estimation model wrapper.

All MoGe source code (DINOv2 backbone, model architecture, geometry utilities)
is inlined here so no external moge or utils3d package is required.

Supports MoGe-1 (relative depth) and MoGe-2 (metric depth, optional normals).
"""

from __future__ import annotations

# ─── Standard library ────────────────────────────────────────────────────────
import importlib
import itertools
import logging
import math
import warnings
from functools import partial
from numbers import Number
from pathlib import Path
from typing import (
    Any, Callable, Dict, IO, List, Literal, Optional, Sequence,
    Tuple, Union,
)

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub import hf_hub_download
from torch import Tensor
from torch.nn.init import trunc_normal_

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_moge import MoGeConfig

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# §1  Minimal utils3d shim (replaces the utils3d package)
# ═════════════════════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════════════════════
# §2  Geometry — NumPy helpers (focal/shift optimisation)
# ═════════════════════════════════════════════════════════════════════════════

def _gauss_newton_1d(residual_fn, jacobian_fn, x0=0.0, ftol=1e-3, max_iter=50):
    x = float(x0)
    for _ in range(max_iter):
        r = residual_fn(x)
        J = jacobian_fn(x)
        JtJ = float(np.dot(J, J))
        if JtJ < 1e-12:
            break
        delta = -float(np.dot(J, r)) / JtJ
        x += delta
        if abs(delta) < ftol * (1.0 + abs(x)):
            break
    return np.float32(x)


def _solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def residuals(shift):
        xy_proj = xy / (z + shift)[:, None]
        denom = np.square(xy_proj).sum()
        f = (xy_proj * uv).sum() / (denom + 1e-12)
        return (f * xy_proj - uv).ravel()

    def jacobian(shift):
        dz = (z + shift)[:, None]
        xy_proj = xy / dz
        denom = np.square(xy_proj).sum()
        f = (xy_proj * uv).sum() / (denom + 1e-12)
        return (f * (-xy / (dz ** 2))).ravel()

    optim_shift = _gauss_newton_1d(residuals, jacobian, x0=0.0, ftol=1e-3)
    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / (np.square(xy_proj).sum() + 1e-12)
    return optim_shift, optim_focal


def _solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def residuals(shift):
        xy_proj = xy / (z + shift)[:, None]
        return (focal * xy_proj - uv).ravel()

    def jacobian(shift):
        dz = (z + shift)[:, None]
        return (focal * (-xy / (dz ** 2))).ravel()

    return _gauss_newton_1d(residuals, jacobian, x0=0.0, ftol=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# §3  Geometry — PyTorch helpers
# ═════════════════════════════════════════════════════════════════════════════

def _normalized_view_plane_uv(
    width: int, height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width  - 1) / width,  span_x * (width  - 1) / width,  width,  dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=-1)


def _recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor = None,
    focal: torch.Tensor = None,
    downsample_size: Tuple[int, int] = (64, 64),
):
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask   = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal  = focal.reshape(-1) if focal is not None else None
    uv     = _normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr     = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr   = None if mask is None else (
        F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    )

    uv_lr_np       = uv_lr.cpu().numpy()
    points_lr_np   = points_lr.detach().cpu().numpy()
    focal_np       = focal.cpu().numpy() if focal is not None else None
    mask_lr_np     = None if mask is None else mask_lr.cpu().numpy()

    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        pts_i = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_i  = uv_lr_np        if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_i.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            s, f = _solve_optimal_focal_shift(uv_i, pts_i)
            optim_focal.append(float(f))
        else:
            s = _solve_optimal_shift(uv_i, pts_i, focal_np[i])
        optim_shift.append(float(s))

    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])
    return optim_focal, optim_shift



# ═════════════════════════════════════════════════════════════════════════════
# §4  DINOv2 layers
# ═════════════════════════════════════════════════════════════════════════════

# ── DropPath ─────────────────────────────────────────────────────────────────
def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class _DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


# ── LayerScale ────────────────────────────────────────────────────────────────
class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ── Mlp ──────────────────────────────────────────────────────────────────────
class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop: float = 0.0, bias: bool = True):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features,    hidden_features, bias=bias)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features,   bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# ── PatchEmbed ────────────────────────────────────────────────────────────────
def _make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten_embedding=True):
        super().__init__()
        image_HW = _make_2tuple(img_size)
        patch_HW = _make_2tuple(patch_size)
        patch_grid = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid
        self.num_patches = patch_grid[0] * patch_grid[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        pH, pW = self.patch_size
        assert H % pH == 0 and W % pW == 0
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# ── SwiGLU ───────────────────────────────────────────────────────────────────
class _SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3  = nn.Linear(hidden_features, out_features,    bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


try:
    _XFORMERS_ENABLED = __import__('os').environ.get("XFORMERS_DISABLED") is None
    if _XFORMERS_ENABLED:
        from xformers.ops import SwiGLU as _XSwiGLU
        _XFORMERS_SWIGLU = True
    else:
        raise ImportError
except ImportError:
    _XSwiGLU = _SwiGLUFFN
    _XFORMERS_SWIGLU = False


class _SwiGLUFFNFused(_XSwiGLU):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features,
                         out_features=out_features, bias=bias)


# ── Attention ─────────────────────────────────────────────────────────────────
try:
    _XFORMERS_ATTN_OK = __import__('os').environ.get("XFORMERS_DISABLED") is None
    if _XFORMERS_ATTN_OK:
        from xformers.ops import memory_efficient_attention as _mea, unbind as _xunbind
        _XFORMERS_ATTN_OK = True
    else:
        raise ImportError
except ImportError:
    _XFORMERS_ATTN_OK = False


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_bias)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _MemEffAttention(_Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not _XFORMERS_ATTN_OK:
            if attn_bias is not None:
                raise AssertionError("xFormers required for nested tensors")
            return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = _xunbind(qkv, 2)
        x = _mea(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── Block / NestedTensorBlock ─────────────────────────────────────────────────
try:
    _XFORMERS_BLOCK_OK = __import__('os').environ.get("XFORMERS_DISABLED") is None
    if _XFORMERS_BLOCK_OK:
        from xformers.ops import fmha as _fmha, scaled_index_add as _sia, index_select_cat as _isc
        _XFORMERS_BLOCK_OK = True
    else:
        raise ImportError
except ImportError:
    _XFORMERS_BLOCK_OK = False

_attn_bias_cache: Dict[Tuple, Any] = {}


def _drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0):
    b, n, d = x.shape
    subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    scale = b / subset_size
    return torch.index_add(x_flat, 0, brange, residual.to(x.dtype), alpha=scale).view_as(x)


def _get_branges_scales(x, sample_drop_ratio=0.0):
    b = x.shape[0]
    subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:subset_size]
    return brange, b / subset_size


def _add_residual(x, brange, residual, scale, scaling_vector=None):
    if scaling_vector is None:
        return torch.index_add(x.flatten(1), 0, brange, residual.to(x.dtype), alpha=scale).view_as(x)
    return _sia(x, brange, residual.to(x.dtype), scaling=scaling_vector, alpha=scale)


def _get_attn_bias_and_cat(x_list, branges=None):
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in _attn_bias_cache:
        seqlens = [x.shape[1] for b, x in zip(batch_sizes, x_list) for _ in range(b)]
        ab = _fmha.BlockDiagonalMask.from_seqlens(seqlens)
        ab._batch_sizes = batch_sizes
        _attn_bias_cache[all_shapes] = ab
    if branges is not None:
        cat = _isc([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        cat = torch.cat(tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list), dim=1)
    return _attn_bias_cache[all_shapes], cat


def _drop_add_residual_stochastic_depth_list(x_list, residual_func, sample_drop_ratio=0.0, scaling_vector=None):
    bs = [_get_branges_scales(x, sample_drop_ratio) for x in x_list]
    branges, scales = [s[0] for s in bs], [s[1] for s in bs]
    attn_bias, x_cat = _get_attn_bias_and_cat(x_list, branges)
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))
    return [_add_residual(x, br, res, sc, scaling_vector).view_as(x)
            for x, br, res, sc in zip(x_list, branges, residual_list, scales)]


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_bias=True,
                 ffn_bias=True, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_class=_Attention, ffn_layer=_Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1        = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp   = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
                               act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2        = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        attn_res = lambda x: self.ls1(self.attn(self.norm1(x)))
        ffn_res  = lambda x: self.ls2(self.mlp(self.norm2(x)))
        if self.training and self.sample_drop_ratio > 0.1:
            x = _drop_add_residual_stochastic_depth(x, attn_res, self.sample_drop_ratio)
            x = _drop_add_residual_stochastic_depth(x, ffn_res,  self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_res(x))
            x = x + self.drop_path1(ffn_res(x))
        else:
            x = x + attn_res(x)
            x = x + ffn_res(x)
        return x


class _NestedTensorBlock(_Block):
    def forward_nested(self, x_list):
        assert isinstance(self.attn, _MemEffAttention)
        if self.training and self.sample_drop_ratio > 0.0:
            attn_res = lambda x, attn_bias=None: self.attn(self.norm1(x), attn_bias=attn_bias)
            ffn_res  = lambda x, attn_bias=None: self.mlp(self.norm2(x))
            sv1 = self.ls1.gamma if isinstance(self.ls1, _LayerScale) else None
            sv2 = self.ls2.gamma if isinstance(self.ls1, _LayerScale) else None
            x_list = _drop_add_residual_stochastic_depth_list(x_list, attn_res, self.sample_drop_ratio, sv1)
            x_list = _drop_add_residual_stochastic_depth_list(x_list, ffn_res,  self.sample_drop_ratio, sv2)
            return x_list
        else:
            attn_res = lambda x, attn_bias=None: self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))
            ffn_res  = lambda x, attn_bias=None: self.ls2(self.mlp(self.norm2(x)))
            attn_bias, x = _get_attn_bias_and_cat(x_list)
            x = x + attn_res(x, attn_bias=attn_bias)
            x = x + ffn_res(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not _XFORMERS_BLOCK_OK:
                raise AssertionError("xFormers required for nested tensors")
            return self.forward_nested(x_or_x_list)
        raise AssertionError


# ═════════════════════════════════════════════════════════════════════════════
# §5  DinoVisionTransformer
# ═════════════════════════════════════════════════════════════════════════════

def _init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        _named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class _BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class _DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 ffn_bias=True, proj_bias=True, drop_path_rate=0.0,
                 drop_path_uniform=False, init_values=None, embed_layer=_PatchEmbed,
                 act_layer=nn.GELU, block_fn=_NestedTensorBlock, ffn_layer="mlp",
                 block_chunks=1, num_register_tokens=0,
                 interpolate_antialias=False, interpolate_offset=0.1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        dpr = ([drop_path_rate] * depth if drop_path_uniform
               else [x.item() for x in torch.linspace(0, drop_path_rate, depth)])

        if ffn_layer == "mlp":
            ffn_cls = _Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_cls = _SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_cls = lambda *a, **kw: nn.Identity()
        else:
            raise NotImplementedError(ffn_layer)

        blocks_list = [
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=ffn_cls, init_values=init_values,
                     attn_class=_MemEffAttention)
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunksize = depth // block_chunks
            chunked = []
            for i in range(0, depth, chunksize):
                chunked.append([nn.Identity()] * i + blocks_list[i: i + chunksize])
            self.blocks = nn.ModuleList([_BlockChunk(p) for p in chunked])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value: bool):
        self._onnx_compatible_mode = value

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        _named_apply(_init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if not self.onnx_compatible_mode and npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0, :]
        patch_pos_embed  = pos_embed[:, 1:, :]
        dim = x.shape[-1]
        h0, w0 = h // self.patch_size, w // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if not self.onnx_compatible_mode and self.interpolate_offset > 0:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sy, sx)
        else:
            kwargs["size"] = (h0, w0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic", antialias=self.interpolate_antialias, **kwargs)
        assert (h0, w0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((class_pos_embed[:, None, :].expand(patch_pos_embed.shape[0], -1, -1),
                          patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, h, w)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total = [], len(self.blocks)
        blocks_to_take = range(total - n, total) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take)
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total - n, total) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take)
        return output

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                       for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken":    x_norm[:, 0],
            "x_norm_regtokens":   x_norm[:, 1: self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }


# ═════════════════════════════════════════════════════════════════════════════
# §6  DINOv2 backbone factory functions
# ═════════════════════════════════════════════════════════════════════════════

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name, patch_size, num_register_tokens=0):
    compact = arch_name.replace("_", "")[:4]
    reg_sfx = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact}{patch_size}{reg_sfx}"


def _make_dinov2_model(*, arch_name="vit_large", img_size=518, patch_size=14,
                       init_values=1.0, ffn_layer="mlp", block_chunks=0,
                       num_register_tokens=0, interpolate_antialias=False,
                       interpolate_offset=0.1, pretrained=True,
                       weights="LVD142M", **kwargs):
    _vit_fns = {
        "vit_small":  _vit_small,
        "vit_base":   _vit_base,
        "vit_large":  _vit_large,
        "vit_giant2": _vit_giant2,
    }
    vit_kwargs = dict(img_size=img_size, patch_size=patch_size, init_values=init_values,
                      ffn_layer=ffn_layer, block_chunks=block_chunks,
                      num_register_tokens=num_register_tokens,
                      interpolate_antialias=interpolate_antialias,
                      interpolate_offset=interpolate_offset)
    vit_kwargs.update(**kwargs)
    model = _vit_fns[arch_name](**vit_kwargs)
    if pretrained:
        model_base_name = _make_dinov2_model_name(arch_name, patch_size)
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


def _vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return _DinoVisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
                                  mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs)


def _vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return _DinoVisionTransformer(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
                                  mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs)


def _vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return _DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
                                  mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs)


def _vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return _DinoVisionTransformer(patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24,
                                  mlp_ratio=4, num_register_tokens=num_register_tokens, **kwargs)


def dinov2_vits14(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_small",  pretrained=pretrained, **kwargs)

def dinov2_vitb14(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_base",   pretrained=pretrained, **kwargs)

def dinov2_vitl14(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_large",  pretrained=pretrained, **kwargs)

def dinov2_vitg14(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused",
                               pretrained=pretrained, **kwargs)

def dinov2_vits14_reg(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained,
                               num_register_tokens=4, interpolate_antialias=True,
                               interpolate_offset=0.0, **kwargs)

def dinov2_vitb14_reg(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained,
                               num_register_tokens=4, interpolate_antialias=True,
                               interpolate_offset=0.0, **kwargs)

def dinov2_vitl14_reg(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained,
                               num_register_tokens=4, interpolate_antialias=True,
                               interpolate_offset=0.0, **kwargs)

def dinov2_vitg14_reg(*, pretrained=True, **kwargs):
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused",
                               pretrained=pretrained, num_register_tokens=4,
                               interpolate_antialias=True, interpolate_offset=0.0, **kwargs)


_BACKBONE_LOADERS: Dict[str, Callable] = {
    "dinov2_vits14":     dinov2_vits14,
    "dinov2_vitb14":     dinov2_vitb14,
    "dinov2_vitl14":     dinov2_vitl14,
    "dinov2_vitg14":     dinov2_vitg14,
    "dinov2_vits14_reg": dinov2_vits14_reg,
    "dinov2_vitb14_reg": dinov2_vitb14_reg,
    "dinov2_vitl14_reg": dinov2_vitl14_reg,
    "dinov2_vitg14_reg": dinov2_vitg14_reg,
}


# ═════════════════════════════════════════════════════════════════════════════
# §7  Model utilities
# ═════════════════════════════════════════════════════════════════════════════

def _wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _Wrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
    module.__class__ = _Wrapper
    return module


def _unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def _wrap_dinov2_attention_with_sdpa(module: nn.Module):
    class _W(module.__class__):
        def forward(self, x: Tensor, attn_bias=None) -> Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = torch.unbind(qkv, 0)
            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    module.__class__ = _W
    return module


# ═════════════════════════════════════════════════════════════════════════════
# §8  MoGe v2 building blocks  (modules.py)
# ═════════════════════════════════════════════════════════════════════════════

class _ResidualConvBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 kernel_size=3, padding_mode='replicate',
                 activation: Literal['relu','leaky_relu','silu','elu'] = 'relu',
                 in_norm: Literal['group_norm','layer_norm','instance_norm','none'] = 'layer_norm',
                 hidden_norm: Literal['group_norm','layer_norm','instance_norm'] = 'group_norm'):
        super().__init__()
        if out_channels    is None: out_channels    = in_channels
        if hidden_channels is None: hidden_channels = in_channels
        act = {'relu': nn.ReLU, 'leaky_relu': partial(nn.LeakyReLU, negative_slope=0.2),
               'silu': nn.SiLU, 'elu': nn.ELU}[activation]

        def _norm(n, c):
            if n == 'group_norm':   return nn.GroupNorm(c // 32, c)
            if n == 'layer_norm':   return nn.GroupNorm(1, c)
            if n == 'instance_norm': return nn.InstanceNorm2d(c)
            return nn.Identity()

        self.layers = nn.Sequential(
            _norm(in_norm, in_channels), act(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode),
            _norm(hidden_norm, hidden_channels), act(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2, padding_mode=padding_mode),
        )
        self.skip_connection = (nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels
                                else nn.Identity())

    def forward(self, x):
        return self.layers(x) + self.skip_connection(x)


class _DINOv2Encoder(nn.Module):
    def __init__(self, backbone: str, intermediate_layers, dim_out: int, **deprecated_kwargs):
        super().__init__()
        self.intermediate_layers = intermediate_layers
        self.backbone_name = backbone
        loader = _BACKBONE_LOADERS[backbone]
        self.backbone = loader(pretrained=False)
        self.dim_features = self.backbone.blocks[0].attn.qkv.in_features
        self.num_features = (intermediate_layers if isinstance(intermediate_layers, int)
                             else len(intermediate_layers))
        self.output_projections = nn.ModuleList([
            nn.Conv2d(self.dim_features, dim_out, kernel_size=1)
            for _ in range(self.num_features)
        ])
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value):
        self._onnx_compatible_mode = value
        self.backbone.onnx_compatible_mode = value

    def init_weights(self):
        pretrained = _BACKBONE_LOADERS[self.backbone_name](pretrained=True).state_dict()
        self.backbone.load_state_dict(pretrained)

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            _wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            _wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def forward(self, image, token_rows, token_cols, return_class_token=False):
        image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14),
                                 mode="bilinear", align_corners=False,
                                 antialias=not self.onnx_compatible_mode)
        image_14 = (image_14 - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(image_14, n=self.intermediate_layers,
                                                          return_class_token=True)
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
            for proj, (feat, _) in zip(self.output_projections, features)
        ], dim=1).sum(dim=1)
        if return_class_token:
            return x, features[-1][1]
        return x


class _Resampler(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 type_: Literal['pixel_shuffle','nearest','bilinear','conv_transpose','pixel_unshuffle','avg_pool','max_pool'],
                 scale_factor=2):
        if type_ == 'pixel_shuffle':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels * (scale_factor**2), 3, 1, 1, padding_mode='replicate'),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='replicate'),
            )
            for i in range(1, scale_factor**2):
                self[0].weight.data[i::scale_factor**2] = self[0].weight.data[0::scale_factor**2]
                self[0].bias.data[i::scale_factor**2]   = self[0].bias.data[0::scale_factor**2]
        elif type_ in ('nearest', 'bilinear'):
            nn.Sequential.__init__(self,
                nn.Upsample(scale_factor=scale_factor, mode=type_,
                            align_corners=False if type_ == 'bilinear' else None),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode='replicate'),
            )
        elif type_ == 'conv_transpose':
            nn.Sequential.__init__(self,
                nn.ConvTranspose2d(in_channels, out_channels, scale_factor, stride=scale_factor),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='replicate'),
            )
            self[0].weight.data[:] = self[0].weight.data[:, :, :1, :1]
        elif type_ == 'pixel_unshuffle':
            nn.Sequential.__init__(self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(in_channels * (scale_factor**2), out_channels, 3, 1, 1, padding_mode='replicate'),
            )
        elif type_ == 'avg_pool':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode='replicate'),
                nn.AvgPool2d(scale_factor, scale_factor),
            )
        elif type_ == 'max_pool':
            nn.Sequential.__init__(self,
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode='replicate'),
                nn.MaxPool2d(scale_factor, scale_factor),
            )
        else:
            raise ValueError(f'Unsupported resampler: {type_}')


class _MLPHead(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(self, *itertools.chain(
            *[(nn.Linear(di, do), nn.ReLU(inplace=True)) for di, do in zip(dims[:-2], dims[1:-1])],
            [nn.Linear(dims[-2], dims[-1])],
        ))


class _ConvStack(nn.Module):
    def __init__(self, dim_in, dim_res_blocks, dim_out, resamplers,
                 dim_times_res_block_hidden=1, num_res_blocks=1,
                 res_block_in_norm='layer_norm', res_block_hidden_norm='group_norm',
                 activation='relu'):
        super().__init__()
        dim_in_list = dim_in if isinstance(dim_in, Sequence) else itertools.repeat(dim_in)
        dim_out_list = dim_out if isinstance(dim_out, Sequence) else itertools.repeat(dim_out)
        resampler_list = resamplers if isinstance(resamplers, Sequence) else itertools.repeat(resamplers)

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(d, dr, 1) if d is not None else nn.Identity()
            for d, dr in zip(dim_in_list, dim_res_blocks)
        ])
        self.resamplers = nn.ModuleList([
            _Resampler(dp, ds, scale_factor=2, type_=r)
            for dp, ds, r in zip(dim_res_blocks[:-1], dim_res_blocks[1:], resampler_list)
        ])
        self.res_blocks = nn.ModuleList([
            nn.Sequential(*(
                _ResidualConvBlockV2(dr, dr, dim_times_res_block_hidden * dr,
                                    activation=activation,
                                    in_norm=res_block_in_norm, hidden_norm=res_block_hidden_norm)
                for _ in range(num_res_blocks[i] if isinstance(num_res_blocks, list) else num_res_blocks)
            )) for i, dr in enumerate(dim_res_blocks)
        ])
        self.output_blocks = nn.ModuleList([
            nn.Conv2d(dr, do, 1) if do is not None else nn.Identity()
            for do, dr in zip(dim_out_list, dim_res_blocks)
        ])

    def enable_gradient_checkpointing(self):
        for i in range(len(self.resamplers)):
            self.resamplers[i] = _wrap_module_with_gradient_checkpointing(self.resamplers[i])
        for i in range(len(self.res_blocks)):
            for j in range(len(self.res_blocks[i])):
                self.res_blocks[i][j] = _wrap_module_with_gradient_checkpointing(self.res_blocks[i][j])

    def forward(self, in_features):
        out_features = []
        x = None
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if x is None:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


# ═════════════════════════════════════════════════════════════════════════════
# §9  MoGe v1 architecture
# ═════════════════════════════════════════════════════════════════════════════

class _ResidualConvBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=None,
                 padding_mode='replicate',
                 activation: Literal['relu','leaky_relu','silu','elu'] = 'relu',
                 norm: Literal['group_norm','layer_norm'] = 'group_norm'):
        super().__init__()
        if out_channels    is None: out_channels    = in_channels
        if hidden_channels is None: hidden_channels = in_channels
        act = {'relu':       lambda: nn.ReLU(inplace=True),
               'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True),
               'silu':       lambda: nn.SiLU(inplace=True),
               'elu':        lambda: nn.ELU(inplace=True)}[activation]
        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels), act(),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels), act(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, padding_mode=padding_mode),
        )
        self.skip_connection = (nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels
                                else nn.Identity())

    def forward(self, x):
        return self.layers(x) + self.skip_connection(x)


class _V1Head(nn.Module):
    def __init__(self, num_features, dim_in, dim_out, dim_proj=512,
                 dim_upsample=(256, 128, 128), dim_times_res_block_hidden=1,
                 num_res_blocks=1, res_block_norm='group_norm',
                 last_res_blocks=0, last_conv_channels=32, last_conv_size=1):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, dim_proj, 1) for _ in range(num_features)
        ])
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(ic + 2, oc),
                *(_ResidualConvBlockV1(oc, oc, dim_times_res_block_hidden * oc,
                                      activation='relu', norm=res_block_norm)
                  for _ in range(num_res_blocks))
            ) for ic, oc in zip([dim_proj] + list(dim_upsample[:-1]), dim_upsample)
        ])
        self.output_block = nn.ModuleList([
            self._make_output_block(dim_upsample[-1] + 2, do, dim_times_res_block_hidden,
                                    last_res_blocks, last_conv_channels, last_conv_size, res_block_norm)
            for do in (dim_out if isinstance(dim_out, (list, tuple)) else [dim_out])
        ])

    def _make_upsampler(self, in_c, out_c):
        up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.Conv2d(out_c, out_c, 3, 1, 1, padding_mode='replicate'),
        )
        up[0].weight.data[:] = up[0].weight.data[:, :, :1, :1]
        return up

    def _make_output_block(self, dim_in, dim_out, dim_times, last_res, last_ch, last_sz, norm):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_ch, 3, 1, 1, padding_mode='replicate'),
            *(_ResidualConvBlockV1(last_ch, last_ch, dim_times * last_ch,
                                   activation='relu', norm=norm)
              for _ in range(last_res)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, dim_out, last_sz, 1, last_sz // 2, padding_mode='replicate'),
        )

    def forward(self, hidden_states, image):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
            for proj, (feat, _) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        for block in self.upsample_blocks:
            uv = _normalized_view_plane_uv(x.shape[-1], x.shape[-2],
                                           aspect_ratio=img_w / img_h,
                                           dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x  = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False)
        uv = _normalized_view_plane_uv(img_w, img_h, aspect_ratio=img_w / img_h,
                                       dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x  = torch.cat([x, uv], dim=1)

        output = [torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
                  for blk in self.output_block]
        return output


class _MoGeModelV1(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, encoder='dinov2_vitl14', intermediate_layers=4,
                 dim_proj=512, dim_upsample=(256, 128, 128),
                 dim_times_res_block_hidden=1, num_res_blocks=1,
                 remap_output='linear', res_block_norm='group_norm',
                 num_tokens_range=(1200, 2500), last_res_blocks=0,
                 last_conv_channels=32, last_conv_size=1, mask_threshold=0.5,
                 **deprecated_kwargs):
        super().__init__()
        if deprecated_kwargs:
            if 'trained_area_range' in deprecated_kwargs:
                tar = deprecated_kwargs.pop('trained_area_range')
                num_tokens_range = [tar[0] // 196, tar[1] // 196]
            warnings.warn(f"Ignored deprecated kwargs: {deprecated_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold

        loader = _BACKBONE_LOADERS[encoder]
        self.backbone = loader(pretrained=False)
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features

        self.head = _V1Head(
            num_features=(intermediate_layers if isinstance(intermediate_layers, int)
                          else len(intermediate_layers)),
            dim_in=dim_feature,
            dim_out=[3, 1],
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def device(self): return next(self.parameters()).device

    @property
    def dtype(self):  return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_kwargs=None, **hf_kwargs):
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
        else:
            path = hf_hub_download(repo_id=pretrained_model_name_or_path, repo_type="model",
                                   filename="model.pt", **hf_kwargs)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        cfg = checkpoint['model_config']
        if model_kwargs:
            cfg.update(model_kwargs)
        model = cls(**cfg)
        model.load_state_dict(checkpoint['model'])
        return model

    def _remap_points(self, points):
        if self.remap_output == 'linear':
            pass
        elif self.remap_output == 'sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap_output: {self.remap_output}")
        return points

    def forward(self, image, num_tokens):
        oh, ow = image.shape[-2:]
        resize_factor = ((num_tokens * 196) / (oh * ow)) ** 0.5
        rh, rw = int(oh * resize_factor), int(ow * resize_factor)
        image = F.interpolate(image, (rh, rw), mode="bicubic", align_corners=False, antialias=True)
        image = (image - self.image_mean) / self.image_std
        image_14 = F.interpolate(image, (rh // 14 * 14, rw // 14 * 14),
                                  mode="bilinear", align_corners=False, antialias=True)
        features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers,
                                                          return_class_token=True)
        output = self.head(features, image)
        points, mask = output
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            points = F.interpolate(points, (oh, ow), mode='bilinear', align_corners=False, antialias=False)
            mask   = F.interpolate(mask,   (oh, ow), mode='bilinear', align_corners=False, antialias=False)
            points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
            points = self._remap_points(points)
        return {'points': points, 'mask': mask}

    @torch.inference_mode()
    def infer(self, image, fov_x=None, resolution_level=9, num_tokens=None,
              apply_mask=True, use_fp16=True):
        if image.dim() == 3:
            omit_batch = True
            image = image.unsqueeze(0)
        else:
            omit_batch = False
        image = image.to(dtype=self.dtype, device=self.device)
        oh, ow = image.shape[-2:]
        aspect_ratio = ow / oh
        if num_tokens is None:
            mn, mx = self.num_tokens_range
            num_tokens = int(mn + (resolution_level / 9) * (mx - mn))

        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens)
        points, mask = output['points'], output['mask']

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            points = points.float(); mask = mask.float()
            if isinstance(fov_x, torch.Tensor): fov_x = fov_x.float()
            mask_binary = mask > self.mask_threshold

            if fov_x is None:
                _, shift = _recover_focal_shift(points, mask_binary)
            else:
                focal = aspect_ratio / (1 + aspect_ratio**2)**0.5 / torch.tan(
                    torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = _recover_focal_shift(points, mask_binary, focal=focal)

            depth = points[..., 2] + shift[..., None, None]

            if apply_mask:
                depth = torch.where(mask_binary, depth, torch.inf)

        ret = {'depth': depth, 'mask': mask_binary}
        if omit_batch:
            ret = {k: v.squeeze(0) for k, v in ret.items()}
        return ret


# ═════════════════════════════════════════════════════════════════════════════
# §10  MoGe v2 architecture
# ═════════════════════════════════════════════════════════════════════════════

class _MoGeModelV2(nn.Module):
    def __init__(self, encoder: Dict[str, Any], neck: Dict[str, Any],
                 points_head=None, mask_head=None, normal_head=None, scale_head=None,
                 remap_output='linear', num_tokens_range=(1200, 3600), **deprecated_kwargs):
        super().__init__()
        if deprecated_kwargs:
            warnings.warn(f"Ignored deprecated kwargs: {deprecated_kwargs}")
        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        self.encoder = _DINOv2Encoder(**encoder)
        self.neck    = _ConvStack(**neck)
        if points_head is not None: self.points_head = _ConvStack(**points_head)
        if mask_head   is not None: self.mask_head   = _ConvStack(**mask_head)
        if normal_head is not None: self.normal_head = _ConvStack(**normal_head)
        if scale_head  is not None: self.scale_head  = _MLPHead(**scale_head)

    @property
    def device(self): return next(self.parameters()).device

    @property
    def dtype(self):  return next(self.parameters()).dtype

    @property
    def onnx_compatible_mode(self):
        return getattr(self, "_onnx_compatible_mode", False)

    @onnx_compatible_mode.setter
    def onnx_compatible_mode(self, value):
        self._onnx_compatible_mode = value
        self.encoder.onnx_compatible_mode = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_kwargs=None, **hf_kwargs):
        if Path(pretrained_model_name_or_path).exists():
            ckpt_path = pretrained_model_name_or_path
        else:
            ckpt_path = hf_hub_download(repo_id=pretrained_model_name_or_path, repo_type="model",
                                        filename="model.pt", **hf_kwargs)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        cfg = checkpoint['model_config']
        if model_kwargs:
            cfg.update(model_kwargs)
        model = cls(**cfg)
        model.load_state_dict(checkpoint['model'], strict=False)
        return model

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.neck.enable_gradient_checkpointing()
        for head in ('points_head', 'normal_head', 'mask_head'):
            if hasattr(self, head):
                getattr(self, head).enable_gradient_checkpointing()

    def enable_pytorch_native_sdpa(self):
        self.encoder.enable_pytorch_native_sdpa()

    def _remap_points(self, points):
        if self.remap_output == 'linear':
            pass
        elif self.remap_output == 'sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == 'sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap_output: {self.remap_output}")
        return points

    def forward(self, image, num_tokens):
        B, _, img_h, img_w = image.shape
        dtype, device = image.dtype, image.device
        aspect_ratio = img_w / img_h
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        for level in range(5):
            uv = _normalized_view_plane_uv(base_w * 2**level, base_h * 2**level,
                                           aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
            features[level] = uv if features[level] is None else torch.cat([features[level], uv], dim=1)

        features = self.neck(features)

        points = getattr(self, 'points_head', None)
        normal = getattr(self, 'normal_head', None)
        mask   = getattr(self, 'mask_head',   None)
        scale  = getattr(self, 'scale_head',  None)

        points = self.points_head(features)[-1] if hasattr(self, 'points_head') else None
        normal = self.normal_head(features)[-1] if hasattr(self, 'normal_head') else None
        mask   = self.mask_head(features)[-1]   if hasattr(self, 'mask_head')   else None
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None

        points, normal, mask = (
            F.interpolate(v, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False)
            if v is not None else None
            for v in (points, normal, mask)
        )
        if points is not None:
            points = self._remap_points(points.permute(0, 2, 3, 1))
        if normal is not None:
            normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        ret = {'points': points, 'normal': normal, 'mask': mask, 'metric_scale': metric_scale}
        return {k: v for k, v in ret.items() if v is not None}

    @torch.inference_mode()
    def infer(self, image, num_tokens=None, resolution_level=9,
              apply_mask=True, fov_x=None, use_fp16=True):
        if image.dim() == 3:
            omit_batch = True
            image = image.unsqueeze(0)
        else:
            omit_batch = False
        image = image.to(dtype=self.dtype, device=self.device)
        oh, ow = image.shape[-2:]
        aspect_ratio = ow / oh

        if num_tokens is None:
            mn, mx = self.num_tokens_range
            num_tokens = int(mn + (resolution_level / 9) * (mx - mn))

        with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                            enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens=num_tokens)

        points, normal, mask, metric_scale = (output.get(k) for k in ('points', 'normal', 'mask', 'metric_scale'))
        points, normal, mask, metric_scale = map(
            lambda x: x.float() if isinstance(x, torch.Tensor) else x,
            (points, normal, mask, metric_scale)
        )
        if isinstance(fov_x, torch.Tensor): fov_x = fov_x.float()

        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            mask_binary = (mask > 0.5) if mask is not None else None

            if points is not None:
                if fov_x is None:
                    _, shift = _recover_focal_shift(points, mask_binary)
                else:
                    focal = aspect_ratio / (1 + aspect_ratio**2)**0.5 / torch.tan(
                        torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                    if focal.ndim == 0:
                        focal = focal[None].expand(points.shape[0])
                    _, shift = _recover_focal_shift(points, mask_binary, focal=focal)

                depth = points[..., 2] + shift[..., None, None]
                if mask_binary is not None:
                    mask_binary &= depth > 0
            else:
                depth = None

            if metric_scale is not None and depth is not None:
                depth *= metric_scale[:, None, None]

            if apply_mask and mask_binary is not None:
                if depth  is not None: depth  = torch.where(mask_binary, depth, torch.inf)
                if normal is not None: normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

        ret = {'depth': depth, 'mask': mask_binary, 'normal': normal}
        ret = {k: v for k, v in ret.items() if v is not None}
        if omit_batch:
            ret = {k: v.squeeze(0) for k, v in ret.items()}
        return ret


# ═════════════════════════════════════════════════════════════════════════════
# §11  Public MoGeModel wrapper  (BaseDepthModel)
# ═════════════════════════════════════════════════════════════════════════════

class MoGeModel(BaseDepthModel):
    """MoGe monocular geometry estimation model.

    - **MoGe-1** (``moge-v1``): relative/affine depth, ViT-L backbone.
    - **MoGe-2** (``moge-v2-*``): metric depth in metres, ViT-S/B/L backbones.
      ``*-normal`` variants additionally predict a surface normal map.

    Usage::

        model = MoGeModel.from_pretrained("moge-v2-vitl")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = MoGeConfig

    def __init__(self, config: MoGeConfig):
        super().__init__(config)
        self._net = None  # set by _load_pretrained_weights

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run depth estimation.

        Args:
            pixel_values: (B, 3, H, W) tensor with values in **[0, 1]**.

        Returns:
            Depth tensor of shape (B, H, W).
        """
        device = next(self._net.parameters()).device
        pixel_values = pixel_values.to(device)

        depths = []
        for i in range(pixel_values.shape[0]):
            output = self._net.infer(pixel_values[i])
            depth  = output["depth"]
            depth  = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
            depths.append(depth)
        return torch.stack(depths)

    @classmethod
    def _load_pretrained_weights(cls, model_id: str, device: str = "cpu", **kwargs) -> "MoGeModel":
        config = MoGeConfig.from_variant(model_id)
        torch_device = torch.device(device)

        if config.version == "v1":
            net = _MoGeModelV1.from_pretrained(config.hub_repo_id)
        else:
            net = _MoGeModelV2.from_pretrained(config.hub_repo_id)

        net = net.to(torch_device).eval()
        model = cls(config)
        model._net = net
        logger.info("Loaded MoGe-%s (%s) from %s",
                    config.version.upper(), config.backbone, config.hub_repo_id)
        return model
