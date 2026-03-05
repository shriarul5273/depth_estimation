"""
Depth Anything v2 — Single-file model implementation.

Architecture: Vendored DINOv2 encoder + DPT decoder head.
All components inlined per the Transformers single-file policy.
Ported from Depth-Estimation-Compare-demo/Depth-Anything-V2/depth_anything_v2/.

Weights loaded as .pth checkpoints from HuggingFace Hub.

Copyright (c) Meta Platforms, Inc. and affiliates (DINOv2 components).
Licensed under the Apache License, Version 2.0.
"""

import logging
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
    _V2_VARIANT_MAP,
)

logger = logging.getLogger(__name__)


# ============================================================================ #
#  DINOv2 Backbone — Inlined layers (single-file policy)
# ============================================================================ #

# --- DropPath ---


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --- LayerScale ---


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, torch.Tensor] = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# --- Mlp ---


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = nn.GELU,
                 drop: float = 0.0, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --- SwiGLU FFN ---


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = None,
                 drop: float = 0.0, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = None,
                 drop: float = 0.0, bias: bool = True):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features, hidden_features=hidden_features,
            out_features=out_features, bias=bias,
        )


# --- PatchEmbed ---


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)."""

    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_chans: int = 3, embed_dim: int = 768,
                 norm_layer: Optional[Callable] = None,
                 flatten_embedding: bool = True):
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f"Input height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input width {W} is not a multiple of patch width {patch_W}"
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# --- Attention ---


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 proj_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        # Always fallback to standard attention (no xformers dependency)
        if attn_bias is not None:
            raise NotImplementedError("attn_bias requires xFormers")
        return super().forward(x)


# --- Block ---


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, proj_bias: bool = True,
                 ffn_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 init_values=None, drop_path: float = 0.0,
                 act_layer: Callable[..., nn.Module] = nn.GELU,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
                 attn_class: Callable[..., nn.Module] = Attention,
                 ffn_layer: Callable[..., nn.Module] = Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x):
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, residual_func=attn_residual_func,
                                                   sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func,
                                                   sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


# --- BlockChunk ---


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


# --- DinoVisionTransformer ---


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DinoVisionTransformer(nn.Module):
    """DINOv2 Vision Transformer backbone."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
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
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer_cls = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError(f"Unknown ffn_layer: {ffn_layer}")

        blocks_list = [
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                ffn_layer=ffn_layer_cls, init_values=init_values,
            )
            for i in range(depth)
        ]

        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i: i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)
        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append({
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1: self.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
                "x_prenorm": x,
                "masks": masks,
            })
        return output

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
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
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


# --- DINOv2 factory ---


def _vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens, **kwargs,
    )


def _vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens, **kwargs,
    )


def _vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens, **kwargs,
    )


def _vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens, **kwargs,
    )


_DINOv2_MODEL_ZOO = {
    "vits": _vit_small,
    "vitb": _vit_base,
    "vitl": _vit_large,
    "vitg": _vit_giant2,
}


def build_dinov2_backbone(model_name: str) -> DinoVisionTransformer:
    """Build a DINOv2 backbone by name."""
    return _DINOv2_MODEL_ZOO[model_name](
        img_size=518, patch_size=14, init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0, num_register_tokens=0,
        interpolate_antialias=False, interpolate_offset=0.1,
    )


# ============================================================================ #
#  DPT Decoder Head — Inlined (single-file policy)
# ============================================================================ #


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """Build the scratch layers for the DPT head."""
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features, nn.ReLU(False), deconv=False, bn=use_bn,
        expand=False, align_corners=True, size=size,
    )


class DPTHead(nn.Module):
    """Dense Prediction Transformer head for monocular depth estimation."""

    def __init__(self, in_channels, features=256, use_bn=False,
                 out_channels=None, use_clstoken=False):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=oc,
                      kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


# ============================================================================ #
#  Composed Model
# ============================================================================ #


class _DepthAnythingV2Net(nn.Module):
    """Internal composed model: DINOv2 backbone + DPT head."""

    def __init__(self, encoder="vitl", features=256, out_channels=None,
                 use_bn=False, use_clstoken=False, intermediate_layer_idx=None):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]

        self.intermediate_layer_idx = intermediate_layer_idx or {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }.get(encoder, [4, 11, 17, 23])

        self.encoder = encoder
        self.pretrained = build_dinov2_backbone(encoder)
        self.depth_head = DPTHead(
            self.pretrained.embed_dim, features, use_bn,
            out_channels=out_channels, use_clstoken=use_clstoken,
        )

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx, return_class_token=True
        )
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        return depth.squeeze(1)


# ============================================================================ #
#  Public model class
# ============================================================================ #


class DepthAnythingV2Model(BaseDepthModel):
    """Depth Anything v2 model.

    Usage::

        model = DepthAnythingV2Model.from_pretrained("depth-anything-v2-vitb")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = DepthAnythingV2Config

    def __init__(self, config: DepthAnythingV2Config):
        super().__init__(config)
        self.net = _DepthAnythingV2Net(
            encoder=config.backbone,
            features=config.features,
            out_channels=config.out_channels,
            use_bn=config.use_bn,
            use_clstoken=config.use_clstoken,
            intermediate_layer_idx=config.intermediate_layer_idx,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W).
        """
        return self.net(pixel_values)

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        """Convenience method matching the original Depth Anything v2 API.

        Args:
            raw_image: BGR numpy array.
            input_size: Resize input to this resolution.

        Returns:
            Depth map as numpy array at original resolution.
        """
        import cv2
        import numpy as np
        from torchvision.transforms import Compose

        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Simple transform matching the original
        target = input_size
        scale = target / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = new_h - (new_h % 14)
        new_w = new_w - (new_w % 14)
        new_h = max(new_h, 14)
        new_w = max(new_w, 14)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (resized - mean) / std

        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        device = next(self.parameters()).device
        tensor = tensor.to(device)

        depth = self.forward(tensor)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthAnythingV2Model":
        """Load v2 weights from HuggingFace Hub .pth checkpoint."""
        from huggingface_hub import hf_hub_download

        # Resolve variant ID → backbone
        backbone = _V2_VARIANT_MAP.get(model_id)
        if backbone is None:
            raise ValueError(
                f"Cannot resolve backbone from '{model_id}'. "
                f"Use one of: {list(_V2_VARIANT_MAP.keys())}"
            )

        config = DepthAnythingV2Config(backbone=backbone)
        model = cls(config)

        # Download checkpoint from HF Hub
        filepath = hf_hub_download(
            repo_id=config.hub_repo_id,
            filename=config.checkpoint_filename,
            repo_type="model",
        )
        logger.info(f"Downloaded v2 checkpoint: {filepath}")

        state_dict = torch.load(filepath, map_location=device, weights_only=True)
        model.net.load_state_dict(state_dict)
        model = model.to(device)

        logger.info(f"Loaded Depth Anything v2 ({backbone}) from {config.hub_repo_id}")
        return model
