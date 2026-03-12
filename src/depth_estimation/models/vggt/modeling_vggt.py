"""
VGGT — self-contained single-file model implementation.

All architecture components (DINOv2 ViT layers, alternating-attention aggregator,
DPT head, camera head) are inlined here so no external vggt package is required.

Loads weights from HuggingFace Hub (facebook/VGGT-1B).
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import logging
import math
import numpy as np
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from ...modeling_utils import BaseDepthModel
from .configuration_vggt import VGGTConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Layer primitives
# ═══════════════════════════════════════════════════════════════════════════════

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
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten_embedding=True):
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

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        pH, pW = self.patch_size
        assert H % pH == 0 and W % pW == 0
        x = self.proj(x)
        H2, W2 = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H2, W2, self.embed_dim)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features,
                         out_features=out_features, bias=bias)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, norm_layer=nn.LayerNorm,
                 qk_norm=False, fused_attn=True, rope=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v,
                                               dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        return super().forward(x)


def drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio=0.0, pos=None):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]
    residual = residual_func(x_subset, pos=pos[brange]) if pos is not None else residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    scale = b / sample_subset_size
    return torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=scale).view_as(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, proj_bias=True,
                 ffn_bias=True, drop=0.0, attn_drop=0.0, init_values=None,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_class=Attention, ffn_layer=Mlp, qk_norm=False, fused_attn=True, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                               attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm,
                               fused_attn=fused_attn, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_res(x, pos=None):
            return self.ls1(self.attn(self.norm1(x), pos=pos))
        def ffn_res(x):
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(x, pos=pos, residual_func=attn_res,
                                                   sample_drop_ratio=self.sample_drop_ratio)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_res,
                                                   sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_res(x, pos=pos))
            x = x + self.drop_path1(ffn_res(x))
        else:
            x = x + attn_res(x, pos=pos)
            x = x + ffn_res(x)
        return x


class NestedTensorBlock(Block):
    """Block that can handle list inputs; falls back to standard forward for tensors."""
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        raise AssertionError("NestedTensorBlock list forward requires xFormers")


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Rotary Position Embedding (RoPE 2D)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionGetter:
    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size, height, width, device):
        if (height, width) not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self.position_cache[height, width] = torch.cartesian_prod(y, x)
        cached = self.position_cache[height, width]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency=100.0, scaling_factor=1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[key]

    @staticmethod
    def _rotate_features(x):
        d = x.shape[-1]
        return torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return tokens * cos + self._rotate_features(tokens) * sin

    def forward(self, tokens, positions):
        assert tokens.size(-1) % 2 == 0
        assert positions.ndim == 3 and positions.shape[-1] == 2
        feature_dim = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_pos, tokens.device, tokens.dtype)
        v_feat, h_feat = tokens.chunk(2, dim=-1)
        v_feat = self._apply_1d_rope(v_feat, positions[..., 0], cos_comp, sin_comp)
        h_feat = self._apply_1d_rope(h_feat, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((v_feat, h_feat), dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  DINOv2 Vision Transformer
# ═══════════════════════════════════════════════════════════════════════════════

def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights_vit_timm(module, name=""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True,
                 proj_bias=True, drop_path_rate=0.0, drop_path_uniform=False,
                 init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU,
                 block_fn=NestedTensorBlock, ffn_layer="mlp", block_chunks=1,
                 num_register_tokens=0, interpolate_antialias=False,
                 interpolate_offset=0.1, qk_norm=False):
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
        self.use_reentrant = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
                                if num_register_tokens else None)

        dpr = ([drop_path_rate] * depth if drop_path_uniform
               else [x.item() for x in torch.linspace(0, drop_path_rate, depth)])

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer_cls = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer_cls = lambda **kw: nn.Identity()
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=ffn_layer_cls, init_values=init_values, qk_norm=qk_norm)
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunksize = depth // block_chunks
            chunked = [[nn.Identity()] * i + blocks_list[i:i + chunksize]
                       for i in range(0, depth, chunksize)]
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
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
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic", antialias=self.interpolate_antialias, **kwargs)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
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
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            if self.training:
                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            else:
                x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
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
            outputs = [out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                       .permute(0, 3, 1, 2).contiguous() for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=384, depth=12,
                                 num_heads=6, mlp_ratio=4,
                                 block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=768, depth=12,
                                 num_heads=12, mlp_ratio=4,
                                 block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24,
                                 num_heads=16, mlp_ratio=4,
                                 block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=1536, depth=40,
                                 num_heads=24, mlp_ratio=4,
                                 block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Aggregator
# ═══════════════════════════════════════════════════════════════════════════════

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def slice_expand_and_flatten(token_tensor, B, S):
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.view(B * S, *combined.shape[2:])


class Aggregator(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24,
                 num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
                 block_fn=Block, qkv_bias=True, proj_bias=True, ffn_bias=True,
                 patch_embed="dinov2_vitl14_reg", aa_order=None, aa_block_size=1,
                 qk_norm=True, rope_freq=100, init_values=0.01):
        super().__init__()
        if aa_order is None:
            aa_order = ["frame", "global"]
        self._build_patch_embed(patch_embed, img_size, patch_size, num_register_tokens,
                                embed_dim=embed_dim)
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     init_values=init_values, qk_norm=qk_norm, rope=self.rope)
            for _ in range(depth)
        ])
        self.global_blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                     init_values=init_values, qk_norm=qk_norm, rope=self.rope)
            for _ in range(depth)
        ])
        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
        self.use_reentrant = False

    def _build_patch_embed(self, patch_embed, img_size, patch_size, num_register_tokens,
                           interpolate_antialias=True, interpolate_offset=0.0,
                           block_chunks=0, init_values=1.0, embed_dim=1024):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {"dinov2_vitl14_reg": vit_large, "dinov2_vitb14_reg": vit_base,
                          "dinov2_vits14_reg": vit_small, "dinov2_vitg2_reg": vit_giant2}
            self.patch_embed = vit_models[patch_embed](
                img_size=img_size, patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks, init_values=init_values)

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, S, 3, H, W) in range [0, 1].
        Returns:
            (list[Tensor], int): aggregated token list and patch_start_idx.
        """
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size,
                                       device=images.device)
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2,
                                      device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx = global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_ints = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos)
                elif attn_type == "global":
                    tokens, global_idx, global_ints = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_ints)):
                output_list.append(torch.cat([frame_ints[i], global_ints[i]], dim=-1))

        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            blk = self.frame_blocks[frame_idx]
            if self.training:
                tokens = checkpoint(blk, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = blk(tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)
        intermediates = []
        for _ in range(self.aa_block_size):
            blk = self.global_blocks[global_idx]
            if self.training:
                tokens = checkpoint(blk, tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = blk(tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Heads
# ═══════════════════════════════════════════════════════════════════════════════

def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(pos_grid, embed_dim, omega_0=100):
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(H, W, embed_dim)


def create_uv_grid(width, height, aspect_ratio=None, dtype=None, device=None):
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    x_coords = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width,
                               steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height,
                               steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def inverse_log_transform(y):
    return torch.sign(y) * torch.expm1(torch.abs(y))


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    fmap = out.permute(0, 2, 3, 1)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]
    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pts3d = (xyz / d) * torch.expm1(d)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")
    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")
    return pts3d, conf_out


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]
    def base_act(x, act):
        if act == "linear": return x
        if act == "inv_log": return inverse_log_transform(x)
        if act == "exp": return torch.exp(x)
        if act == "relu": return F.relu(x)
        raise ValueError(f"Unknown act_type: {act}")
    return torch.cat([base_act(T, trans_act), base_act(quat, quat_act), base_act(fl, fl_act)], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class CameraHead(nn.Module):
    def __init__(self, dim_in=2048, trunk_depth=4, pose_encoding_type="absT_quaR_FoV",
                 num_heads=16, mlp_ratio=4, init_values=0.01,
                 trans_act="linear", quat_act="linear", fl_act="relu"):
        super().__init__()
        if pose_encoding_type != "absT_quaR_FoV":
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        self.target_dim = 9
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk = nn.Sequential(*[
            Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
            for _ in range(trunk_depth)
        ])
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2,
                               out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list, num_iterations=4):
        tokens = aggregated_tokens_list[-1]
        pose_tokens = self.token_norm(tokens[:, :, 0])
        pred_pose_enc_list = []
        pred_pose_enc = None
        B, S, C = pose_tokens.shape
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                module_input = self.embed_pose(pred_pose_enc.detach())
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pt_mod = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pt_mod = pt_mod + pose_tokens
            pt_mod = self.trunk(pt_mod)
            delta = self.pose_branch(self.trunk_norm(pt_mod))
            pred_pose_enc = delta if pred_pose_enc is None else pred_pose_enc + delta
            pred_pose_enc_list.append(activate_pose(pred_pose_enc, self.trans_act,
                                                     self.quat_act, self.fl_act))
        return pred_pose_enc_list


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape2 = out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, 3, 1, 1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, 3, 1, 1, bias=False, groups=groups)
    return scratch


def custom_interpolate(x, size=None, scale_factor=None, mode="bilinear", align_corners=True):
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    if size[0] * size[1] * x.shape[0] * x.shape[1] > INT_MAX:
        chunks = torch.chunk(x, chunks=(size[0] * size[1] * x.shape[0] * x.shape[1] // INT_MAX) + 1, dim=0)
        return torch.cat([nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners)
                          for c in chunks], dim=0).contiguous()
    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.norm1 = self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False,
                 align_corners=True, size=None, has_residual=True, groups=1):
        super().__init__()
        self.align_corners = align_corners
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))
        output = self.resConfUnit2(output)
        modifier = ({"size": size} if size is not None else
                    {"size": self.size} if self.size is not None else
                    {"scale_factor": 2})
        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def _make_fusion_block(features, size=None, has_residual=True, groups=1):
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), deconv=False, bn=False,
                              expand=False, align_corners=True, size=size,
                              has_residual=has_residual, groups=groups)


class DPTHead(nn.Module):
    def __init__(self, dim_in, patch_size=14, output_dim=4, activation="inv_log",
                 conf_activation="expp1", features=256,
                 out_channels=None, intermediate_layer_idx=None,
                 pos_embed=True, feature_only=False, down_ratio=1):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, 1, 1, 0) for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4, 0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2, 0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1),
        ])
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        h1, h2 = features, 32
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(h1, h1, 3, 1, 1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(h1, h1 // 2, 3, 1, 1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(h1 // 2, h2, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(h2, output_dim, 1, 1, 0),
            )

    def forward(self, aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=8):
        B, S, _, H, W = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        all_preds, all_conf = [], []
        for fs in range(0, S, frames_chunk_size):
            fe = min(fs + frames_chunk_size, S)
            preds_c, conf_c = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, fs, fe)
            all_preds.append(preds_c)
            all_conf.append(conf_c)
        return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(self, aggregated_tokens_list, images, patch_start_idx,
                      frames_start_idx=None, frames_end_idx=None):
        if frames_start_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = x.contiguous().view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
        out = self._scratch_forward(out)
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio),
             int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)
        if self.feature_only:
            return out.view(B, S, *out.shape[1:])
        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        return preds.view(B, S, *preds.shape[1:]), conf.view(B, S, *conf.shape[1:])

    def _apply_pos_embed(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pos_embed = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def _scratch_forward(self, features):
        l1, l2, l3, l4 = features
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return self.scratch.output_conv1(out)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  VGGT main model
# ═══════════════════════════════════════════════════════════════════════════════

class _VGGT(nn.Module):
    """Internal VGGT architecture (no HuggingFace mixin)."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_depth=True, enable_point=True):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (DPTHead(dim_in=2 * embed_dim, output_dim=4,
                                   activation="inv_log", conf_activation="expp1")
                           if enable_point else None)
        self.depth_head = (DPTHead(dim_in=2 * embed_dim, output_dim=2,
                                   activation="exp", conf_activation="expp1")
                           if enable_depth else None)

    def forward(self, images):
        """
        Args:
            images: (B, S, 3, H, W) in range [0, 1].
        Returns:
            dict with 'depth' key: (B, S, H, W, 1).
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        with torch.amp.autocast('cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
            if self.depth_head is not None:
                depth_pred, depth_conf = self.depth_head(
                    aggregated_tokens_list, images, patch_start_idx)
                predictions["depth"] = depth_pred
                predictions["depth_conf"] = depth_conf
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images, patch_start_idx)
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
        predictions["images"] = images
        return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Library wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class VGGTModel(BaseDepthModel):
    """VGGT monocular depth estimation wrapper.

    Adapts the multi-frame VGGT architecture for single-image depth estimation
    by treating each image as a sequence of length S=1.

    Input:  ``pixel_values`` — ``(B, 3, H, W)`` tensor with values in ``[0, 1]``.
    Output: Depth tensor ``(B, H, W)`` in metric units (metres).
    """

    config_class = VGGTConfig

    def __init__(self, config: VGGTConfig):
        super().__init__(config)
        self._net = None  # set by _load_pretrained_weights

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        device = next(self._net.parameters()).device
        x = pixel_values.to(device)

        # Pad to patch-size multiple
        patch = self.config.patch_size
        H, W = x.shape[-2], x.shape[-1]
        H_pad = math.ceil(H / patch) * patch
        W_pad = math.ceil(W / patch) * patch
        if H_pad != H or W_pad != W:
            x = F.interpolate(x, size=(H_pad, W_pad), mode="bilinear", align_corners=False)

        # (B, 3, H, W) → (B, 1, 3, H, W)
        images = x.unsqueeze(1)
        preds = self._net(images)

        # depth: (B, S=1, H, W, 1) → (B, H, W)
        depth = preds["depth"][:, 0, :, :, 0]

        if depth.shape[-2] != H or depth.shape[-1] != W:
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(1)
        return depth

    @classmethod
    def _load_pretrained_weights(cls, model_id: str, device: str = "cpu", **kwargs) -> "VGGTModel":
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        config = VGGTConfig.from_variant(model_id)
        torch_device = torch.device(device)

        net = _VGGT(img_size=config.img_size, patch_size=config.patch_size,
                    embed_dim=config.embed_dim)

        ckpt_path = hf_hub_download(repo_id=config.hub_repo_id, filename="model.safetensors")
        state_dict = load_file(ckpt_path, device=device)
        net.load_state_dict(state_dict, strict=False)
        net = net.to(torch_device).eval()

        model = cls(config)
        model._net = net
        return model
