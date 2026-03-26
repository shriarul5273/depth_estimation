"""
Depth Anything v3 — Single-file model implementation.

Architecture: DA3 DinoV2 backbone + DualDPT or DPT decoder head.

All components (layer primitives, RoPE, backbone, DPT heads) are inlined
here following the single-file policy used by other models in this library.

Ported from depth_anything_3 (ByteDance, Apache 2.0) and
Meta's DINOv2 (Meta Platforms, Apache 2.0 / CC-BY-NC 4.0).
Weights loaded as safetensors checkpoints from HuggingFace Hub.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from ...modeling_utils import BaseDepthModel
from .configuration_depth_anything_v3 import DepthAnythingV3Config, _V3_VARIANT_MAP

logger = logging.getLogger(__name__)


# ============================================================================ #
#  Layer primitives — PatchEmbed                                               #
# ============================================================================ #


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)"""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
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
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


# ============================================================================ #
#  Layer primitives — DropPath                                                 #
# ============================================================================ #


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


# ============================================================================ #
#  Layer primitives — LayerScale                                               #
# ============================================================================ #


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def extra_repr(self) -> str:
        return f"{self.dim}, init_values={self.init_values}, inplace={self.inplace}"


# ============================================================================ #
#  Layer primitives — Mlp                                                      #
# ============================================================================ #


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
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


# ============================================================================ #
#  Layer primitives — SwiGLU FFN                                               #
# ============================================================================ #


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
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


try:
    from xformers.ops import SwiGLU

    XFORMERS_AVAILABLE = True
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


# ============================================================================ #
#  Layer primitives — Attention                                                #
# ============================================================================ #


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos=None, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(
                    (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
                    if attn_mask is not None
                    else None
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================ #
#  Layer primitives — Block                                                    #
# ============================================================================ #


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
    pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]
    if pos is not None:
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)
    x_flat = x.flatten(1)
    residual = residual.flatten(1)
    residual_scale_factor = b / sample_subset_size
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        ln_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim, eps=ln_eps)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=ln_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor, pos=None, attn_mask=None) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor, pos=None, attn_mask=None) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                pos=pos,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + ffn_residual_func(x)
        return x


# ============================================================================ #
#  Layer primitives — RoPE                                                     #
# ============================================================================ #


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions
        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert (
            positions.ndim == 3 and positions.shape[-1] == 2
        ), "Positions must have shape (batch_size, n_tokens, 2)"
        feature_dim = tokens.size(-1) // 2
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical_features, horizontal_features), dim=-1)


# ============================================================================ #
#  Reference view selection                                                    #
# ============================================================================ #


THRESH_FOR_REF_SELECTION = 3

RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]


def select_reference_view(
    x: torch.Tensor,
    strategy: RefViewStrategy = "saddle_balanced",
) -> torch.Tensor:
    B, S, N, C = x.shape
    if S <= 1:
        return torch.zeros(B, dtype=torch.long, device=x.device)
    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    elif strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)
    img_class_feat = x[:, :, 0] / x[:, :, 0].norm(dim=-1, keepdim=True)
    if strategy == "saddle_balanced":
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        sim_score = sim_no_diag.sum(dim=-1) / (S - 1)
        feat_norm = x[:, :, 0].norm(dim=-1)
        feat_var = img_class_feat.var(dim=-1)

        def normalize_metric(metric):
            min_val = metric.min(dim=1, keepdim=True).values
            max_val = metric.max(dim=1, keepdim=True).values
            return (metric - min_val) / (max_val - min_val + 1e-8)

        sim_score_norm = normalize_metric(sim_score)
        norm_norm = normalize_metric(feat_norm)
        var_norm = normalize_metric(feat_var)
        balance_score = (
            (sim_score_norm - 0.5).abs()
            + (norm_norm - 0.5).abs()
            + (var_norm - 0.5).abs()
        )
        b_idx = balance_score.argmin(dim=1)
    elif strategy == "saddle_sim_range":
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        sim_max = sim_no_diag.max(dim=-1).values
        sim_min = sim_no_diag.min(dim=-1).values
        sim_range = sim_max - sim_min
        b_idx = sim_range.argmax(dim=1)
    else:
        raise ValueError(
            f"Unknown reference view selection strategy: {strategy}. "
            f"Must be one of: 'first', 'middle', 'saddle_balanced', 'saddle_sim_range'"
        )
    return b_idx


def reorder_by_reference(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    B, S = x.shape[0], x.shape[1]
    if S <= 1:
        return x
    positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
    b_idx_expanded = b_idx.unsqueeze(1)
    reorder_indices = positions.clone()
    reorder_indices = torch.where(
        (positions > 0) & (positions <= b_idx_expanded),
        positions - 1,
        positions,
    )
    reorder_indices[:, 0] = b_idx
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
    return x[batch_indices, reorder_indices]


def restore_original_order(x: torch.Tensor, b_idx: torch.Tensor) -> torch.Tensor:
    B, S = x.shape[0], x.shape[1]
    if S <= 1:
        return x
    target_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
    b_idx_expanded = b_idx.unsqueeze(1)
    restore_indices = torch.where(
        target_positions < b_idx_expanded,
        target_positions + 1,
        target_positions,
    )
    restore_indices = torch.scatter(
        restore_indices,
        dim=1,
        index=b_idx_expanded,
        src=torch.zeros_like(b_idx_expanded),
    )
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
    return x[batch_indices, restore_indices]


# ============================================================================ #
#  DINOv2 backbone                                                             #
# ============================================================================ #


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
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
        init_values=1.0,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        rope_freq=100,
        plus_cam_token=False,
        cat_token=True,
    ):
        super().__init__()
        self.patch_start_idx = 1
        norm_layer = nn.LayerNorm
        self.num_features = self.embed_dim = embed_dim
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        if self.rope_start != -1:
            self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
            self.position_getter = PositionGetter() if self.rope is not None else None
        else:
            self.rope = None

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=i >= qknorm_start if qknorm_start != -1 else False,
                rope=self.rope if i >= rope_start and rope_start != -1 else None,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)

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
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_cls_token(self, B, S):
        cls_token = self.cls_token.expand(B, S, -1)
        cls_token = cls_token.reshape(B * S, -1, self.embed_dim)
        return cls_token

    def prepare_tokens_with_masks(self, x, masks=None, cls_token=None, **kwargs):
        B, S, nc, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        cls_token = self.prepare_cls_token(B, S)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        x = rearrange(x, "(b s) n c -> b s n c", b=B, s=S)
        return x

    def _prepare_rope(self, B, S, H, W, device):
        pos = None
        pos_nodiff = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)
            pos = rearrange(pos, "(b s) n c -> b s n c", b=B)
            pos_nodiff = torch.zeros_like(pos).to(pos.dtype)
            if self.patch_start_idx > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device).to(pos.dtype)
                pos_special = rearrange(pos_special, "(b s) n c -> b s n c", b=B)
                pos = torch.cat([pos_special, pos], dim=2)
                pos_nodiff = pos_nodiff + 1
                pos_nodiff = torch.cat([pos_special, pos_nodiff], dim=2)
        return pos, pos_nodiff

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if (
                self.alt_start != -1
                and (i == self.alt_start - 1)
                and x.shape[1] >= THRESH_FOR_REF_SELECTION
                and kwargs.get("cam_token", None) is None
            ):
                strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
                logger.info(f"Selecting reference view using strategy: {strategy}")
                b_idx = select_reference_view(x, strategy=strategy)
                x = reorder_by_reference(x, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    logger.info("Using camera conditions provided by the user")
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and "b_idx" in locals():
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def process_attention(self, x, block, attn_type="global", pos=None, attn_mask=None):
        b, s, n = x.shape[:3]
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
        x = block(x, pos=pos, attn_mask=attn_mask)
        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        return x

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        export_feat_layers: List[int] = [],
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        outputs, aux_outputs = self._get_intermediate_layers_not_chunked(
            x, n, export_feat_layers=export_feat_layers, **kwargs
        )
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.embed_dim:
            outputs = [self.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
            outputs = [
                torch.cat(
                    [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        aux_outputs = [self.norm(out) for out in aux_outputs]
        outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
        aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]
        return tuple(zip(outputs, camera_tokens)), aux_outputs


def vit_small(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_base(patch_size=16, num_register_tokens=0, depth=12, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(patch_size=16, num_register_tokens=0, depth=24, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_giant2(patch_size=16, num_register_tokens=0, depth=40, **kwargs):
    """Close to ViT-giant, with embed-dim 1536 and 24 heads."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=depth,
        num_heads=24,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


class DinoV2(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[name]
        ffn_layer = "swiglufused" if name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )

    def forward(self, x, **kwargs):
        return self.pretrained.get_intermediate_layers(x, self.out_layers, **kwargs)


# ============================================================================ #
#  DPT decoder — utilities                                                     #
# ============================================================================ #


class Permute(nn.Module):
    """nn.Module wrapper around Tensor.permute for use inside nn.Sequential."""

    dims: Tuple[int, ...]

    def __init__(self, dims: Tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1).float()


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag_factor = (aspect_ratio ** 2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def custom_interpolate(
    x: torch.Tensor,
    size: Union[Tuple[int, int], None] = None,
    scale_factor: Union[float, None] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Safe interpolation that avoids INT_MAX overflow."""
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [
            nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()
    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


# ============================================================================ #
#  DPT decoder — building blocks                                               #
# ============================================================================ #


class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for fusion."""

    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Top-down fusion block: (optional) residual merge + upsampling + 1x1 contraction."""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: Tuple[int, int] = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)
        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))
        y = self.resConfUnit2(y)
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}
        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        y = self.out_conv(y)
        return y


def _make_fusion_block(
    features: int,
    size: Tuple[int, int] = None,
    has_residual: bool = True,
    groups: int = 1,
    inplace: bool = False,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(
    in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False
) -> nn.Module:
    scratch = nn.Module()
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)
    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch


# ============================================================================ #
#  DPT head                                                                    #
# ============================================================================ #


class DPT(nn.Module):
    """DPT dense prediction head (main + optional sky head)."""

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        head_name: str = "depth",
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",
        use_ln_for_heads: bool = False,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_main = head_name
        self.sky_name = sky_name
        self.out_dim = output_dim
        self.has_conf = output_dim > 1
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        if norm_type == "layer":
            self.norm = nn.LayerNorm(dim_in)
        elif norm_type == "idt":
            self.norm = nn.Identity()
        else:
            raise Exception(f"Unknown norm_type {norm_type}, should be 'layer' or 'idt'.")

        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(list(out_channels), features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet2 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet3 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False, inplace=fusion_block_inplace)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        ln_seq = (
            [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))]
            if use_ln_for_heads
            else []
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            *ln_seq,
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
        **kwargs,
    ) -> Dict:
        from addict import Dict as AddictDict
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        extra_kwargs = {}
        if "images" in kwargs:
            extra_kwargs.update({"images": rearrange(kwargs["images"], "B S ... -> (B S) ...")})
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx, **extra_kwargs)
            out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return AddictDict(out_dict)
        out_dicts = []
        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            kw = {}
            if "images" in extra_kwargs:
                kw.update({"images": extra_kwargs["images"][s0:s1]})
            out_dicts.append(
                self._forward_impl([f[s0:s1] for f in feats], H, W, patch_start_idx, **kw)
            )
        out_dict = {k: torch.cat([od[k] for od in out_dicts], dim=0) for k in out_dicts[0].keys()}
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return AddictDict(out_dict)

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)
        fused = self._fuse(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = self.scratch.output_conv1(fused)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)
        feat = fused
        main_logits = self.scratch.output_conv2(feat)
        outs: Dict[str, torch.Tensor] = {}
        if self.has_conf:
            fmap = main_logits.permute(0, 2, 3, 1)
            pred = self._apply_activation_single(fmap[..., :-1], self.activation)
            conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
            outs[self.head_main] = pred.squeeze(1)
            outs[f"{self.head_main}_conf"] = conf.squeeze(1)
        else:
            outs[self.head_main] = self._apply_activation_single(main_logits, self.activation).squeeze(1)
        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(feat)
            outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)
        return outs

    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x

    def _apply_sky_activation(self, x: torch.Tensor) -> torch.Tensor:
        act = self.sky_activation.lower() if isinstance(self.sky_activation, str) else self.sky_activation
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "relu":
            return torch.relu(x)
        return x

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe


# ============================================================================ #
#  DualDPT head                                                                #
# ============================================================================ #


class DualDPT(nn.Module):
    """Dual-head DPT with independent main and auxiliary fusion chains."""

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num
        self.head_main, self.head_aux = head_names
        self.intermediate_layer_idx: Tuple[int, int, int, int] = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # Auxiliary fusion chain
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1_aux = nn.ModuleList(
            [self._make_aux_out1_block(head_features_1) for _ in range(self.aux_levels)]
        )
        use_ln = True
        ln_seq = (
            [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))]
            if use_ln
            else []
        )
        self.scratch.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                    *ln_seq,
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features_2, 7, kernel_size=1, stride=1, padding=0),
                )
                for _ in range(self.aux_levels)
            ]
        )

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        from addict import Dict as AddictDict
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return AddictDict(out_dict)
        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
            out_dicts.append(
                self._forward_impl([feat[s0:s1] for feat in feats], H, W, patch_start_idx)
            )
        out_dict = {
            k: torch.cat([od[k] for od in out_dicts], dim=0) for k in out_dicts[0].keys()
        }
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return AddictDict(out_dict)

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)
        fused_main, fused_aux_pyr = self._fuse(resized_feats)
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)
        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = self._add_pos_embed(last_aux, W, H)
        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_last = last_aux_logits.permute(0, 2, 3, 1)
        aux_pred = self._apply_activation_single(fmap_last[..., :-1], "linear")
        aux_conf = self._apply_activation_single(fmap_last[..., -1], self.conf_activation)
        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,
            f"{self.head_aux}_conf": aux_conf,
        }

    def _fuse(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[torch.Tensor] = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)
        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)
        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](aux) for i, aux in enumerate(aux_list)]
        return out, aux_list

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _make_aux_out1_block(self, in_ch: int) -> nn.Sequential:
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 3:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 1:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1))
        raise ValueError(f"aux_out1_conv_num {self.aux_out1_conv_num} not supported")

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        return x


# ============================================================================ #
#  Camera decoder                                                               #
# ============================================================================ #


class CameraDec(nn.Module):
    """MLP that maps cls-token features → compact pose encoding (T, quat, FOV)."""

    def __init__(self, dim_in: int = 1536) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(),
            nn.Linear(dim_in, dim_in), nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim_in, 3)
        self.fc_qvec = nn.Linear(dim_in, 4)
        self.fc_fov = nn.Sequential(nn.Linear(dim_in, 2), nn.ReLU())

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Args: feat (B, S, dim_in). Returns pose_enc (B, S, 9)."""
        B, S = feat.shape[:2]
        feat = feat.reshape(B * S, -1).float()
        feat = self.backbone(feat)
        out_t = self.fc_t(feat).reshape(B, S, 3)
        out_qvec = self.fc_qvec(feat).reshape(B, S, 4)
        out_fov = self.fc_fov(feat).reshape(B, S, 2)
        return torch.cat([out_t, out_qvec, out_fov], dim=-1)


def _fov_to_intrinsics(pose_enc: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Derive pixel-space intrinsics from pose encoding's FOV fields."""
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), min=1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), min=1e-6)
    B, S = fov_h.shape
    K = torch.zeros(B, S, 3, 3, device=pose_enc.device, dtype=fx.dtype)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = W / 2.0
    K[..., 1, 2] = H / 2.0
    K[..., 2, 2] = 1.0
    return K


# ============================================================================ #
#  Composed model                                                               #
# ============================================================================ #


class _DepthAnythingV3Net(nn.Module):
    """Internal composed model: DA3 DinoV2 backbone + DualDPT or DPT head."""

    def __init__(self, config: DepthAnythingV3Config):
        super().__init__()
        self.backbone = DinoV2(
            name=config.dinov2_name,
            out_layers=config.out_layers,
            alt_start=config.alt_start,
            qknorm_start=config.qknorm_start,
            rope_start=config.rope_start,
            cat_token=config.cat_token,
        )
        if config.head_cls == "DualDPT":
            self.head = DualDPT(
                dim_in=config.head_dim_in,
                features=config.features,
                out_channels=config.out_channels,
                output_dim=config.output_dim,
            )
        else:
            self.head = DPT(
                dim_in=config.head_dim_in,
                features=config.features,
                out_channels=config.out_channels,
                output_dim=config.output_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        x = x.unsqueeze(1)
        feats, _ = self.backbone(x, export_feat_layers=[])
        output = self.head(feats, H, W, patch_start_idx=0)
        return output["depth"].squeeze(1)

    def _forward_full(self, x: torch.Tensor) -> Dict:
        H, W = x.shape[-2], x.shape[-1]
        xin = x.unsqueeze(1)
        feats, _ = self.backbone(xin, export_feat_layers=[])
        output = self.head(feats, H, W, patch_start_idx=0)
        return dict(output)


class _DepthAnythingV3NetWithCamDec(_DepthAnythingV3Net):
    """Like _DepthAnythingV3Net but also loads and runs a CameraDec head."""

    def __init__(self, config: DepthAnythingV3Config):
        super().__init__(config)
        self.cam_dec = CameraDec(dim_in=config.head_dim_in)

    def _forward_full(self, x: torch.Tensor) -> Dict:
        H, W = x.shape[-2], x.shape[-1]
        xin = x.unsqueeze(1)
        feats, _ = self.backbone(xin, export_feat_layers=[])
        output = dict(self.head(feats, H, W, patch_start_idx=0))
        cam_token = feats[-1][1]
        pose_enc = self.cam_dec(cam_token)
        output["intrinsics"] = _fov_to_intrinsics(pose_enc, H, W)
        return output


# ============================================================================ #
#  Public model classes                                                        #
# ============================================================================ #


class DepthAnythingV3Model(BaseDepthModel):
    """Depth Anything v3 model.

    Usage::

        model = DepthAnythingV3Model.from_pretrained("depth-anything-v3-large")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = DepthAnythingV3Config

    def __init__(self, config: DepthAnythingV3Config):
        super().__init__(config)
        self.net = _DepthAnythingV3Net(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values)

    def _backbone_module(self):
        """Return the DinoV2 backbone (_DepthAnythingV3Net.backbone)."""
        return self.net.backbone

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthAnythingV3Model":
        """Load v3 weights from HuggingFace Hub safetensors checkpoint."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        backbone = _V3_VARIANT_MAP.get(model_id)
        if backbone is None:
            raise ValueError(
                f"Cannot resolve backbone from '{model_id}'. "
                f"Use one of: {list(_V3_VARIANT_MAP.keys())}"
            )

        config = DepthAnythingV3Config(backbone=backbone)
        model = cls(config)

        filepath = hf_hub_download(
            repo_id=config.hub_repo_id,
            filename=config.checkpoint_filename,
            repo_type="model",
        )
        logger.info(f"Downloaded v3 checkpoint: {filepath}")

        state_dict = load_file(filepath, device=device)
        remapped = {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        model.net.load_state_dict(remapped, strict=False)
        model = model.to(device)

        logger.info(f"Loaded Depth Anything v3 ({backbone}) from {config.hub_repo_id}")
        return model


class DepthAnythingV3NestedModel(BaseDepthModel):
    """Depth Anything v3 Nested model: giant (multi-view) + metric_large.

    Usage::

        model = DepthAnythingV3NestedModel.from_pretrained(
            "depth-anything-v3-nested-giant-large"
        )
        depth = model(pixel_values)  # (B, H, W) metric-scale depth tensor
    """

    config_class = DepthAnythingV3Config

    def __init__(self, config: DepthAnythingV3Config):
        super().__init__(config)
        giant_config = DepthAnythingV3Config(backbone="giant")
        metric_config = DepthAnythingV3Config(backbone="metric_large")
        self.da3 = _DepthAnythingV3NetWithCamDec(giant_config)
        self.da3_metric = _DepthAnythingV3Net(metric_config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        giant_out = self.da3._forward_full(pixel_values)
        depth = giant_out["depth"]
        depth_conf = giant_out["depth_conf"]
        intrinsics = giant_out.get("intrinsics")

        metric_out = self.da3_metric._forward_full(pixel_values)
        metric_depth = metric_out["depth"]
        sky = metric_out.get("sky")

        if intrinsics is not None:
            focal = (intrinsics[..., 0, 0] + intrinsics[..., 1, 1]) / 2.0
            metric_depth = metric_depth * focal[..., None, None] / 300.0

        if sky is not None:
            non_sky = sky < 0.3
            n_non_sky = non_sky.sum().item()
            n_sky = (~non_sky).sum().item()

            if n_non_sky > 10 and n_sky > 10:
                conf_ns = depth_conf[non_sky]
                if conf_ns.numel() > 100_000:
                    idx = torch.randperm(conf_ns.numel(), device=conf_ns.device)[:100_000]
                    conf_ns = conf_ns[idx]
                med_conf = torch.quantile(conf_ns, 0.5)

                align_mask = (
                    (depth_conf >= med_conf)
                    & non_sky
                    & (metric_depth > 1e-2)
                    & (depth > 1e-3)
                )

                if align_mask.sum() > 10:
                    a = metric_depth[align_mask]
                    b = depth[align_mask]
                    scale = torch.dot(a, b) / torch.dot(b, b).clamp_min(1e-12)
                    depth = depth * scale

                non_sky_d = depth[non_sky]
                if non_sky_d.numel() > 100_000:
                    idx = torch.randint(0, non_sky_d.numel(), (100_000,), device=non_sky_d.device)
                    non_sky_d = non_sky_d[idx]
                sky_fill = min(torch.quantile(non_sky_d, 0.99).item(), 200.0)
                depth = depth.clone()
                depth[~non_sky] = sky_fill

        return depth.squeeze(1)

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthAnythingV3NestedModel":
        """Load weights for both sub-models from their respective HF Hub repos."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config = DepthAnythingV3Config(backbone="giant")
        model = cls(config)

        def _load_submodel(backbone_name: str) -> Dict:
            sub_cfg = DepthAnythingV3Config(backbone=backbone_name)
            path = hf_hub_download(
                repo_id=sub_cfg.hub_repo_id,
                filename=sub_cfg.checkpoint_filename,
                repo_type="model",
            )
            logger.info(f"Downloaded v3 checkpoint ({backbone_name}): {path}")
            sd = load_file(path, device="cpu")
            return {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}

        giant_sd = _load_submodel("giant")
        metric_sd = _load_submodel("metric_large")

        combined = {}
        combined.update({f"da3.{k}": v for k, v in giant_sd.items()})
        combined.update({f"da3_metric.{k}": v for k, v in metric_sd.items()})
        del giant_sd, metric_sd

        model.load_state_dict(combined, strict=False)
        del combined
        model = model.to(device)

        logger.info("Loaded Depth Anything v3 Nested (giant + metric_large)")
        return model
