"""
Pixel-Perfect Depth — Single-file model implementation.

Diffusion-based monocular depth estimation that combines:
  - A DINOv2 ViT-L semantics encoder (Depth Anything V2 backbone)
  - A DiT (Diffusion Transformer) with 2D Rotary Position Embeddings
  - A rectified-flow (linear schedule) Euler sampler

Weights:
  - Semantics encoder: depth-anything/Depth-Anything-V2-Large (depth_anything_v2_vitl.pth)
  - Full model (DiT + encoder): gangweix/Pixel-Perfect-Depth (ppd.pth)

Reference implementation: https://github.com/gangweix/pixel-perfect-depth
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_ppd import PixelPerfectDepthConfig
# Reuse the already-inlined DINOv2 backbone from depth_anything_v2
from ..depth_anything_v2.modeling_depth_anything_v2 import (
    DinoVisionTransformer,
    build_dinov2_backbone,
)

logger = logging.getLogger(__name__)


# ============================================================================ #
#  Inlined components — Mlp
# ============================================================================ #


class _Mlp(nn.Module):
    """MLP used in DiT blocks.

    Note: ``act_layer`` must be passed as an **instantiated** module, not a class.
    This matches the PPD source where ``approx_gelu = nn.GELU(approximate="tanh")``
    is created and then passed directly.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer if act_layer is not None else nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ============================================================================ #
#  Inlined components — PatchEmbed
# ============================================================================ #


class _PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D).

    Ported from ppd/models/patch_embed.py.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        flatten_embedding: bool = True,
    ):
        super().__init__()
        image_HW = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_HW = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = nn.Identity()

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


# ============================================================================ #
#  Inlined components — 2D Rotary Position Embedding
# ============================================================================ #


class _PositionGetter:
    """Generates and caches 2D spatial positions for patches.

    Ported from ppd/models/rope.py.
    """

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[(height, width)] = positions
        cached = self.position_cache[(height, width)]
        return cached.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class _RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding.

    Ported from ppd/models/rope.py.
    """

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
            self.frequency_cache[cache_key] = (angles.cos().to(dtype), angles.sin().to(dtype))
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
        assert tokens.size(-1) % 2 == 0
        assert positions.ndim == 3 and positions.shape[-1] == 2

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
#  Inlined components — Attention with optional RoPE
# ============================================================================ #


class _Attention(nn.Module):
    """Multi-head attention with optional 2D RoPE.

    Ported from ppd/models/attention.py.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope: Optional[_RotaryPositionEmbedding2D] = None,
        fused_attn: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
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
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
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
#  Inlined components — DiT building blocks
# ============================================================================ #


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Ported from ppd/models/dit.py.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class _DiTBlock(nn.Module):
    """DiT block with adaptive layer norm (adaLN-Zero) and optional RoPE.

    Ported from ppd/models/dit.py.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rope: Optional[_RotaryPositionEmbedding2D] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, rope=rope
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = nn.GELU(approximate="tanh")
        self.mlp = _Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), pos=pos
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class _FinalLayer(nn.Module):
    """Final layer of the DiT.

    Ported from ppd/models/dit.py.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class _DiT(nn.Module):
    """Cascade DiT with 2D RoPE for depth estimation.

    Architecture:
      - PatchEmbed (patch_size=16) maps (N, 4, H, W) -> (N, T, D)  where T = (H//16)*(W//16)
      - Blocks 0-11 operate at T resolution with pos0
      - After block 11: semantics fusion + 2x upsample -> T' = (H//8)*(W//8)
      - Blocks 12-23 operate at T' resolution with pos1
      - FinalLayer (patch_size=8) + unpatchify -> (N, 1, H, W)

    Ported from ppd/models/dit.py (RoPE variant).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.rope = _RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = _PositionGetter()

        self.x_embedder = _PatchEmbed(in_chans=in_channels, embed_dim=hidden_size)
        self.t_embedder = _TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList(
            [
                _DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, rope=self.rope)
                for _ in range(depth)
            ]
        )

        self.proj_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
        )

        self.final_layer = _FinalLayer(hidden_size, 8, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reconstruct spatial map from token sequence.

        Uses patch_size=8 to convert (N, T', D') -> (N, 1, H, W).
        """
        c = self.out_channels
        p = 8
        h = height // p
        w = width // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(
        self,
        x: torch.Tensor,
        semantics: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, 4, H, W) concatenated [latent, image_condition].
            semantics: (N, T, hidden_size) DINOv2 patch tokens.
            timestep: (N,) or scalar diffusion timestep.

        Returns:
            (N, 1, H, W) predicted velocity (denoised depth).
        """
        N, C, H, W = x.shape
        if len(timestep.shape) == 0:
            timestep = timestep[None]

        pos0 = self.position_getter(N, H // 16, W // 16, device=x.device)
        pos1 = self.position_getter(N, H // 8, W // 8, device=x.device)

        x = self.x_embedder(x)  # (N, T, D)
        N, T, D = x.shape
        t = self.t_embedder(timestep)  # (N, D)

        for i, block in enumerate(self.blocks):
            if i < 12:
                x = block(x, t, pos0)
            else:
                x = block(x, t, pos1)

            if i == 11:
                semantics = F.normalize(semantics, dim=-1)
                x = self.proj_fusion(torch.cat([x, semantics], dim=-1))  # (N, T, D*4)
                p = 16
                x = x.reshape(shape=(N, H // p, W // p, 2, 2, D))
                x = torch.einsum("nhwpqc->nchpwq", x)
                x = x.reshape(shape=(N, D, (H // p) * 2, (W // p) * 2))
                x = x.flatten(2).transpose(1, 2)  # (N, T', D) where T' = (H//8)*(W//8)

        x = self.final_layer(x, t)          # (N, T', p*p*C)
        x = self.unpatchify(x, height=H, width=W)  # (N, 1, H, W)
        return x


# ============================================================================ #
#  Inlined components — Diffusion schedule + sampler
# ============================================================================ #


class _LinearSchedule:
    """Linear interpolation schedule (rectified flow / flow matching).

    x_t = (1 - t/T) * x_0 + (t/T) * x_T

    Ported from ppd/utils/schedule.py.
    """

    def __init__(self, T: Union[int, float] = 1000):
        self.T = T

    def forward(
        self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = t[(...,) + (None,) * (x_0.ndim - t.ndim)] if t.ndim < x_0.ndim else t
        return (1 - t / self.T) * x_0 + (t / self.T) * x_T

    def convert_from_pred(
        self,
        pred: torch.Tensor,
        pred_type: str,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert velocity prediction to x_0 and x_T."""
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        A_t = 1 - t / self.T
        B_t = t / self.T
        pred_x_0 = x_t - B_t * pred
        pred_x_T = x_t + A_t * pred
        return pred_x_0, pred_x_T


class _Timesteps:
    """Discretized sampling timesteps.

    Ported from ppd/utils/timesteps.py.
    """

    def __init__(self, T: int, steps: int, device: torch.device = "cpu"):
        self.T = T
        self.timesteps = torch.arange(T, -1, -(T + 1) / steps, device=device).round().int()

    def __len__(self) -> int:
        return len(self.timesteps)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> torch.Tensor:
        return self.timesteps[idx]

    def index(self, t: torch.Tensor) -> torch.Tensor:
        i, j = t.reshape(-1, 1).eq(self.timesteps).nonzero(as_tuple=True)
        idx = torch.full_like(t, fill_value=-1, dtype=torch.int)
        idx.view(-1)[i] = j.int()
        return idx


class _EulerSampler:
    """Euler ODE solver for the linear schedule.

    Ported from ppd/utils/sampler.py.
    """

    def __init__(
        self,
        schedule: _LinearSchedule,
        timesteps: _Timesteps,
        prediction_type: str = "velocity",
    ):
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type

    def step(
        self, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return self._step_to(pred, x_t, t, self._get_next_timestep(t))

    def _step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        t = t[(...,) + (None,) * (x_t.ndim - t.ndim)] if t.ndim < x_t.ndim else t
        s = s[(...,) + (None,) * (x_t.ndim - s.ndim)] if s.ndim < x_t.ndim else s
        T = self.schedule.T
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(
            pred, self.prediction_type, x_t, t
        )
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        return pred_x_s

    def _get_next_timestep(self, t: torch.Tensor) -> torch.Tensor:
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1
        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        return s.where(next_idx < steps, torch.tensor(-1, device=t.device, dtype=s.dtype))


# ============================================================================ #
#  Semantics encoder — thin DINOv2 ViT-L wrapper
# ============================================================================ #


class _PPDSemanticsEncoder(nn.Module):
    """DINOv2 ViT-L used purely as a semantic feature extractor.

    Replicates ppd/models/depth_anything_v2/dpt.py ``DepthAnythingV2.forward``:
      - Normalises the [0,1] input image with ImageNet mean/std
      - Resizes to a DINOv2-compatible resolution ((H//16)*14 x (W//16)*14)
      - Returns x_norm_patchtokens of shape (N, T, 1024) where T = (H//16)*(W//16)
    """

    def __init__(self):
        super().__init__()
        self.pretrained: DinoVisionTransformer = build_dinov2_backbone("vitl")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level semantic features.

        Args:
            x: (N, 3, H, W) image tensor in [0, 1] range.

        Returns:
            (N, (H//16)*(W//16), 1024) patch token features.
        """
        ori_h, ori_w = x.shape[-2:]

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        new_h = (ori_h // 16) * 14
        new_w = (ori_w // 16) * 14
        x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)

        return self.pretrained.forward_features(x)["x_norm_patchtokens"]


# ============================================================================ #
#  Main composed network
# ============================================================================ #


class _PixelPerfectDepthNet(nn.Module):
    """Combines semantics encoder + DiT + Euler sampler.

    Matches the top-level ``PixelPerfectDepth`` class from ppd/models/ppd.py.
    Sub-module attribute names (``semantics_encoder``, ``dit``) deliberately
    mirror the original so that checkpoints load correctly with ``strict=False``.
    """

    def __init__(
        self,
        sampling_steps: int = 10,
        dit_in_channels: int = 4,
        dit_out_channels: int = 1,
        dit_hidden_size: int = 1024,
        dit_depth: int = 24,
        dit_num_heads: int = 16,
        dit_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.semantics_encoder = _PPDSemanticsEncoder()
        self.dit = _DiT(
            in_channels=dit_in_channels,
            out_channels=dit_out_channels,
            hidden_size=dit_hidden_size,
            depth=dit_depth,
            num_heads=dit_num_heads,
            mlp_ratio=dit_mlp_ratio,
        )
        self.sampling_steps = sampling_steps
        self._schedule = _LinearSchedule(T=1000)
        self._sampling_timesteps: Optional[_Timesteps] = None
        self._sampler: Optional[_EulerSampler] = None
        self._sampler_device: Optional[torch.device] = None
        self._sampler_steps: int = -1

    def _build_sampler(self, steps: int, device: torch.device):
        self._sampling_timesteps = _Timesteps(T=self._schedule.T, steps=steps, device=device)
        self._sampler = _EulerSampler(
            schedule=self._schedule,
            timesteps=self._sampling_timesteps,
            prediction_type="velocity",
        )
        self._sampler_device = device
        self._sampler_steps = steps

    def _ensure_sampler(self, steps: int, device: torch.device):
        if self._sampler is None or self._sampler_steps != steps or self._sampler_device != device:
            self._build_sampler(steps, device)

    @torch.no_grad()
    def _forward_test(self, image: torch.Tensor, steps: int) -> torch.Tensor:
        """Core diffusion sampling pass.

        Args:
            image: (1, 3, H, W) in [0, 1] range, H and W multiples of 16.
            steps: Number of Euler steps.

        Returns:
            (1, 1, H, W) depth in [0, ~1] range.
        """
        device = image.device
        self._ensure_sampler(steps, device)

        semantics = self.semantics_encoder(image)      # (1, T, 1024)
        cond = image - 0.5                              # (1, 3, H, W)
        latent = torch.randn(
            1, 1, image.shape[2], image.shape[3], device=device
        )

        for timestep in self._sampling_timesteps:
            x_in = torch.cat([latent, cond], dim=1)    # (1, 4, H, W)
            pred = self.dit(x=x_in, semantics=semantics, timestep=timestep)
            latent = self._sampler.step(pred=pred, x_t=latent, t=timestep)

        return latent + 0.5  # shift back to [0, ~1]

    def infer(
        self,
        image: torch.Tensor,
        sampling_steps: Optional[int] = None,
        use_fp16: bool = True,
    ) -> torch.Tensor:
        """Run inference on a single preprocessed image.

        Args:
            image: (1, 3, H, W) in [0, 1] range, H and W multiples of 16.
            sampling_steps: Override sampling steps (uses config default if None).
            use_fp16: Use float16 autocast on CUDA (default True).

        Returns:
            (1, 1, H, W) depth tensor.
        """
        steps = sampling_steps or self.sampling_steps
        device = image.device
        use_autocast = use_fp16 and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            return self._forward_test(image, steps)


# ============================================================================ #
#  Public model class
# ============================================================================ #


class PixelPerfectDepthModel(BaseDepthModel):
    """Pixel-Perfect Depth model.

    Diffusion-based monocular depth estimation conditioned on DINOv2 semantics.

    Usage::

        model = PixelPerfectDepthModel.from_pretrained("pixel-perfect-depth")
        depth = model(pixel_values)  # (B, H, W) tensor, values in [0, 1]
    """

    config_class = PixelPerfectDepthConfig

    def __init__(self, config: PixelPerfectDepthConfig):
        super().__init__(config)
        self._net: Optional[_PixelPerfectDepthNet] = None

    def _ensure_net(self):
        if self._net is not None:
            return
        self._net = _PixelPerfectDepthNet(
            sampling_steps=self.config.sampling_steps,
            dit_in_channels=self.config.dit_in_channels,
            dit_out_channels=self.config.dit_out_channels,
            dit_hidden_size=self.config.dit_hidden_size,
            dit_depth=self.config.dit_depth,
            dit_num_heads=self.config.dit_num_heads,
            dit_mlp_ratio=self.config.dit_mlp_ratio,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: (B, 3, H, W) ImageNet-normalized tensor from the processor.

        Returns:
            Depth tensor (B, H, W) with values in [0, 1].
        """
        self._ensure_net()
        device = pixel_values.device

        # Move net to the same device as input
        net_device = next(self._net.parameters()).device
        if net_device != device:
            self._net = self._net.to(device)

        # Denormalize from ImageNet stats -> [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        images = (pixel_values * std + mean).clamp(0, 1)

        # Pad H and W to be multiples of 16
        B, C, H, W = images.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))

        depths = []
        for i in range(B):
            depth = self._net.infer(images[i : i + 1])  # (1, 1, H', W')
            depths.append(depth.squeeze())

        result = torch.stack(depths)  # (B, H', W')

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            result = result[:, :H, :W]

        return result

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "PixelPerfectDepthModel":
        """Build the network and load checkpoints.

        Loading sequence:
          1. Load DINOv2 ViT-L weights from the DA-V2-Large checkpoint into the
             semantics encoder (strict=False — DPT head keys are silently ignored).
          2. Load the PPD checkpoint (ppd.pth) into the full network with
             strict=False, which updates the DiT weights and may also update
             the semantics encoder with fine-tuned values.
        """
        from huggingface_hub import hf_hub_download

        config = PixelPerfectDepthConfig()
        torch_device = torch.device(device)

        net = _PixelPerfectDepthNet(
            sampling_steps=config.sampling_steps,
            dit_in_channels=config.dit_in_channels,
            dit_out_channels=config.dit_out_channels,
            dit_hidden_size=config.dit_hidden_size,
            dit_depth=config.dit_depth,
            dit_num_heads=config.dit_num_heads,
            dit_mlp_ratio=config.dit_mlp_ratio,
        )

        # Step 1: Initialise semantics encoder with DA-V2 ViT-L weights
        da_v2_path = hf_hub_download(
            repo_id=config.semantics_hub_repo_id,
            filename=config.semantics_hub_filename,
            repo_type="model",
        )
        da_v2_state = torch.load(da_v2_path, map_location="cpu")
        missing, unexpected = net.semantics_encoder.load_state_dict(da_v2_state, strict=False)
        logger.info(
            "Loaded DA-V2 ViT-L semantics encoder from %s "
            "(missing=%d, unexpected=%d)",
            config.semantics_hub_repo_id,
            len(missing),
            len(unexpected),
        )

        # Step 2: Load full PPD checkpoint (DiT + possibly updated encoder)
        ppd_path = hf_hub_download(
            repo_id=config.hub_repo_id,
            filename=config.hub_filename,
            repo_type="model",
        )
        ppd_state = torch.load(ppd_path, map_location="cpu")
        missing, unexpected = net.load_state_dict(ppd_state, strict=False)
        logger.info(
            "Loaded PPD checkpoint from %s (missing=%d, unexpected=%d)",
            config.hub_repo_id,
            len(missing),
            len(unexpected),
        )

        net = net.to(torch_device).eval()

        model = cls(config)
        model._net = net

        logger.info("Pixel-Perfect Depth model ready on %s", torch_device)
        return model
