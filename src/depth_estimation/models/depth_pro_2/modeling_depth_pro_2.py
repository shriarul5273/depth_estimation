"""
Apple Depth Pro 2 — Self-contained single-file model implementation.

All network components (multi-view DINOv3 encoder, DPT decoder, inverse-depth
head, focal-length estimator) are inlined here, ported from
https://github.com/apple-aiml-research/ml-depth-pro-2. Loads weights from
HuggingFace Hub (``apple/DepthPro2-{S,B,L,H}`` + ``apple/DepthPro2-focal``).

Model weights are released under the Apple AMLR license (research use only:
non-commercial scientific research and academic development).
"""

# Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Network code adapted from ml-depth-pro-2 under its LICENSE (see the
# upstream repository for the full text and disclaimers).

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from ...modeling_utils import BaseDepthModel
from .configuration_depth_pro_2 import DepthPro2Config

logger = logging.getLogger(__name__)

#: Input sides are padded to a multiple of this (1/32 is the coarsest level).
INPUT_MULTIPLE = 32

_PRECISIONS = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_precision(precision: str, device: torch.device) -> torch.dtype:
    """Map a config precision string to an autocast dtype for ``device``."""
    if precision == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
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


def _create_dinov3(preset: str) -> nn.Module:
    try:
        vit = timm.create_model(
            preset, pretrained=False, num_classes=0, dynamic_img_size=True
        )
    except RuntimeError as e:  # unknown model name in old timm
        raise ImportError(
            f"Depth Pro 2 needs a timm version with DINOv3 models "
            f"('{preset}'); install timm>=1.0.20."
        ) from e
    vit.rope.normalize_coords = "max"
    return vit


# ---------------------------------------------------------------------------
# Encoder (network/encoder.py)
# ---------------------------------------------------------------------------


class MultiViewEncoder(nn.Module):
    """Multi-view DINOv3 transformer producing a 4-level feature pyramid per frame.

    Blocks in ``cross_frame_blocks`` attend jointly over the patch tokens of
    all frames of a clip; a single image is a clip of one frame.
    """

    def __init__(
        self,
        backbone_preset: str,
        reassembled_blocks: Sequence[int],
        cross_frame_blocks: Sequence[int],
        dims_reassemble: Sequence[int],
    ):
        super().__init__()
        vit = _create_dinov3(backbone_preset)

        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.reg_token = vit.reg_token
        self.rope = vit.rope
        self.blocks = vit.blocks

        self.embed_dim: int = vit.embed_dim
        self.patch_size: int = vit.patch_embed.patch_size[0]
        self.num_prefix_tokens: int = vit.num_prefix_tokens
        self.reassembled_blocks = tuple(reassembled_blocks)
        self.cross_frame_blocks = frozenset(cross_frame_blocks)

        dim = self.embed_dim
        d = dims_reassemble
        self.reassemble = nn.ModuleList(
            [
                # 1/16 -> 1/4.
                nn.Sequential(
                    nn.Conv2d(dim, d[0], kernel_size=1),
                    nn.ConvTranspose2d(d[0], d[0], kernel_size=4, stride=4),
                ),
                # 1/16 -> 1/8.
                nn.Sequential(
                    nn.Conv2d(dim, d[1], kernel_size=1),
                    nn.ConvTranspose2d(d[1], d[1], kernel_size=2, stride=2),
                ),
                # 1/16.
                nn.Sequential(nn.Conv2d(dim, d[2], kernel_size=1)),
                # 1/16 -> 1/32.
                nn.Sequential(
                    nn.Conv2d(dim, d[3], kernel_size=1),
                    nn.Conv2d(d[3], d[3], kernel_size=3, stride=2, padding=1),
                ),
            ]
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed frames ``(T, 3, H, W)`` and prepend class and register tokens."""
        tokens = self.patch_embed(x).flatten(1, 2)
        num_frames = tokens.shape[0]
        prefix = [
            self.cls_token.expand(num_frames, -1, -1),
            self.reg_token.expand(num_frames, -1, -1),
        ]
        return torch.cat(prefix + [tokens], dim=1)

    def cross_frame_attention(
        self, block: nn.Module, tokens: torch.Tensor, rope: torch.Tensor
    ) -> torch.Tensor:
        """Run one block jointly over the patch tokens of all frames."""
        num_frames, _, dim = tokens.shape
        npt = self.num_prefix_tokens
        prefix, patches = tokens[:, :npt], tokens[:, npt:]
        x = torch.cat(
            [prefix.mean(dim=0, keepdim=True), patches.reshape(1, -1, dim)], dim=1
        )
        x = block(x, rope=rope.repeat(num_frames, *(1,) * (rope.ndim - 1)))
        return torch.cat([prefix, x[:, npt:].reshape(num_frames, -1, dim)], dim=1)

    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        """Run the blocks over frames ``(T, 3, H, W)``; return tapped tokens and grid."""
        grid_size = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size)
        rope = self.rope.get_embed(shape=grid_size)

        tokens = self.embed(x)
        features = []
        for idx, block in enumerate(self.blocks):
            if idx in self.cross_frame_blocks:
                tokens = self.cross_frame_attention(block, tokens, rope)
            else:
                tokens = block(tokens, rope=rope)
            if idx in self.reassembled_blocks:
                features.append(tokens[:, self.num_prefix_tokens :])
        return features, grid_size

    def reassemble_tokens(
        self, features: Sequence[torch.Tensor], grid_size: Tuple[int, int]
    ) -> List[torch.Tensor]:
        """Reassemble tapped tokens into the 4-level pyramid. Independent per frame."""
        num_frames = features[0].shape[0]
        return [
            layer(
                feature.transpose(1, 2).reshape(num_frames, -1, *grid_size).contiguous()
            )
            for layer, feature in zip(self.reassemble, features)
        ]


# ---------------------------------------------------------------------------
# Decoder (network/decoder.py)
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module):
        super().__init__()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class _FeatureFusionBlock2d(nn.Module):
    """DPT feature fusion: add the skip features, refine, and upsample 2x."""

    def __init__(self, dim: int, with_skip: bool):
        super().__init__()
        if with_skip:
            self.resnet1 = self._residual_block(dim)
        self.resnet2 = self._residual_block(dim)
        self.deconv = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=False)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if skip is not None:
            x = x + self.resnet1(skip)
        x = self.resnet2(x)
        x = self.deconv(x)
        return self.out_conv(x)

    @staticmethod
    def _residual_block(dim: int) -> _ResidualBlock:
        return _ResidualBlock(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
        )


class DPTDecoder(nn.Module):
    """Decoder fusing a 4-level feature pyramid from low to high resolution."""

    def __init__(self, dims_reassemble: Sequence[int], dim_decoder: int):
        super().__init__()
        self.dim_out = dim_decoder
        num_levels = len(dims_reassemble)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(dim, dim_decoder, kernel_size=3, padding=1, bias=False)
                for dim in dims_reassemble
            ]
        )
        self.fusions = nn.ModuleList(
            [
                _FeatureFusionBlock2d(dim_decoder, with_skip=i < num_levels - 1)
                for i in range(num_levels)
            ]
        )

    def forward(self, encodings: Sequence[torch.Tensor]) -> torch.Tensor:
        """Decode the pyramid (highest resolution first) into 1/2-res features."""
        features = self.fusions[-1](self.convs[-1](encodings[-1]))
        for i in range(len(encodings) - 2, -1, -1):
            features = self.fusions[i](features, self.convs[i](encodings[i]))
        return features


def _create_inverse_depth_head(dim_in: int) -> nn.Sequential:
    """Head mapping 1/2-resolution decoder features to full-resolution inverse depth."""
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_in // 2, kernel_size=3, padding=1),
        nn.ConvTranspose2d(dim_in // 2, dim_in // 2, kernel_size=2, stride=2),
        nn.Conv2d(dim_in // 2, 32, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1),
        nn.ReLU(),
    )


# ---------------------------------------------------------------------------
# Focal-length estimator (network/fov.py)
# ---------------------------------------------------------------------------

#: Wider frames (long over short side) are centre-cropped to this.
_MAX_ASPECT = 1.7778


def _sincos_2d(gh: int, gw: int, dim: int, device: torch.device) -> torch.Tensor:
    """``(gh*gw, dim)`` sin-cos code; both axes scale by the longer side."""
    g = max(gh, gw)
    ys = (torch.arange(gh, device=device) + 0.5 - gh / 2) / g
    xs = (torch.arange(gw, device=device) + 0.5 - gw / 2) / g
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    freq = torch.pi * 2.0 ** torch.linspace(0.0, 7.0, dim // 4, device=device)
    return torch.cat(
        [f(c.reshape(-1, 1) * freq) for c in (yy, xx) for f in (torch.sin, torch.cos)],
        dim=-1,
    )


def _crop_angles_to_full(
    angles: torch.Tensor,
    crop_hw: Tuple[float, float],
    full_hw: Tuple[float, float],
) -> torch.Tensor:
    """``(..., 2)`` (hfov, vfov) degrees of a centred crop -> the full frame's."""
    (ch, cw), (h, w) = crop_hw, full_hw
    a = torch.deg2rad(angles.to(torch.promote_types(angles.dtype, torch.float32)))
    fx = (cw / 2) / torch.tan(a[..., 0] / 2)
    fy = (ch / 2) / torch.tan(a[..., 1] / 2)
    out = torch.stack(
        [2 * torch.atan((w / 2) / fx), 2 * torch.atan((h / 2) / fy)], dim=-1
    )
    return torch.rad2deg(out).to(angles.dtype)


def _median_focal_length_px(
    fov_deg: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Focal length in pixels at the lower median of ``(T, 2)`` FoVs of the shorter side."""
    hfov_deg, vfov_deg = fov_deg.median(dim=0).values
    if width < height:
        return 0.5 * width / torch.tan(torch.deg2rad(hfov_deg) / 2.0)
    return 0.5 * height / torch.tan(torch.deg2rad(vfov_deg) / 2.0)


class FOVHead(nn.Module):
    """A query token attends over four blocks' pooled patch, class and register tokens."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_taps: int = 4,
        num_prefix: int = 5,
        dim: int = 248,
        depth: int = 2,
        heads: int = 4,
        pool_stride: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix = num_prefix
        self.dim = dim
        self.pool_stride = pool_stride
        self.tap_norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(num_taps))
        self.proj = nn.Linear(embed_dim * num_taps, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.prefix_embed = nn.Parameter(torch.zeros(1, num_prefix, dim))
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        """Map tapped tokens ``(T, num_prefix + gh*gw, num_taps*C)`` to ``(T, 2)`` logits."""
        num_frames = tokens.shape[0]
        gh, gw = grid_size
        taps = tokens.float().split(self.embed_dim, dim=-1)
        x = self.proj(
            torch.cat([norm(t) for norm, t in zip(self.tap_norms, taps)], dim=-1)
        )
        prefix, x = x[:, : self.num_prefix], x[:, self.num_prefix :]
        x = x.transpose(1, 2).reshape(num_frames, self.dim, gh, gw)
        x = F.avg_pool2d(x, self.pool_stride, ceil_mode=True)
        x = x.flatten(2).transpose(1, 2) + _sincos_2d(
            *x.shape[-2:], self.dim, x.device
        )
        query = self.query.expand(num_frames, -1, -1)
        x = torch.cat([query, prefix + self.prefix_embed, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, 0]))


class FOVEstimator(nn.Module):
    """DINOv3 ViT-S/16 and an attention head regressing each frame's two FoV angles."""

    max_short_side = 512
    min_fov_deg = 1.0
    max_fov_deg = 179.0
    tapped_blocks = frozenset((2, 5, 8, 11))
    # Cross-frame blocks in training; a frame alone keeps its input prefix tokens.
    cross_frame_blocks = frozenset((5, 7, 9, 11))

    def __init__(self):
        super().__init__()
        vit = _create_dinov3("vit_small_patch16_dinov3")
        self.trunk = nn.Module()
        self.trunk.patch_embed = vit.patch_embed
        self.trunk.cls_token = vit.cls_token
        self.trunk.reg_token = vit.reg_token
        self.trunk.rope = vit.rope
        self.trunk.blocks = vit.blocks
        self.num_prefix_tokens: int = vit.num_prefix_tokens
        self.head = FOVHead(embed_dim=vit.embed_dim, num_prefix=self.num_prefix_tokens)

    def input_size(self, height: int, width: int) -> Tuple[int, int]:
        """Resize target of a ``height`` x ``width`` frame; never upsampled."""
        short = max(32, min(self.max_short_side, min(height, width) // 32 * 32))
        scale = short / min(height, width)
        return (
            max(32, round(height * scale / 32) * 32),
            max(32, round(width * scale / 32) * 32),
        )

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize frames to ``input_size``; centre-crop those wider than 16:9."""
        new_h, new_w = self.input_size(*x.shape[-2:])
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True)
        lo, hi = min(new_h, new_w), max(new_h, new_w)
        if hi <= _MAX_ASPECT * lo:
            return x
        keep = max(32, int(_MAX_ASPECT * lo) // 32 * 32)
        off = (hi - keep) // 2
        return x[..., off : off + keep] if new_w > new_h else x[..., off : off + keep, :]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        trunk = self.trunk
        grid_size = (x.shape[-2] // 16, x.shape[-1] // 16)
        rope = trunk.rope.get_embed(shape=grid_size)
        tokens = trunk.patch_embed(x).flatten(1, 2)
        num_frames = tokens.shape[0]
        prefix = [
            t.expand(num_frames, -1, -1) for t in (trunk.cls_token, trunk.reg_token)
        ]
        tokens = torch.cat(prefix + [tokens], dim=1)
        npt = self.num_prefix_tokens
        taps = []
        for idx, block in enumerate(trunk.blocks):
            out = block(tokens, rope=rope)
            if idx in self.cross_frame_blocks:
                out = torch.cat([tokens[:, :npt], out[:, npt:]], dim=1)
            tokens = out
            if idx in self.tapped_blocks:
                taps.append(tokens)
        return torch.cat(taps, dim=-1), grid_size

    def forward(self, x: torch.Tensor, precision: torch.dtype) -> torch.Tensor:
        """Predict ``(T, 2)`` horizontal and vertical FoV in degrees for ``(T, 3, H, W)``."""
        height, width = x.shape[-2:]
        y = self.resize(x)
        with _autocast(x.device, precision):
            tokens, grid_size = self.encode(y)
        logits = self.head(tokens, grid_size)
        fov_deg = self.min_fov_deg + torch.sigmoid(logits) * (
            self.max_fov_deg - self.min_fov_deg
        )
        full_h, full_w = self.input_size(height, width)
        crop_h, crop_w = y.shape[-2:]
        if (crop_h, crop_w) == (full_h, full_w):
            return fov_deg
        crop_in_source = (crop_h * height / full_h, crop_w * width / full_w)
        return _crop_angles_to_full(fov_deg, crop_in_source, (height, width))

    def focal_length_px(self, x: torch.Tensor, precision: torch.dtype) -> torch.Tensor:
        """Estimate one focal length in pixels for frames ``(T, 3, H, W)``."""
        return _median_focal_length_px(self.forward(x, precision), *x.shape[-2:])


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class DepthPro2Model(BaseDepthModel):
    """Apple Depth Pro 2 model.

    Sharp, temporally consistent metric monocular depth at the input's
    native resolution (padded to a multiple of 32).

    Usage::

        model = DepthPro2Model.from_pretrained("depth-pro-2-l")
        depth = model(pixel_values)  # (B, H, W) tensor in meters

    ``forward()`` treats every batch element as an independent photo. To
    predict a video clip whose frames attend to each other and share one
    focal length, use :meth:`infer` with a ``(T, 3, H, W)`` tensor.
    """

    config_class = DepthPro2Config

    def __init__(self, config: DepthPro2Config):
        super().__init__(config)
        self.encoder = MultiViewEncoder(
            backbone_preset=config.backbone_preset,
            reassembled_blocks=config.reassembled_blocks,
            cross_frame_blocks=config.cross_frame_blocks,
            dims_reassemble=config.dims_reassemble,
        )
        self.decoder = DPTDecoder(
            dims_reassemble=config.dims_reassemble, dim_decoder=config.dim_decoder
        )
        self.head = _create_inverse_depth_head(self.decoder.dim_out)
        self.fov = FOVEstimator() if config.use_fov else None

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    @property
    def precision(self) -> torch.dtype:
        return _resolve_precision(self.config.precision, self.device)

    def predict_inverse_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Focal-normalised inverse depth ``(T, H, W)`` for one clip ``(T, 3, H, W)``."""
        height, width = x.shape[-2:]
        pad_h = -height % INPUT_MULTIPLE
        pad_w = -width % INPUT_MULTIPLE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        precision = self.precision
        with _autocast(x.device, precision):
            features, grid_size = self.encoder.encode(x)

        inverse_depth = []
        for frame in range(x.shape[0]):
            tokens = [f[frame : frame + 1] for f in features]
            with _autocast(x.device, precision):
                decoded = self.decoder(
                    self.encoder.reassemble_tokens(tokens, grid_size)
                )
            inverse_depth.append(self.head(decoded.float()))
        return torch.cat(inverse_depth)[:, 0, :height, :width]

    def _predict(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        width = x.shape[-1]
        inverse_depth = self.predict_inverse_depth(x)

        if f_px is not None:
            f_px = torch.as_tensor(
                f_px, dtype=torch.float32, device=x.device
            ).squeeze()
        elif self.fov is not None:
            # The focal estimator is a separate pretrained network; keep it
            # out of the depth gradient path during fine-tuning.
            with torch.no_grad():
                f_px = self.fov.focal_length_px(x, self.precision)

        # Depth is infinite where the network predicts zero inverse depth (sky).
        depth = None if f_px is None else f_px / (width * inverse_depth)
        return {"inverse_depth": inverse_depth, "depth": depth, "focallength_px": f_px}

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Infer inverse depth, metric depth and focal length at native resolution.

        Args:
            x: Normalised image ``(3, H, W)`` or clip ``(T, 3, H, W)``. A clip's
                frames attend to each other and share one focal length, so
                pass unrelated photos one at a time.
            f_px: Focal length in pixels of ``x`` (e.g. from EXIF). Estimated
                with the focal-length network when ``None``.

        Returns:
            Dict with ``inverse_depth`` (focal-normalised), ``depth`` in meters
            (``(H, W)`` or ``(T, H, W)``, ``inf`` where infinitely far, such as
            sky) and ``focallength_px`` (0-d tensor).
            ``depth`` is ``None`` if no focal length is given or estimated.
        """
        if x.ndim not in (3, 4) or x.shape[-3] != 3:
            raise ValueError(
                f"infer takes a (3, H, W) or (T, 3, H, W) tensor, got {tuple(x.shape)}."
            )
        is_image = x.ndim == 3
        if is_image:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        out = self._predict(x, f_px)
        if is_image:
            out["inverse_depth"] = out["inverse_depth"][0]
            if out["depth"] is not None:
                out["depth"] = out["depth"][0]
        return out

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W) normalised with ImageNet
                mean/std. Each batch element is an independent image.

        Returns:
            Depth tensor (B, H, W) in meters, 0 where depth is infinite (sky)
            or beyond ``config.max_depth``. Focal-normalised inverse depth if
            the model was built with ``use_fov=False``. Use :meth:`infer` for
            the raw prediction with ``inf`` sky.
        """
        pixel_values = pixel_values.to(self.device)
        outputs = []
        for i in range(pixel_values.shape[0]):
            result = self._predict(pixel_values[i : i + 1])
            depth = result["depth"]
            if depth is None:
                outputs.append(result["inverse_depth"])
                continue
            # Like MoGe, mark infinite / out-of-range depth (sky) as invalid
            # (0) so the shared min-max normalisation and metrics stay sane.
            valid = torch.isfinite(depth)
            if self.config.max_depth is not None:
                valid &= depth <= self.config.max_depth
            outputs.append(torch.where(valid, depth, torch.zeros_like(depth)))
        return torch.cat(outputs)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _backbone_module(self) -> nn.Module:
        """Return the multi-view DINOv3 encoder."""
        return self.encoder

    def unfreeze_top_k_backbone_layers(self, k: int) -> None:
        """Unfreeze the last k transformer blocks of the DINOv3 encoder."""
        for block in list(self.encoder.blocks)[-k:]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(
            f"Unfroze top {k} Depth Pro 2 encoder blocks. "
            f"Trainable params: {self._count_trainable():,}"
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
    ) -> "DepthPro2Model":
        """Build the architecture and load the depth + focal checkpoints from HF Hub."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config = DepthPro2Config.from_variant(model_id)
        for key, value in kwargs.items():
            setattr(config, key, value)

        torch_device = torch.device(device)
        with torch.device(torch_device):
            model = cls(config)

        path = hf_hub_download(repo_id=config.hub_repo_id, filename=config.hub_filename)
        state_dict = load_file(path)
        if model.fov is not None:
            fov_path = hf_hub_download(
                repo_id=config.fov_repo_id, filename=config.hub_filename
            )
            state_dict.update(
                {f"fov.{k}": v for k, v in load_file(fov_path).items()}
            )
        model.load_state_dict(state_dict, strict=True)

        model = model.to(torch_device).eval()
        logger.info("Loaded Apple Depth Pro 2 from %s", config.hub_repo_id)
        return model
