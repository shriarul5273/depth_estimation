"""
Apple DepthPro — Self-contained single-file model implementation.

All network components (ViT utilities, encoder, decoder, FOV head) are inlined
here. Loads weights from HuggingFace Hub (apple/DepthPro).
"""

# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from __future__ import annotations

import logging
import math
import types
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from torch.utils.checkpoint import checkpoint

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_depth_pro import DepthProConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ViT utilities (vit.py)
# ---------------------------------------------------------------------------

def _make_vit_b16_backbone(
    model,
    encoder_feature_dims,
    encoder_feature_layer_ids,
    vit_features,
    start_index=1,
    use_grad_checkpointing=False,
) -> nn.Module:
    """Make a ViTb16 backbone for the DPT model."""
    if use_grad_checkpointing:
        model.set_grad_checkpointing()

    vit_model = nn.Module()
    vit_model.hooks = encoder_feature_layer_ids
    vit_model.model = model
    vit_model.features = encoder_feature_dims
    vit_model.vit_features = vit_features
    vit_model.model.start_index = start_index
    vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
    vit_model.model.is_vit = True
    vit_model.model.forward = vit_model.model.forward_features

    return vit_model


def _forward_features_eva_fixed(self, x):
    """Encode features (EVA variant)."""
    x = self.patch_embed(x)
    x, rot_pos_embed = self._pos_embed(x)
    for blk in self.blocks:
        if self.grad_checkpointing:
            x = checkpoint(blk, x, rot_pos_embed)
        else:
            x = blk(x, rot_pos_embed)
    x = self.norm(x)
    return x


def _resize_vit(model: nn.Module, img_size) -> nn.Module:
    """Resample the ViT module to the given size."""
    patch_size = model.patch_embed.patch_size
    model.patch_embed.img_size = img_size
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    model.patch_embed.grid_size = grid_size

    pos_embed = resample_abs_pos_embed(
        model.pos_embed,
        grid_size,
        num_prefix_tokens=(
            0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
        ),
    )
    model.pos_embed = torch.nn.Parameter(pos_embed)
    return model


def _resize_patch_embed(model: nn.Module, new_patch_size=(16, 16)) -> nn.Module:
    """Resample the ViT patch size to the given one."""
    if hasattr(model, "patch_embed"):
        old_patch_size = model.patch_embed.patch_size

        if (
            new_patch_size[0] != old_patch_size[0]
            or new_patch_size[1] != old_patch_size[1]
        ):
            patch_embed_proj = model.patch_embed.proj.weight
            patch_embed_proj_bias = model.patch_embed.proj.bias
            use_bias = patch_embed_proj_bias is not None
            _, _, h, w = patch_embed_proj.shape

            new_patch_embed_proj = torch.nn.functional.interpolate(
                patch_embed_proj,
                size=[new_patch_size[0], new_patch_size[1]],
                mode="bicubic",
                align_corners=False,
            )
            new_patch_embed_proj = (
                new_patch_embed_proj * (h / new_patch_size[0]) * (w / new_patch_size[1])
            )

            model.patch_embed.proj = nn.Conv2d(
                in_channels=model.patch_embed.proj.in_channels,
                out_channels=model.patch_embed.proj.out_channels,
                kernel_size=new_patch_size,
                stride=new_patch_size,
                bias=use_bias,
            )
            if use_bias:
                model.patch_embed.proj.bias = patch_embed_proj_bias
            model.patch_embed.proj.weight = torch.nn.Parameter(new_patch_embed_proj)

            model.patch_size = new_patch_size
            model.patch_embed.patch_size = new_patch_size
            model.patch_embed.img_size = (
                int(model.patch_embed.img_size[0] * new_patch_size[0] / old_patch_size[0]),
                int(model.patch_embed.img_size[1] * new_patch_size[1] / old_patch_size[1]),
            )

    return model


# ---------------------------------------------------------------------------
# ViT factory (vit_factory.py)
# ---------------------------------------------------------------------------

ViTPreset = Literal["dinov2l16_384"]


@dataclass
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int

    img_size: int = 384
    patch_size: int = 16

    timm_preset: Optional[str] = None
    timm_img_size: int = 384
    timm_patch_size: int = 16

    encoder_feature_layer_ids: List[int] = None
    encoder_feature_dims: List[int] = None


VIT_CONFIG_DICT: Dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        encoder_feature_layer_ids=[5, 11, 17, 23],
        encoder_feature_dims=[256, 512, 1024, 1024],
        img_size=384,
        patch_size=16,
        timm_preset="vit_large_patch14_dinov2",
        timm_img_size=518,
        timm_patch_size=14,
    ),
}


def create_vit(
    preset: ViTPreset,
    use_pretrained: bool = False,
    checkpoint_uri: str | None = None,
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """Create and load a ViT backbone module."""
    config = VIT_CONFIG_DICT[preset]

    img_size = (config.img_size, config.img_size)
    patch_size = (config.patch_size, config.patch_size)

    if "eva02" in preset:
        model = timm.create_model(config.timm_preset, pretrained=use_pretrained)
        model.forward_features = types.MethodType(_forward_features_eva_fixed, model)
    else:
        model = timm.create_model(
            config.timm_preset, pretrained=use_pretrained, dynamic_img_size=True
        )
    model = _make_vit_b16_backbone(
        model,
        encoder_feature_dims=config.encoder_feature_dims,
        encoder_feature_layer_ids=config.encoder_feature_layer_ids,
        vit_features=config.embed_dim,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    if config.patch_size != config.timm_patch_size:
        model.model = _resize_patch_embed(model.model, new_patch_size=patch_size)
    if config.img_size != config.timm_img_size:
        model.model = _resize_vit(model.model, img_size=img_size)

    if checkpoint_uri is not None:
        state_dict = torch.load(checkpoint_uri, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=False
        )
        if len(unexpected_keys) != 0:
            raise KeyError(f"Found unexpected keys when loading vit: {unexpected_keys}")
        if len(missing_keys) != 0:
            raise KeyError(f"Keys are missing when loading vit: {missing_keys}")

    return model.model


# ---------------------------------------------------------------------------
# Encoder (encoder.py)
# ---------------------------------------------------------------------------

class DepthProEncoder(nn.Module):
    """DepthPro Encoder combining patch and image encoders at multiple resolutions."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        super().__init__()

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)

        patch_encoder_embed_dim = patch_encoder.embed_dim
        image_encoder_embed_dim = image_encoder.embed_dim

        self.out_size = int(
            patch_encoder.patch_embed.img_size[0] // patch_encoder.patch_embed.patch_size[0]
        )

        def _create_project_upsample_block(
            dim_in: int,
            dim_out: int,
            upsample_layers: int,
            dim_int: Optional[int] = None,
        ) -> nn.Module:
            if dim_int is None:
                dim_int = dim_out
            blocks = [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_int,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ]
            blocks += [
                nn.ConvTranspose2d(
                    in_channels=dim_int if i == 0 else dim_out,
                    out_channels=dim_out,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for i in range(upsample_layers)
            ]
            return nn.Sequential(*blocks)

        self.upsample_latent0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim,
            dim_int=self.dims_encoder[0],
            dim_out=decoder_features,
            upsample_layers=3,
        )
        self.upsample_latent1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[0], upsample_layers=2
        )
        self.upsample0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[1], upsample_layers=1
        )
        self.upsample1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[2], upsample_layers=1
        )
        self.upsample2 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[3], upsample_layers=1
        )
        self.upsample_lowres = nn.ConvTranspose2d(
            in_channels=image_encoder_embed_dim,
            out_channels=self.dims_encoder[3],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.fuse_lowres = nn.Conv2d(
            in_channels=(self.dims_encoder[3] + self.dims_encoder[3]),
            out_channels=self.dims_encoder[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(self._hook0)
        self.patch_encoder.blocks[self.hook_block_ids[1]].register_forward_hook(self._hook1)

    def _hook0(self, model, input, output):
        self.backbone_highres_hook0 = output

    def _hook1(self, model, input, output):
        self.backbone_highres_hook1 = output

    @property
    def img_size(self) -> int:
        return self.patch_encoder.patch_embed.img_size[0] * 4

    def _create_pyramid(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = x
        x1 = F.interpolate(x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False)
        return x0, x1, x2

    def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        patch_size = 384
        patch_stride = int(patch_size * (1 - overlap_ratio))
        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1
        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size
            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])
        return torch.cat(x_patch_list, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        steps = int(math.sqrt(x.shape[0] // batch_size))
        idx = 0
        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]
                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]
                output_row_list.append(output)
                idx += 1
            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        return torch.cat(output_list, dim=-2)

    def reshape_feature(self, embeddings: torch.Tensor, width, height, cls_token_offset=1):
        b, hw, c = embeddings.shape
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]
        embeddings = embeddings.reshape(b, height, width, c).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch_size = x.shape[0]

        x0, x1, x2 = self._create_pyramid(x)

        x0_patches = self.split(x0, overlap_ratio=0.25)
        x1_patches = self.split(x1, overlap_ratio=0.5)
        x2_patches = x2

        x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_patches), dim=0)

        x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = self.reshape_feature(
            x_pyramid_encodings, self.out_size, self.out_size
        )

        x_latent0_encodings = self.reshape_feature(
            self.backbone_highres_hook0, self.out_size, self.out_size
        )
        x_latent0_features = self.merge(
            x_latent0_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        x_latent1_encodings = self.reshape_feature(
            self.backbone_highres_hook1, self.out_size, self.out_size
        )
        x_latent1_features = self.merge(
            x_latent1_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )

        x0_features = self.merge(x0_encodings, batch_size=batch_size, padding=3)
        x1_features = self.merge(x1_encodings, batch_size=batch_size, padding=6)
        x2_features = x2_encodings

        x_global_features = self.image_encoder(x2_patches)
        x_global_features = self.reshape_feature(
            x_global_features, self.out_size, self.out_size
        )

        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)
        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)

        x_global_features = self.upsample_lowres(x_global_features)
        x_global_features = self.fuse_lowres(
            torch.cat((x2_features, x_global_features), dim=1)
        )

        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]


# ---------------------------------------------------------------------------
# Decoder (decoder.py)
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module, shortcut: nn.Module | None = None) -> None:
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_x = self.residual(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + delta_x


class _FeatureFusionBlock2d(nn.Module):
    def __init__(self, num_features: int, deconv: bool = False, batch_norm: bool = False):
        super().__init__()

        self.resnet1 = self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)

        self.use_deconv = deconv
        if deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.out_conv = nn.Conv2d(
            num_features, num_features, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        x = x0
        if x1 is not None:
            res = self.resnet1(x1)
            x = self.skip_add.add(x, res)
        x = self.resnet2(x)
        if self.use_deconv:
            x = self.deconv(x)
        x = self.out_conv(x)
        return x

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool):
        def _create_block(dim: int, batch_norm: bool) -> list[nn.Module]:
            layers = [
                nn.ReLU(False),
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not batch_norm,
                ),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = nn.Sequential(
            *_create_block(dim=num_features, batch_norm=batch_norm),
            *_create_block(dim=num_features, batch_norm=batch_norm),
        )
        return _ResidualBlock(residual)


class MultiresConvDecoder(nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(self, dims_encoder: Iterable[int], dim_decoder: int):
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder
        self.dim_out = dim_decoder

        num_encoders = len(self.dims_encoder)

        conv0 = (
            nn.Conv2d(self.dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
            if self.dims_encoder[0] != dim_decoder
            else nn.Identity()
        )
        convs = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                nn.Conv2d(
                    self.dims_encoder[i],
                    dim_decoder,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
        self.convs = nn.ModuleList(convs)

        fusions = []
        for i in range(num_encoders):
            fusions.append(
                _FeatureFusionBlock2d(num_features=dim_decoder, deconv=(i != 0), batch_norm=False)
            )
        self.fusions = nn.ModuleList(fusions)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(
                f"Got encoder output levels={num_levels}, expected levels={num_encoders}."
            )

        features = self.convs[-1](encodings[-1])
        lowres_features = features
        features = self.fusions[-1](features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        return features, lowres_features


# ---------------------------------------------------------------------------
# FOV network (fov.py)
# ---------------------------------------------------------------------------

class FOVNetwork(nn.Module):
    """Field of View estimation network."""

    def __init__(self, num_features: int, fov_encoder: Optional[nn.Module] = None):
        super().__init__()

        fov_head0 = [
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        ]
        fov_head = [
            nn.Conv2d(num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=0),
        ]
        if fov_encoder is not None:
            self.encoder = nn.Sequential(
                fov_encoder, nn.Linear(fov_encoder.embed_dim, num_features // 2)
            )
            self.downsample = nn.Sequential(*fov_head0)
        else:
            fov_head = fov_head0 + fov_head
        self.head = nn.Sequential(*fov_head)

    def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "encoder"):
            x = F.interpolate(
                x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False
            )
            x = self.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.downsample(lowres_feature)
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = lowres_feature
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class _DepthProNet(nn.Module):
    """DepthPro neural network (encoder + decoder + depth head + optional FOV head)."""

    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(dim_decoder // 2, last_dims[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.head[4].bias.data.fill_(0)

        if use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)

    @property
    def img_size(self) -> int:
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass — returns (canonical_inverse_depth, fov_deg)."""
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg = self.fov.forward(x, features_0.detach())

        return canonical_inverse_depth, fov_deg

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
        interpolation_mode: str = "bilinear",
    ) -> dict:
        """Infer metric depth and focal length for a given image tensor."""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        resize = H != self.img_size or W != self.img_size

        if resize:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth, fov_deg = self.forward(x)

        if f_px is None:
            f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))

        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = F.interpolate(
                inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
            )

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        return {"depth": depth.squeeze(), "focallength_px": f_px}


def _build_depth_pro_net(config: DepthProConfig, device: torch.device) -> _DepthProNet:
    """Instantiate the DepthPro network architecture from config."""
    patch_encoder = create_vit(preset=config.patch_encoder_preset, use_pretrained=False)
    image_encoder = create_vit(preset=config.image_encoder_preset, use_pretrained=False)

    fov_encoder = None
    if config.use_fov_head and config.fov_encoder_preset is not None:
        fov_encoder = create_vit(preset=config.fov_encoder_preset, use_pretrained=False)

    vit_cfg = VIT_CONFIG_DICT[config.patch_encoder_preset]
    encoder = DepthProEncoder(
        dims_encoder=vit_cfg.encoder_feature_dims,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=vit_cfg.encoder_feature_layer_ids,
        decoder_features=config.decoder_features,
    )
    decoder = MultiresConvDecoder(
        dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=config.decoder_features,
    )
    net = _DepthProNet(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        use_fov_head=config.use_fov_head,
        fov_encoder=fov_encoder,
    ).to(device)
    return net


class DepthProModel(BaseDepthModel):
    """Apple DepthPro model.

    Sharp monocular metric depth estimation using the bundled network implementation.

    Usage::

        model = DepthProModel.from_pretrained("depth-pro")
        depth = model(pixel_values)  # (B, H, W) tensor in meters
    """

    config_class = DepthProConfig

    def __init__(self, config: DepthProConfig):
        super().__init__(config)
        self._net: Optional[_DepthProNet] = None

    def _ensure_net(self):
        if self._net is not None:
            return
        device = torch.device(_auto_detect_device())
        self._net = _build_depth_pro_net(self.config, device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W) normalized with the model's
                mean/std ([0.5, 0.5, 0.5] for DepthPro).

        Returns:
            Depth tensor (B, H, W) in meters.
        """
        self._ensure_net()

        device = next(self._net.parameters()).device
        pixel_values = pixel_values.to(device)

        depths = []
        for i in range(pixel_values.shape[0]):
            result = self._net.infer(pixel_values[i : i + 1], f_px=None)
            depths.append(result["depth"])
        return torch.stack(depths)

    def _backbone_module(self):
        """Return the DepthPro encoder (_DepthProNet.encoder).

        Calls _ensure_net() to lazily initialise the network if needed.
        The encoder is a DepthProEncoder with patch_encoder and image_encoder.
        """
        self._ensure_net()
        return self._net.encoder

    def unfreeze_top_k_backbone_layers(self, k: int) -> None:
        """Unfreeze the last k blocks of both ViT encoders in DepthProEncoder.

        Overrides the DINOv2-specific base implementation.
        DepthProEncoder has two timm ViT models: patch_encoder and image_encoder.

        Args:
            k: Number of blocks to unfreeze from the top of each ViT encoder.
        """
        self._ensure_net()
        enc = self._net.encoder
        for vit in (enc.patch_encoder, enc.image_encoder):
            if hasattr(vit, "blocks"):
                for block in list(vit.blocks)[-k:]:
                    for param in block.parameters():
                        param.requires_grad = True
        logger.info(
            f"Unfroze top {k} DepthPro encoder blocks. "
            f"Trainable params: {self._count_trainable():,}"
        )

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthProModel":
        """Build the DepthPro architecture and load checkpoint from HF Hub."""
        from huggingface_hub import hf_hub_download

        config = DepthProConfig()
        torch_device = torch.device(device)

        net = _build_depth_pro_net(config, torch_device)

        checkpoint_path = hf_hub_download(
            repo_id=config.hub_repo_id,
            filename=config.hub_filename,
            repo_type="model",
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = net.load_state_dict(state_dict=state_dict, strict=True)

        if unexpected_keys:
            raise KeyError(f"Unexpected keys loading DepthPro: {unexpected_keys}")
        missing_keys = [k for k in missing_keys if "fc_norm" not in k]
        if missing_keys:
            raise KeyError(f"Missing keys loading DepthPro: {missing_keys}")

        net = net.to(torch_device).eval()

        model = cls(config)
        model._net = net

        logger.info("Loaded Apple DepthPro from %s", config.hub_repo_id)
        return model
