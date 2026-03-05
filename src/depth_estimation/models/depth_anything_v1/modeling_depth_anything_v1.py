"""
Depth Anything v1 — Single-file model implementation.

Architecture: DINOv2 encoder (via torch.hub) + DPT decoder head.
All components inlined per the Transformers single-file policy.
Ported from Depth-Estimation-Compare-demo/Depth-Anything/depth_anything/.

Weights loaded via HuggingFace Hub (PyTorchModelHubMixin).
"""

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_depth_anything_v1 import DepthAnythingV1Config, _V1_VARIANT_MAP

logger = logging.getLogger(__name__)


# ============================================================================ #
#  Inlined architecture components (single-file policy)
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

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
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

    def __init__(self, nclass, in_channels, features=256, use_bn=False,
                 out_channels=None, use_clstoken=False):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]

        self.nclass = nclass
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

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
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


class DPT_DINOv2(nn.Module):
    """DPT model with DINOv2 backbone (loaded via torch.hub)."""

    def __init__(self, encoder="vitl", features=256, out_channels=None,
                 use_bn=False, use_clstoken=False, localhub=False):
        super().__init__()
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]

        assert encoder in ["vits", "vitb", "vitl"]

        if localhub:
            torchhub_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..",
                "Depth-Estimation-Compare-demo", "Depth-Anything", "torchhub",
                "facebookresearch_dinov2_main",
            )
            torchhub_path = os.path.abspath(torchhub_path)
            self.pretrained = torch.hub.load(
                torchhub_path, f"dinov2_{encoder}14", source="local", pretrained=False
            )
        else:
            self.pretrained = torch.hub.load(
                "facebookresearch/dinov2", f"dinov2_{encoder}14"
            )

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHead(
            1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1)


class _DepthAnythingHubModel(DPT_DINOv2, PyTorchModelHubMixin):
    """Internal class combining DPT_DINOv2 with HuggingFace Hub loading."""

    def __init__(self, config):
        super().__init__(**config)


# ============================================================================ #
#  Public model class
# ============================================================================ #


class DepthAnythingV1Model(BaseDepthModel):
    """Depth Anything v1 model.

    Usage::

        model = DepthAnythingV1Model.from_pretrained("depth-anything-v1-vitb")
        depth = model(pixel_values)  # (B, H, W) tensor
    """

    config_class = DepthAnythingV1Config

    def __init__(self, config: DepthAnythingV1Config):
        super().__init__(config)
        self.net = DPT_DINOv2(
            encoder=config.backbone,
            features=config.features,
            out_channels=config.out_channels,
            use_bn=config.use_bn,
            use_clstoken=config.use_clstoken,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W).
        """
        return self.net(pixel_values)

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "DepthAnythingV1Model":
        """Load v1 weights from HuggingFace Hub.

        Uses PyTorchModelHubMixin for downloading from repos like
        ``LiheYoung/depth_anything_vitb14``.
        """
        # Resolve variant ID → backbone
        backbone = _V1_VARIANT_MAP.get(model_id)
        if backbone is None:
            # Try treating model_id as direct HF Hub ID
            backbone = model_id.split("_")[-1].replace("14", "")
            if backbone not in ("vits", "vitb", "vitl"):
                raise ValueError(
                    f"Cannot resolve backbone from '{model_id}'. "
                    f"Use one of: {list(_V1_VARIANT_MAP.keys())}"
                )

        config = DepthAnythingV1Config(backbone=backbone)

        # Load via PyTorchModelHubMixin
        hub_model = _DepthAnythingHubModel.from_pretrained(
            config.hub_model_id
        )

        # Wrap in our model class
        model = cls(config)
        model.net.load_state_dict(hub_model.state_dict())
        model = model.to(device)

        logger.info(f"Loaded Depth Anything v1 ({backbone}) from {config.hub_model_id}")
        return model
