"""
Apple Depth Pro 2 configuration.
"""

from typing import List, Optional

from ...configuration_utils import BaseDepthConfig


# Architecture presets, mirroring the ``config.json`` shipped in each
# ``apple/DepthPro2-*`` Hub repository (and ``PRESETS`` in ml-depth-pro-2).
_DEPTHPRO2_VARIANT_MAP = {
    "depth-pro-2-s": {
        "backbone": "vits",
        "hub_repo_id": "apple/DepthPro2-S",
        "backbone_preset": "vit_small_patch16_dinov3",
        "reassembled_blocks": [2, 5, 8, 11],
        "cross_frame_blocks": [5, 7, 9, 11],
        "dims_reassemble": [48, 96, 192, 384],
        "dim_decoder": 64,
    },
    "depth-pro-2-b": {
        "backbone": "vitb",
        "hub_repo_id": "apple/DepthPro2-B",
        "backbone_preset": "vit_base_patch16_dinov3",
        "reassembled_blocks": [2, 5, 8, 11],
        "cross_frame_blocks": [5, 7, 9, 11],
        "dims_reassemble": [96, 192, 384, 768],
        "dim_decoder": 128,
    },
    "depth-pro-2-l": {
        "backbone": "vitl",
        "hub_repo_id": "apple/DepthPro2-L",
        "backbone_preset": "vit_large_patch16_dinov3.lvd1689m",
        "reassembled_blocks": [5, 11, 17, 23],
        "cross_frame_blocks": [9, 11, 13, 15, 17, 19, 21, 23],
        "dims_reassemble": [256, 512, 1024, 1024],
        "dim_decoder": 256,
    },
    "depth-pro-2-h": {
        "backbone": "vith+",
        "hub_repo_id": "apple/DepthPro2-H",
        "backbone_preset": "vit_huge_plus_patch16_dinov3",
        "reassembled_blocks": [7, 15, 23, 31],
        "cross_frame_blocks": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        "dims_reassemble": [256, 512, 1024, 1024],
        "dim_decoder": 256,
    },
}


class DepthPro2Config(BaseDepthConfig):
    """Configuration for Apple Depth Pro 2.

    Sharp, temporally consistent metric monocular depth. A DINOv3 multi-view
    transformer feeds a DPT decoder that predicts focal-normalised inverse
    depth; a separate DINOv3 ViT-S focal-length estimator
    (``apple/DepthPro2-focal``) turns it into metric depth when the focal
    length is unknown. Uses the bundled network implementation — no external
    ``depth_pro_2`` package needed.

    The network runs at any resolution that is a multiple of 32, so
    ``patch_size`` is 32 here (the processor's resize multiple) even though
    the ViT itself uses 16-pixel patches. ``input_size`` is the shorter side
    the processor resizes to; raise it for sharper (slower) predictions.

    ``precision`` is the autocast dtype of the backbones and decoder:
    ``"auto"`` (bfloat16 on CUDA/MPS as trained, float32 on CPU),
    ``"float32"``, ``"bfloat16"`` or ``"float16"``. The heads always run in
    float32.

    ``forward()`` returns 0 (invalid) where depth is infinite (sky) or beyond
    ``max_depth`` meters, like MoGe; ``model.infer()`` returns the raw
    prediction with ``inf`` sky.
    """

    model_type = "depth-pro-2"

    def __init__(
        self,
        backbone: str = "vits",
        input_size: int = 768,
        patch_size: int = 32,
        is_metric: bool = True,
        hub_repo_id: str = "apple/DepthPro2-S",
        hub_filename: str = "model.safetensors",
        fov_repo_id: Optional[str] = "apple/DepthPro2-focal",
        use_fov: bool = True,
        backbone_preset: str = "vit_small_patch16_dinov3",
        reassembled_blocks: Optional[List[int]] = None,
        cross_frame_blocks: Optional[List[int]] = None,
        dims_reassemble: Optional[List[int]] = None,
        dim_decoder: int = 64,
        precision: str = "auto",
        max_depth: Optional[float] = 1000.0,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            max_depth=max_depth,
            **kwargs,
        )
        self.hub_repo_id = hub_repo_id
        self.hub_filename = hub_filename
        self.fov_repo_id = fov_repo_id
        self.use_fov = use_fov
        self.backbone_preset = backbone_preset
        self.reassembled_blocks = list(reassembled_blocks or [2, 5, 8, 11])
        self.cross_frame_blocks = list(cross_frame_blocks or [5, 7, 9, 11])
        self.dims_reassemble = list(dims_reassemble or [48, 96, 192, 384])
        self.dim_decoder = dim_decoder
        self.precision = precision

    @classmethod
    def from_variant(cls, variant_id: str) -> "DepthPro2Config":
        if variant_id not in _DEPTHPRO2_VARIANT_MAP:
            raise ValueError(
                f"Unknown Depth Pro 2 variant '{variant_id}'. "
                f"Available: {list(_DEPTHPRO2_VARIANT_MAP.keys())}"
            )
        return cls(**_DEPTHPRO2_VARIANT_MAP[variant_id])
