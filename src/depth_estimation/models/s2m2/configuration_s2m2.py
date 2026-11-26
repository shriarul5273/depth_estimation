"""
S2M2 configuration.
"""

from typing import List, Optional

from ...configuration_utils import BaseDepthConfig


# Architecture presets, mirroring ``load_model()`` in junhong-3dv/s2m2
# (``src/s2m2/core/utils/model_utils.py``). Checkpoints are named
# ``CH{feature_channels}NTR{num_transformer}.pth`` in ``minimok/s2m2``.
_S2M2_VARIANT_MAP = {
    "s2m2-s": {"backbone": "s", "feature_channels": 128, "num_transformer": 1},
    "s2m2-m": {"backbone": "m", "feature_channels": 192, "num_transformer": 2},
    "s2m2-l": {"backbone": "l", "feature_channels": 256, "num_transformer": 3},
    "s2m2-xl": {"backbone": "xl", "feature_channels": 384, "num_transformer": 3},
}


class S2M2Config(BaseDepthConfig):
    """Configuration for S2M2 (Scalable Stereo Matching Model, ICCV 2025).

    S2M2 is a **stereo** model: it takes a rectified left/right image pair and
    predicts the left view's disparity, plus occlusion and confidence maps.
    Pass a pair to the pipeline as a tuple, ``pipe((left, right))``.

    Disparity is inverse depth, so ``forward()`` / ``pipeline()`` return it as
    a relative depth map (larger = closer), like Depth Anything. For metric
    depth, call ``model.infer(left, right, focal_length_px=..., baseline=...)``
    with the rectified camera's focal length and baseline.

    The network works on inputs that are multiples of 32, so ``patch_size`` is
    32 (the processor's resize multiple). ``input_size`` is the shorter side
    the processor resizes to; ``infer()`` instead runs at native resolution.

    Inputs are normalised to ``[-1, 1]`` (``mean = std = 0.5``), as upstream.

    ``precision`` is the autocast dtype: ``"auto"`` (float16 on CUDA, as the
    upstream demos run, float32 elsewhere), ``"float32"``, ``"bfloat16"`` or
    ``"float16"``.

    ``allow_negative`` permits negative disparity, for imperfectly rectified
    pairs (upstream ``--allow_negative``; off by default).

    ``refine_iter`` is the number of local iterative refinement steps
    (upstream ``--num_refine``, default 3).
    """

    model_type = "s2m2"

    def __init__(
        self,
        backbone: str = "s",
        input_size: int = 512,
        patch_size: int = 32,
        feature_channels: int = 128,
        num_transformer: int = 1,
        dim_expansion: int = 1,
        refine_iter: int = 3,
        allow_negative: bool = False,
        precision: str = "auto",
        hub_repo_id: str = "minimok/s2m2",
        hub_filename: Optional[str] = None,
        is_stereo: bool = True,
        is_metric: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            is_metric=is_metric,
            mean=mean or [0.5, 0.5, 0.5],
            std=std or [0.5, 0.5, 0.5],
            **kwargs,
        )
        self.feature_channels = feature_channels
        self.num_transformer = num_transformer
        self.dim_expansion = dim_expansion
        self.refine_iter = refine_iter
        self.allow_negative = allow_negative
        self.precision = precision
        self.hub_repo_id = hub_repo_id
        self.hub_filename = (
            hub_filename or f"CH{feature_channels}NTR{num_transformer}.pth"
        )
        self.is_stereo = is_stereo

    @classmethod
    def from_variant(cls, variant_id: str) -> "S2M2Config":
        if variant_id not in _S2M2_VARIANT_MAP:
            raise ValueError(
                f"Unknown S2M2 variant '{variant_id}'. "
                f"Available: {list(_S2M2_VARIANT_MAP.keys())}"
            )
        return cls(**_S2M2_VARIANT_MAP[variant_id])
