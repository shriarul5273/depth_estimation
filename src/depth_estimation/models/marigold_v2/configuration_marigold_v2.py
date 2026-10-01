"""
Marigold V2 configuration.
"""

from ...configuration_utils import BaseDepthConfig


_MARIGOLD_V2_VARIANT_MAP = {
    # Default: log-space affine-invariant depth (best quality)
    "marigold-v2": "depth/Log-base",
    # See-through log depth (predicts geometry behind glass)
    "marigold-v2-log-layered": "depth/Log-layered",
    # Linear (Marigold V1-style) depth
    "marigold-v2-uniform": "depth/Uniform-base",
    # Inverse depth / disparity
    "marigold-v2-disparity": "depth/Disparity-base",
}

# Per-checkpoint text-embedding prefix (used to select precomputed embeddings)
_CHECKPOINT_EMBED_PREFIX = {
    "depth/Log-base": "qwen_edit_2509_qwen_depth_realimg512",
    "depth/Log-layered": "qwen_edit_2509_qwen_depth_realimg512",
    "depth/Uniform-base": "qwen_edit_2509_qwen_depth_realimg512",
    "depth/Disparity-base": "qwen_edit_2509_qwen_depth_realimg512",
}


class MarigoldV2Config(BaseDepthConfig):
    """Configuration for Marigold V2.

    Single-step diffusion transformer depth estimator built on a
    Qwen-Image-Edit-2509 backbone fine-tuned with LoRA adapters.
    Outputs affine-invariant depth (log-space for the default checkpoint).

    Requires: ``diffusers>=0.38``, ``peft``, ``safetensors``, and optionally
    ``bitsandbytes`` for 4-bit memory-efficient inference.

    The Qwen-Image-Edit-2509 base model (~17 GB) must be accessible either as a
    HuggingFace Hub repo id (``qwen_hub_id``) or a local directory path.
    LoRA adapters and precomputed text embeddings are downloaded from
    ``huawei-bayerlab/marigold-v2-0`` on HuggingFace Hub.
    """

    model_type = "marigold-v2"

    def __init__(
        self,
        backbone: str = "marigold-v2",
        input_size: int = 1024,
        patch_size: int = 16,
        # Which LoRA checkpoint subfolder in hub_model_id to load
        checkpoint: str = "depth/Log-base",
        # HuggingFace repo for LoRA adapters + text embeddings
        hub_model_id: str = "huawei-bayerlab/marigold-v2-0",
        # HuggingFace repo (or local path) for the Qwen base model
        qwen_hub_id: str = "Qwen/Qwen-Image-Edit-2509",
        # Use 4-bit quantization (requires bitsandbytes + CUDA)
        quantize_4bit: bool = True,
        # LoRA configuration (must match the trained checkpoint)
        lora_rank: int = 128,
        lora_alpha: int = 128,
        seed: int = 2025,
        **kwargs,
    ):
        super().__init__(
            backbone=backbone,
            input_size=input_size,
            patch_size=patch_size,
            **kwargs,
        )
        self.checkpoint = checkpoint
        self.hub_model_id = hub_model_id
        self.qwen_hub_id = qwen_hub_id
        self.quantize_4bit = quantize_4bit
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.seed = seed

    @property
    def embed_prefix(self) -> str:
        return _CHECKPOINT_EMBED_PREFIX.get(
            self.checkpoint, "qwen_edit_2509_qwen_depth_realimg512"
        )

    @classmethod
    def from_variant(cls, variant_id: str) -> "MarigoldV2Config":
        if variant_id not in _MARIGOLD_V2_VARIANT_MAP:
            raise ValueError(
                f"Unknown Marigold V2 variant '{variant_id}'. "
                f"Available: {list(_MARIGOLD_V2_VARIANT_MAP.keys())}"
            )
        return cls(checkpoint=_MARIGOLD_V2_VARIANT_MAP[variant_id])
