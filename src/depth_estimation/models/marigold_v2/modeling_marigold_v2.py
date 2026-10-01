"""
Marigold V2 — Single-file model implementation.

Single-step diffusion transformer depth estimation using a Qwen-Image-Edit-2509
backbone with LoRA adapters fine-tuned for monocular depth prediction.

Architecture:
  - Backbone: Qwen-Image-Edit-2509 (VAE + DiT transformer)
  - Adapters: LoRA (rank 128) on the DiT transformer
  - Inference: single rectified-flow step at t=0.499 with precomputed
    text-prompt embeddings (no text encoder needed at runtime)

Checkpoints available on HuggingFace at huawei-bayerlab/marigold-v2-0:
  - depth/Log-base        : affine-invariant log depth (default)
  - depth/Log-layered     : see-through log depth (glass/transparent objects)
  - depth/Uniform-base    : affine-invariant linear depth
  - depth/Disparity-base  : affine-invariant inverse depth / disparity

Requires:
  - diffusers>=0.38 (QwenImageEditPipeline, AutoencoderKLQwenImage,
    QwenImageTransformer2DModel)
  - peft>=0.18 (LoRA adapter loading)
  - safetensors>=0.4 (checkpoint loading)
  - bitsandbytes (optional; enables 4-bit quantization for ~17 GB VRAM
    reduction; falls back to bfloat16 without it)

Reference: https://github.com/huawei-bayerlab/marigold-v2
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from ...modeling_utils import BaseDepthModel
from .configuration_marigold_v2 import MarigoldV2Config

logger = logging.getLogger(__name__)

# LoRA target modules (proj_out excluded: it is dequantized to bf16 and its
# weights are stored as base weights in trainables.safetensors, not as LoRA)
_LORA_TARGET_MODULES = [
    "img_in",
    "txt_in",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_add_out",
    "norm.linear",
    "a_to_out",
    "b_to_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "norm_out.linear",
]


def _check_diffusers():
    try:
        import diffusers

        return (
            hasattr(diffusers, "QwenImageEditPipeline")
            and hasattr(diffusers, "AutoencoderKLQwenImage")
            and hasattr(diffusers, "QwenImageTransformer2DModel")
        )
    except ImportError:
        return False


def _check_peft():
    try:
        import peft  # noqa: F401

        return True
    except ImportError:
        return False


def _check_bitsandbytes():
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


class MarigoldV2Model(BaseDepthModel):
    """Marigold V2 depth estimation model.

    Wraps the Qwen-Image-Edit-2509 DiT + VAE with LoRA adapters for
    single-step monocular depth prediction.

    Usage::

        model = MarigoldV2Model.from_pretrained("marigold-v2")
        depth = model(pixel_values)   # (B, H, W) in [0, 1], affine-invariant
    """

    config_class = MarigoldV2Config
    # Wraps diffusers' Qwen pipeline components that contain non-traceable ops
    # (the DiT packs/unpacks latents with runtime-dependent shapes, the 4-bit
    # quantized linear layers are not torch.fx-traceable). ONNX export is not
    # meaningful here.
    _onnx_exportable = False

    def __init__(self, config: MarigoldV2Config):
        super().__init__(config)
        self._vae = None
        self._transformer = None
        self._prompt_embeds = None
        self._prompt_mask = None

    def _backbone_module(self):
        raise NotImplementedError(
            "MarigoldV2Model wraps a Qwen-Image-Edit DiT and does not expose "
            "trainable nn.Module parameters directly. Access self._transformer "
            "for the underlying PeftModel."
        )

    def _ensure_loaded(self):
        if self._vae is not None:
            return

        if not _check_diffusers():
            raise ImportError(
                "Marigold V2 requires diffusers>=0.38 with Qwen support. "
                "Install with: pip install 'diffusers>=0.38'"
            )
        if not _check_peft():
            raise ImportError(
                "Marigold V2 requires the peft package. "
                "Install with: pip install peft"
            )

        device = self.device
        use_4bit = self.config.quantize_4bit and _check_bitsandbytes() and device.type == "cuda"
        run_dtype = torch.bfloat16

        self._vae = self._load_vae(device, run_dtype)
        self._transformer = self._load_transformer(device, run_dtype, use_4bit)
        self._load_trainable_weights(self._vae, self._transformer)
        self._prompt_embeds, self._prompt_mask = self._load_text_embeddings(device)

    # ------------------------------------------------------------------
    # Component loaders
    # ------------------------------------------------------------------

    def _load_vae(self, device, dtype):
        from diffusers import AutoencoderKLQwenImage

        logger.info(f"Loading Qwen VAE from {self.config.qwen_hub_id}")
        vae = AutoencoderKLQwenImage.from_pretrained(
            self.config.qwen_hub_id,
            subfolder="vae",
            torch_dtype=dtype,
        )
        vae.requires_grad_(False)
        vae.to(device, dtype=dtype)
        return vae

    def _load_transformer(self, device, dtype, use_4bit: bool):
        from diffusers import QwenImageTransformer2DModel
        from peft import LoraConfig

        logger.info(
            f"Loading Qwen transformer from {self.config.qwen_hub_id} "
            f"(4-bit={'yes' if use_4bit else 'no'})"
        )
        load_kwargs: dict = dict(
            pretrained_model_name_or_path=self.config.qwen_hub_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        if use_4bit:
            from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

            load_kwargs["quantization_config"] = DiffusersBnbConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                # transformer_blocks.0.img_mod is kept in full precision to
                # avoid numerical instability at the first modulation step
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
            )

        transformer = QwenImageTransformer2DModel.from_pretrained(**load_kwargs)
        transformer.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=_LORA_TARGET_MODULES,
        )
        transformer.add_adapter(lora_cfg)
        transformer.to(device)
        return transformer

    def _load_trainable_weights(self, vae, transformer):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict

        filename = f"{self.config.checkpoint}/trainables.safetensors"
        logger.info(f"Downloading LoRA weights: {self.config.hub_model_id}/{filename}")
        path = hf_hub_download(self.config.hub_model_id, filename=filename)
        state_dict = load_file(path, device="cpu")

        # Separate LoRA (transformer) keys from base-weight keys (proj_out,
        # VAE decoder). proj_out was dequantized to bf16 during training and
        # stored as a plain weight, not a LoRA pair.
        lora_keys = {k: v for k, v in state_dict.items() if "lora_" in k}
        proj_out_keys = {k: v for k, v in state_dict.items() if "proj_out" in k and "lora_" not in k}
        vae_keys = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("decoder.") or k.startswith("post_quant_conv.")
        }

        if lora_keys:
            incompatible = set_peft_model_state_dict(
                transformer, lora_keys, adapter_name="default"
            )
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if unexpected:
                logger.warning(f"Unexpected LoRA keys: {unexpected[:5]}...")
            logger.info(f"Loaded {len(lora_keys)} LoRA weight tensors")

        if proj_out_keys:
            # Load fine-tuned proj_out weights directly into the base model
            base = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
            missing, unexpected = base.load_state_dict(proj_out_keys, strict=False)
            logger.info(f"Loaded {len(proj_out_keys)} proj_out weight tensors")

        if vae_keys:
            missing, _ = vae.load_state_dict(vae_keys, strict=False)
            logger.info(f"Loaded {len(vae_keys)} VAE decoder weight tensors")

    def _load_text_embeddings(self, device):
        import torch
        from huggingface_hub import hf_hub_download

        prefix = self.config.embed_prefix
        logger.info(f"Downloading text embeddings (prefix: {prefix})")

        embeds_path = hf_hub_download(
            self.config.hub_model_id,
            filename=f"qwen_text_embeddings/{prefix}_prompt_embeds.pt",
        )
        mask_path = hf_hub_download(
            self.config.hub_model_id,
            filename=f"qwen_text_embeddings/{prefix}_prompt_mask.pt",
        )
        prompt_embeds = torch.load(embeds_path, map_location="cpu", weights_only=False).contiguous()
        prompt_mask = torch.load(mask_path, map_location="cpu", weights_only=False).contiguous()
        if prompt_mask.dtype != torch.bool:
            prompt_mask = prompt_mask > 0
        return prompt_embeds, prompt_mask

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _latent_stats(vae, ref):
        shape = (1, vae.config.z_dim, 1, 1, 1)
        mean = torch.tensor(
            vae.config.latents_mean, device=ref.device, dtype=ref.dtype
        ).view(shape)
        std_inv = (
            1.0
            / torch.tensor(
                vae.config.latents_std, device=ref.device, dtype=ref.dtype
            ).view(shape)
        )
        return mean, std_inv

    def _encode_image(self, rgb_norm: torch.Tensor) -> torch.Tensor:
        """Encode [B, 3, H, W] in [-1, 1] to normalized Qwen VAE latents."""
        vae = self._vae
        x = rgb_norm.to(device=vae.device, dtype=vae.dtype)
        # Qwen VAE expects 5D: [B, C, T=1, H, W]
        out = vae.encode(x.unsqueeze(2)).latent_dist.sample()
        mean, std_inv = self._latent_stats(vae, out)
        return (out - mean) * std_inv

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents to [B, 3, H, W] in [-1, 1]."""
        vae = self._vae
        mean, std_inv = self._latent_stats(vae, latents)
        # Reverse: normalized = (out - mean) * std_inv  →  out = normalized / std_inv + mean
        lat = latents / std_inv + mean
        decoded = vae.decode(lat)
        # Drop temporal dim: [B, C, T, H, W] → [B, C, H, W]
        return decoded.sample[:, :, 0]

    def _run_dit_step(self, latents: torch.Tensor) -> torch.Tensor:
        """Run the single rectified-flow DiT step at t=0.499."""
        from diffusers import QwenImageEditPipeline

        transformer = self._transformer
        vae = self._vae

        # Flatten temporal dim if present
        if latents.ndim == 5 and latents.shape[2] == 1:
            model_input_4d = latents[:, :, 0]
        else:
            model_input_4d = latents

        B, C, H, W = model_input_4d.shape
        device = next(transformer.parameters()).device
        run_dtype = torch.bfloat16

        # Replicate prompt embeddings to match batch size
        pe = self._prompt_embeds
        pm = self._prompt_mask
        if pe.size(0) != B:
            pe = pe[:1].expand(B, *pe.shape[1:]) if B > 1 else pe[:B]
        if pm.size(0) != B:
            pm = pm[:1].expand(B, *pm.shape[1:]) if B > 1 else pm[:B]
        pe = pe.to(device=device, dtype=run_dtype)
        pm = pm.to(device=device, dtype=torch.bool)

        packed = QwenImageEditPipeline._pack_latents(
            model_input_4d.to(run_dtype),
            batch_size=B,
            num_channels_latents=C,
            height=H,
            width=W,
        )
        timestep = torch.full((B,), 499.0, device=device, dtype=run_dtype) / 1000.0
        img_shapes = [[(1, H // 2, W // 2)]] * B
        txt_seq_lens = pm.sum(dim=1).tolist()

        with torch.no_grad():
            model_pred = transformer(
                hidden_states=packed,
                timestep=timestep,
                encoder_hidden_states=pe,
                encoder_hidden_states_mask=pm,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                guidance=None,
                return_dict=False,
            )[0]

        temporal_downsample = getattr(vae.config, "temperal_downsample", None)
        vae_scale_factor = 2 ** len(temporal_downsample) if temporal_downsample else 8

        model_pred = QwenImageEditPipeline._unpack_latents(
            model_pred,
            height=H * vae_scale_factor,
            width=W * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        # Velocity prediction: output_latent = input_latent - velocity
        return latents.to(vae.dtype) - model_pred.to(vae.dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run depth estimation.

        Args:
            pixel_values: Input tensor (B, 3, H, W), ImageNet-normalized.

        Returns:
            Depth tensor (B, H, W) in [0, 1]. Values are affine-invariant
            (up to an unknown scale and shift per image).
        """
        self._ensure_loaded()

        # Denormalize from ImageNet to [0, 1], then rescale to [-1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(3, 1, 1)

        B, _, H, W = pixel_values.shape

        results = []
        for i in range(B):
            img_01 = (pixel_values[i] * std + mean).clamp(0, 1)
            rgb_norm = img_01 * 2.0 - 1.0  # [-1, 1]

            with torch.no_grad():
                lat = self._encode_image(rgb_norm.unsqueeze(0))
                lat_out = self._run_dit_step(lat)
                pixel_pred = self._decode_latents(lat_out)  # [1, 3, H_enc, W_enc]

            # Average 3 output channels → depth in [-1, 1]
            depth = pixel_pred.mean(dim=1)  # [1, H_enc, W_enc]

            # Resize back to input spatial size if VAE changed it
            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(
                    depth.unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Map from [-1, 1] to [0, 1]
            depth = ((depth + 1.0) * 0.5).clamp(0.0, 1.0)
            results.append(depth.squeeze(0))

        return torch.stack(results)  # [B, H, W]

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "MarigoldV2Model":
        config = MarigoldV2Config.from_variant(model_id)
        model = cls(config)
        model = model.to(device)
        model._ensure_loaded()
        logger.info(f"Loaded Marigold V2 ({config.checkpoint}) from {config.hub_model_id}")
        return model
