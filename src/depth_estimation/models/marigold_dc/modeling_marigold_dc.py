"""
Marigold-DC — Single-file model implementation.

Marigold Depth Completion: extends MarigoldDepthPipeline from diffusers
for dense depth prediction guided by sparse depth measurements.

Requires: ``diffusers>=0.25``
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...modeling_utils import BaseDepthModel, _auto_detect_device
from .configuration_marigold_dc import MarigoldDCConfig, _MARIGOLD_DC_VARIANT_MAP

logger = logging.getLogger(__name__)


def _check_diffusers_available():
    try:
        from diffusers import MarigoldDepthPipeline, DDIMScheduler
        return True
    except ImportError:
        return False


class MarigoldDCModel(BaseDepthModel):
    """Marigold Depth Completion model.

    Extends MarigoldDepthPipeline with sparse depth guidance for
    dense metric depth prediction.

    Usage::

        model = MarigoldDCModel.from_pretrained("marigold-dc")

        # Standard depth estimation (no sparse guidance)
        depth = model(pixel_values)  # (B, H, W) tensor

        # With sparse depth guidance
        depth = model.predict_with_guidance(
            image_pil, sparse_depth_np, num_inference_steps=50
        )
    """

    config_class = MarigoldDCConfig

    def __init__(self, config: MarigoldDCConfig):
        super().__init__(config)
        self._pipe = None

    def _ensure_pipe(self):
        """Lazy-load the Marigold pipeline."""
        if self._pipe is not None:
            return
        if not _check_diffusers_available():
            raise ImportError(
                "Marigold-DC requires the `diffusers` package (>=0.25). "
                "Install with: pip install diffusers>=0.25"
            )
        from diffusers import MarigoldDepthPipeline, DDIMScheduler

        device = torch.device(_auto_detect_device())
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        pipe = MarigoldDepthPipeline.from_pretrained(
            self.config.hub_model_id, prediction_type="depth"
        ).to(device, dtype=dtype)
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        self._pipe = pipe

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass (standard Marigold depth estimation, no guidance).

        Args:
            pixel_values: Input tensor (B, 3, H, W), normalized.

        Returns:
            Depth tensor (B, H, W).
        """
        self._ensure_pipe()
        batch_size = pixel_values.shape[0]
        h, w = pixel_values.shape[2], pixel_values.shape[3]

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(pixel_values.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(pixel_values.device)

        depths = []
        for i in range(batch_size):
            img = pixel_values[i] * std + mean
            img = img.clamp(0, 1)
            img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Run Marigold pipeline (standard, no sparse guidance)
            output = self._pipe(
                pil_image,
                num_inference_steps=self.config.num_inference_steps,
                ensemble_size=self.config.ensemble_size,
                processing_resolution=self.config.processing_resolution,
            )

            depth_map = output.prediction.squeeze()
            if isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()

            depth_tensor = torch.from_numpy(depth_map).float()
            if depth_tensor.shape != (h, w):
                depth_tensor = F.interpolate(
                    depth_tensor.unsqueeze(0).unsqueeze(0),
                    size=(h, w), mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
            depths.append(depth_tensor)

        return torch.stack(depths)

    def predict_with_guidance(
        self,
        image: Image.Image,
        sparse_depth: np.ndarray,
        num_inference_steps: int = 50,
        ensemble_size: int = 1,
        processing_resolution: int = 768,
        seed: int = 2024,
    ) -> np.ndarray:
        """Run depth completion with sparse depth guidance.

        This is the core Marigold-DC functionality — uses sparse depth
        measurements to guide the diffusion process for metric depth.

        Args:
            image: Input PIL image.
            sparse_depth: Sparse depth guidance (H, W), zeros where missing.
            num_inference_steps: Number of denoising steps.
            ensemble_size: Number of ensemble members.
            processing_resolution: Processing resolution.
            seed: Random seed.

        Returns:
            Dense depth prediction as numpy array (H, W).
        """
        self._ensure_pipe()
        from diffusers import DDIMScheduler

        device = self._pipe._execution_device
        generator = torch.Generator(device=device).manual_seed(seed)

        pipe = self._pipe

        # Prepare empty text conditioning
        with torch.no_grad():
            if pipe.empty_text_embedding is None:
                text_inputs = pipe.tokenizer(
                    "", padding="do_not_pad",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                pipe.empty_text_embedding = pipe.text_encoder(text_input_ids)[0]

        # Preprocess
        image_t, padding, original_resolution = pipe.image_processor.preprocess(
            image, processing_resolution=processing_resolution,
            device=device, dtype=pipe.dtype,
        )

        # Encode image
        with torch.no_grad():
            image_latent, pred_latent = pipe.prepare_latents(
                image_t, None, generator, ensemble_size, 1
            )
        del image_t

        # Sparse depth setup
        sparse_depth_t = torch.from_numpy(sparse_depth)[None, None].float().to(device)
        sparse_mask = sparse_depth_t > 0
        sparse_range = (sparse_depth_t[sparse_mask].max() - sparse_depth_t[sparse_mask].min()).item()
        sparse_lower = sparse_depth_t[sparse_mask].min().item()

        def affine_to_metric(depth):
            return (scale ** 2) * sparse_range * depth + (shift ** 2) * sparse_lower

        def latent_to_metric(latent):
            pred = pipe.decode_prediction(latent)
            pred = affine_to_metric(pred)
            pred = pipe.image_processor.unpad_image(pred, padding)
            pred = pipe.image_processor.resize_antialias(
                pred, original_resolution, "bilinear", is_aa=False
            )
            return pred

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        ensemble_predictions = []
        for eidx in range(ensemble_size):
            cur_img_lat = image_latent[eidx:eidx + 1]
            cur_pred_lat = pred_latent[eidx:eidx + 1]

            scale = torch.nn.Parameter(torch.ones(1, device=device))
            shift = torch.nn.Parameter(torch.ones(1, device=device))
            cur_pred_lat = torch.nn.Parameter(cur_pred_lat)

            optimizer = torch.optim.Adam([
                {"params": [scale, shift], "lr": 0.005},
                {"params": [cur_pred_lat], "lr": 0.05},
            ])

            for _, t in enumerate(pipe.scheduler.timesteps):
                optimizer.zero_grad()

                batch_lat = torch.cat([cur_img_lat, cur_pred_lat], dim=1)
                noise = pipe.unet(
                    batch_lat, t,
                    encoder_hidden_states=pipe.empty_text_embedding,
                    return_dict=False,
                )[0]

                with torch.no_grad():
                    alpha_t = pipe.scheduler.alphas_cumprod[t]
                    beta_t = 1 - alpha_t
                    pred_eps = (alpha_t ** 0.5) * noise + (beta_t ** 0.5) * cur_pred_lat

                step_out = pipe.scheduler.step(noise, t, cur_pred_lat, generator=generator)
                pred_orig = step_out.pred_original_sample

                metric_est = latent_to_metric(pred_orig)
                loss = (
                    F.l1_loss(metric_est[sparse_mask], sparse_depth_t[sparse_mask])
                    + F.mse_loss(metric_est[sparse_mask], sparse_depth_t[sparse_mask])
                )
                loss.backward()

                with torch.no_grad():
                    eps_norm = torch.linalg.norm(pred_eps).item()
                    grad_norm = torch.linalg.norm(cur_pred_lat.grad).item()
                    cur_pred_lat.grad *= eps_norm / max(grad_norm, 1e-8)

                optimizer.step()

                with torch.no_grad():
                    cur_pred_lat.data = pipe.scheduler.step(
                        noise, t, cur_pred_lat, generator=generator
                    ).prev_sample

                del pred_orig, metric_est, step_out, pred_eps, noise
                torch.cuda.empty_cache()

            with torch.no_grad():
                prediction = latent_to_metric(cur_pred_lat.detach())
                ensemble_predictions.append(prediction)

        if ensemble_size > 1:
            ensemble_t = torch.cat(ensemble_predictions, dim=0)
            prediction = ensemble_t.median(dim=0, keepdim=True).values
        else:
            prediction = ensemble_predictions[0]

        result = pipe.image_processor.pt_to_numpy(prediction)
        return result.squeeze()

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "MarigoldDCModel":
        """Load Marigold-DC model."""
        config = MarigoldDCConfig()
        model = cls(config)
        model._ensure_pipe()

        logger.info(f"Loaded Marigold-DC from {config.hub_model_id}")
        return model
