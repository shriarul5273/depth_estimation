"""Standard depth estimation metrics and per-dataset accumulator.

All seven metrics follow the protocol established by Eigen et al. (NeurIPS
2014) [1] and further standardised in Eigen & Fergus (ICCV 2015) [2].  They
are used in virtually every monocular depth estimation paper.

References
----------
[1] Eigen, D., Puhrsch, C., & Fergus, R. (2014).
    Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.
    NeurIPS 2014. https://arxiv.org/abs/1406.2283

[2] Eigen, D., & Fergus, R. (2015).
    Predicting Depth, Surface Normals and Semantic Labels with a Common
    Multi-Scale Convolutional Architecture.
    ICCV 2015. https://arxiv.org/abs/1411.4734

[3] Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2022).
    Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot
    Cross-Dataset Transfer. IEEE TPAMI.
    https://arxiv.org/abs/1907.01341
    (establishes least-squares scale-and-shift alignment for relative models)

Per-sample computation
----------------------
Use :class:`DepthMetrics` to compute all 7 metrics for a single prediction::

    metrics = DepthMetrics()
    result = metrics(pred, target, valid_mask)
    # {"abs_rel": ..., "sq_rel": ..., "rmse": ..., "rmse_log": ...,
    #  "delta1": ..., "delta2": ..., "delta3": ...}

Dataset-level accumulation
---------------------------
Use :class:`Evaluator` to accumulate metrics across batches so that RMSE is
computed correctly (sqrt of the mean squared error over *all* valid pixels,
not the mean of per-batch RMSEs)::

    ev = Evaluator()
    for pred_batch, gt_batch, mask_batch in dataloader:
        ev.update(pred_batch, gt_batch, mask_batch)
    results = ev.compute()
    ev.reset()
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Alignment helper
# ---------------------------------------------------------------------------

def align_least_squares(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """Compute the least-squares scale and shift that maps *pred* onto *target*.

    Solves:  ``target[mask] ≈ scale * pred[mask] + shift``

    Args:
        pred:   Predicted depth, shape ``(H, W)``, any scale.
        target: Ground-truth depth, shape ``(H, W)``, in metres.
        mask:   Boolean validity mask, shape ``(H, W)``.

    Returns:
        ``(scale, shift)`` such that ``scale * pred + shift`` best approximates
        ``target`` in the least-squares sense.
    """
    p = pred[mask].astype(np.float64)
    g = target[mask].astype(np.float64)

    if len(p) < 2:
        return 1.0, 0.0

    # Normal equations: [[Σp², Σp], [Σp, n]] * [s, t]ᵀ = [Σp·g, Σg]ᵀ
    A = np.array([[np.dot(p, p), p.sum()],
                  [p.sum(),      len(p)]])
    b = np.array([np.dot(p, g), g.sum()])

    try:
        scale, shift = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        scale, shift = 1.0, 0.0

    return float(scale), float(shift)


# ---------------------------------------------------------------------------
# Per-prediction metrics
# ---------------------------------------------------------------------------

class DepthMetrics:
    """Compute all 7 standard depth estimation metrics for one prediction.

    Metrics follow Eigen et al., NeurIPS 2014 [1] and Eigen & Fergus,
    ICCV 2015 [2].

    =========  ==============================  ==========  ==============
    Name       Formula (over valid pixels)     Range       Direction
    =========  ==============================  ==========  ==============
    abs_rel    mean(|d - d̂| / d)              [0, ∞)      lower ↓
    sq_rel     mean((d - d̂)² / d)             [0, ∞)      lower ↓
    rmse       √mean((d - d̂)²)               [0, ∞)      lower ↓
    rmse_log   √mean((log d - log d̂)²)        [0, ∞)      lower ↓
    delta1     % pixels: max(d/d̂, d̂/d) < 1.25    [0, 1]  higher ↑
    delta2     same, threshold 1.25²           [0, 1]      higher ↑
    delta3     same, threshold 1.25³           [0, 1]      higher ↑
    =========  ==============================  ==========  ==============

    Args:
        eps: Minimum depth for log-space stability. Default ``1e-6``.

    Example::

        metrics = DepthMetrics()
        result = metrics(pred, target, valid_mask)
        print(result["abs_rel"], result["delta1"])
    """

    # Metrics where lower is better
    LOWER_IS_BETTER = {"abs_rel", "sq_rel", "rmse", "rmse_log"}
    # Metrics where higher is better
    HIGHER_IS_BETTER = {"delta1", "delta2", "delta3"}

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute metrics for one batch element or a full batch.

        Args:
            pred:       Predicted depth, shape ``(H, W)`` or ``(B, H, W)``
                        or ``(B, 1, H, W)``, float32, positive values.
            target:     Ground-truth depth, same shape as *pred*, in metres.
            valid_mask: Boolean mask, same shape as *pred*.  When ``None``,
                        all pixels with ``target > eps`` are treated as valid.

        Returns:
            Dict with keys ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``,
            ``delta1``, ``delta2``, ``delta3``.  All values are Python floats.
        """
        pred = pred.squeeze().float()
        target = target.squeeze().float()

        if valid_mask is None:
            valid_mask = target > self.eps
        else:
            valid_mask = valid_mask.squeeze().bool() & (target > self.eps)

        # Flatten to 1-D for vectorised ops
        p = pred[valid_mask].clamp(min=self.eps)
        t = target[valid_mask].clamp(min=self.eps)

        if p.numel() == 0:
            return {k: 0.0 for k in
                    ("abs_rel", "sq_rel", "rmse", "rmse_log",
                     "delta1", "delta2", "delta3")}

        diff = p - t
        thresh = torch.maximum(p / t, t / p)

        abs_rel  = ((diff.abs()) / t).mean().item()
        sq_rel   = ((diff ** 2) / t).mean().item()
        rmse     = diff.pow(2).mean().sqrt().item()
        rmse_log = (p.log() - t.log()).pow(2).mean().sqrt().item()
        delta1   = (thresh < 1.25     ).float().mean().item()
        delta2   = (thresh < 1.25 ** 2).float().mean().item()
        delta3   = (thresh < 1.25 ** 3).float().mean().item()

        return {
            "abs_rel":  abs_rel,
            "sq_rel":   sq_rel,
            "rmse":     rmse,
            "rmse_log": rmse_log,
            "delta1":   delta1,
            "delta2":   delta2,
            "delta3":   delta3,
        }


# ---------------------------------------------------------------------------
# Dataset-level accumulator
# ---------------------------------------------------------------------------

class Evaluator:
    """Accumulate depth metrics across batches for correct dataset-level RMSE.

    Computing RMSE as the mean of per-batch RMSEs is biased when batches have
    different numbers of valid pixels.  This class instead accumulates the
    *sum of squared errors* and divides by the *total* valid pixel count at
    the end before taking the square root.

    Args:
        eps: Minimum depth for log-space stability. Default ``1e-6``.

    Example::

        ev = Evaluator()
        for pred, target, mask in loader:
            ev.update(pred, target, mask)
        results = ev.compute()
        # {"abs_rel": 0.043, "rmse": 0.312, "delta1": 0.982, ...}
        ev.reset()
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._sum_abs_rel  = 0.0
        self._sum_sq_rel   = 0.0
        self._sum_sq_err   = 0.0   # for RMSE: Σ(d - d̂)²
        self._sum_sq_log   = 0.0   # for RMSE-log: Σ(log d - log d̂)²
        self._sum_delta1   = 0.0
        self._sum_delta2   = 0.0
        self._sum_delta3   = 0.0
        self._n_pixels     = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate metrics for one batch.

        Args:
            pred:       Predicted depth ``(B, H, W)`` or ``(B, 1, H, W)``.
            target:     Ground-truth depth, same shape, in metres.
            valid_mask: Boolean mask, same shape. Defaults to ``target > eps``.
        """
        pred   = pred.squeeze(1).float()    if pred.dim()   == 4 else pred.float()
        target = target.squeeze(1).float()  if target.dim() == 4 else target.float()

        if valid_mask is None:
            mask = target > self.eps
        else:
            mask = valid_mask.squeeze(1).bool() if valid_mask.dim() == 4 else valid_mask.bool()
            mask = mask & (target > self.eps)

        p = pred[mask].clamp(min=self.eps)
        t = target[mask].clamp(min=self.eps)
        n = p.numel()

        if n == 0:
            return

        diff   = p - t
        thresh = torch.maximum(p / t, t / p)

        self._sum_abs_rel += (diff.abs() / t).sum().item()
        self._sum_sq_rel  += ((diff ** 2) / t).sum().item()
        self._sum_sq_err  += diff.pow(2).sum().item()
        self._sum_sq_log  += (p.log() - t.log()).pow(2).sum().item()
        self._sum_delta1  += (thresh < 1.25     ).float().sum().item()
        self._sum_delta2  += (thresh < 1.25 ** 2).float().sum().item()
        self._sum_delta3  += (thresh < 1.25 ** 3).float().sum().item()
        self._n_pixels    += n

    def compute(self) -> dict:
        """Return aggregated metrics over all :meth:`update` calls.

        Returns:
            Dict with keys ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``,
            ``delta1``, ``delta2``, ``delta3``, ``n_pixels``.
        """
        n = self._n_pixels
        if n == 0:
            return {k: 0.0 for k in
                    ("abs_rel", "sq_rel", "rmse", "rmse_log",
                     "delta1", "delta2", "delta3", "n_pixels")}

        return {
            "abs_rel":  self._sum_abs_rel / n,
            "sq_rel":   self._sum_sq_rel  / n,
            "rmse":     math.sqrt(self._sum_sq_err  / n),
            "rmse_log": math.sqrt(self._sum_sq_log  / n),
            "delta1":   self._sum_delta1  / n,
            "delta2":   self._sum_delta2  / n,
            "delta3":   self._sum_delta3  / n,
            "n_pixels": n,
        }

    @property
    def n_pixels(self) -> int:
        """Total number of valid pixels accumulated so far."""
        return self._n_pixels
