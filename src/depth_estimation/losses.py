"""
Depth estimation loss functions.

All losses accept depth tensors of shape (B, H, W) or (B, 1, H, W) and an
optional boolean valid_mask of the same shape. Pixels outside the mask are
excluded from every computation.

No imports from within the depth_estimation package — only torch and torch.nn.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class ScaleInvariantLoss(nn.Module):
    """Scale-invariant logarithmic depth loss (SI-log).

    Penalises the log-difference between prediction and ground truth while
    being invariant to global scale.

    Formula (per sample)::

        L_si = mean(d_i^2) - lam * mean(d_i)^2
        where d_i = log(pred_i) - log(target_i)  over valid pixels

    Args:
        lam: Variance weight (lambda). Default 0.85.
        eps: Minimum depth value for numerical stability. Default 1e-8.
    """

    def __init__(self, lam: float = 0.85, eps: float = 1e-8):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute SI-log loss.

        Args:
            pred:       Predicted depth (B, H, W) or (B, 1, H, W), positive.
            target:     Ground-truth depth, same shape.
            valid_mask: Boolean mask, same shape. Defaults to ``target > eps``.

        Returns:
            Scalar loss tensor.
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        if valid_mask is None:
            valid_mask = target > self.eps
        else:
            valid_mask = valid_mask.squeeze(1).bool() & (target > self.eps)

        pred_log = torch.log(pred.clamp(min=self.eps))
        target_log = torch.log(target.clamp(min=self.eps))
        diff = pred_log - target_log  # (B, H, W)

        batch_losses = []
        for b in range(pred.shape[0]):
            mask_b = valid_mask[b]
            n = mask_b.sum().float()
            if n < 1:
                batch_losses.append(pred.new_zeros(()))
                continue
            d = diff[b][mask_b]
            loss_b = (d ** 2).mean() - self.lam * (d.mean() ** 2)
            batch_losses.append(loss_b)

        return torch.stack(batch_losses).mean()


class GradientLoss(nn.Module):
    """Image-gradient loss in log-depth space.

    Encourages predicted depth edges to align with ground-truth edges.
    Computes L1 loss on x- and y-direction finite differences of log-depth.

    Args:
        eps: Minimum depth for log stability. Default 1e-8.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient loss.

        Args:
            pred:       Predicted depth (B, H, W) or (B, 1, H, W).
            target:     Ground-truth depth, same shape.
            valid_mask: Boolean mask, same shape.

        Returns:
            Scalar loss tensor.
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        if valid_mask is None:
            valid_mask = target > self.eps
        else:
            valid_mask = valid_mask.squeeze(1).bool() & (target > self.eps)

        log_pred = torch.log(pred.clamp(min=self.eps))
        log_target = torch.log(target.clamp(min=self.eps))

        # x-direction gradient
        pred_dx = log_pred[:, :, 1:] - log_pred[:, :, :-1]
        target_dx = log_target[:, :, 1:] - log_target[:, :, :-1]
        mask_dx = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]

        # y-direction gradient
        pred_dy = log_pred[:, 1:, :] - log_pred[:, :-1, :]
        target_dy = log_target[:, 1:, :] - log_target[:, :-1, :]
        mask_dy = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

        n_dx = mask_dx.sum()
        n_dy = mask_dy.sum()

        loss_x = (pred_dx - target_dx).abs()[mask_dx].sum() / n_dx.clamp(min=1).float()
        loss_y = (pred_dy - target_dy).abs()[mask_dy].sum() / n_dy.clamp(min=1).float()

        return loss_x + loss_y


class BerHuLoss(nn.Module):
    """Reverse Huber (BerHu) loss.

    Behaves like L1 for small errors and L2 for large errors.
    More robust to outliers than pure L2. Standard for metric depth (KITTI).

    Formula::

        berhu(x) = |x|                       if |x| <= c
                   (x^2 + c^2) / (2 * c)    if |x| > c
        c = 0.2 * max(|pred - target|)  per batch over valid pixels
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute BerHu loss.

        Args:
            pred:       Predicted depth (B, H, W) or (B, 1, H, W).
            target:     Ground-truth depth, same shape.
            valid_mask: Boolean mask, same shape.

        Returns:
            Scalar loss tensor.
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        if valid_mask is None:
            valid_mask = target > 1e-8
        else:
            valid_mask = valid_mask.squeeze(1).bool()

        err = (pred - target).abs()
        valid_err = err[valid_mask]

        if valid_err.numel() == 0:
            return pred.new_zeros(())

        c = (0.2 * valid_err.max()).clamp(min=1e-8)

        l1_region = valid_err[valid_err <= c]
        l2_region = valid_err[valid_err > c]

        loss_l1 = l1_region.sum() if l1_region.numel() > 0 else pred.new_zeros(())
        loss_l2 = ((l2_region ** 2 + c ** 2) / (2.0 * c)).sum() if l2_region.numel() > 0 else pred.new_zeros(())

        n = valid_err.numel()
        return (loss_l1 + loss_l2) / n


class CombinedDepthLoss(nn.Module):
    """Weighted combination of SI-log and gradient losses.

    The default combination used by :class:`~depth_estimation.trainer.DepthTrainer`.

    Args:
        si_weight:   Weight for :class:`ScaleInvariantLoss`. Default 1.0.
        grad_weight: Weight for :class:`GradientLoss`. Default 0.5.
        lam:         SI-log variance weight (lambda). Default 0.85.

    Example::

        loss_fn = CombinedDepthLoss()
        pred   = torch.rand(2, 480, 640).clamp(min=1e-3)
        target = torch.rand(2, 480, 640).clamp(min=1e-3)
        losses = loss_fn(pred, target)
        losses["loss"].backward()
    """

    def __init__(
        self,
        si_weight: float = 1.0,
        grad_weight: float = 0.5,
        lam: float = 0.85,
    ):
        super().__init__()
        self.si_weight = si_weight
        self.grad_weight = grad_weight
        self.si_loss = ScaleInvariantLoss(lam=lam)
        self.grad_loss = GradientLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred:       Predicted depth (B, H, W) or (B, 1, H, W).
            target:     Ground-truth depth, same shape.
            valid_mask: Boolean mask, same shape. Optional.

        Returns:
            Dict with keys ``"loss"`` (total), ``"si_loss"``, ``"grad_loss"``.
        """
        si = self.si_loss(pred, target, valid_mask)
        grad = self.grad_loss(pred, target, valid_mask)
        total = self.si_weight * si + self.grad_weight * grad
        return {"loss": total, "si_loss": si, "grad_loss": grad}
