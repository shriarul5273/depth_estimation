"""
Error visualization for depth predictions.

plot_error_map — per-pixel error heatmap between predicted and ground-truth depth
"""

from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt


_VALID_METRICS = ("abs_rel", "sq_rel", "log10", "rmse")


def plot_error_map(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    metric: Literal["abs_rel", "sq_rel", "log10", "rmse"] = "abs_rel",
    save: Optional[str] = None,
) -> None:
    """Render a per-pixel error heatmap between predicted and ground-truth depth.

    Invalid ground-truth pixels (gt == 0) are shown as blank (NaN).

    Args:
        pred_depth: ``(H, W)`` float32 predicted depth map.
        gt_depth:   ``(H, W)`` float32 ground-truth depth map. Zeros are treated
                    as invalid.
        metric: Error formula to use.  One of:
                ``"abs_rel"``  → |pred - gt| / gt
                ``"sq_rel"``   → (pred - gt)² / gt
                ``"log10"``    → |log10(pred) - log10(gt)|
                ``"rmse"``     → (pred - gt)²  (visualises squared error per pixel)
        save: If given, save the figure to this path; otherwise call plt.show().

    Raises:
        ValueError: If ``metric`` is not one of the supported values.
    """
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"metric must be one of {_VALID_METRICS}, got {metric!r}"
        )

    valid = gt_depth > 0
    error = np.full_like(pred_depth, np.nan, dtype=np.float32)

    p = pred_depth[valid]
    g = gt_depth[valid]

    if metric == "abs_rel":
        error[valid] = np.abs(p - g) / g
    elif metric == "sq_rel":
        error[valid] = (p - g) ** 2 / g
    elif metric == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            error[valid] = np.abs(np.log10(np.maximum(p, 1e-8)) - np.log10(np.maximum(g, 1e-8)))
    elif metric == "rmse":
        error[valid] = (p - g) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(error, cmap="hot")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Per-pixel error: {metric}")
    ax.axis("off")
    plt.tight_layout()

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close(fig)
