"""
Display functions for depth maps.

show_depth    — single depth map, interactive or saved
compare_depths — side-by-side subplot grid for multiple results
"""

from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from ..output import DepthOutput
from ..processing_utils import DepthProcessor


def show_depth(
    result: Union[DepthOutput, np.ndarray],
    colormap: str = "Spectral_r",
    title: Optional[str] = None,
    save: Optional[str] = None,
) -> None:
    """Display a single depth map.

    Args:
        result: A ``DepthOutput`` or a raw ``(H, W)`` float32 array in [0, 1].
        colormap: Matplotlib colormap name (used only when result has no colored_depth).
        title: Optional figure title.
        save: If given, save the figure to this path instead of calling plt.show().
    """
    if isinstance(result, DepthOutput):
        if result.colored_depth is not None:
            arr = result.colored_depth
        else:
            arr = DepthProcessor._colorize(result.depth, colormap)
    elif isinstance(result, np.ndarray):
        arr = DepthProcessor._colorize(result, colormap)
    else:
        raise TypeError(f"Expected DepthOutput or np.ndarray, got {type(result)}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(arr)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close(fig)


def compare_depths(
    results: List[Union[DepthOutput, np.ndarray]],
    labels: Optional[List[str]] = None,
    colormap: str = "Spectral_r",
    save: Optional[str] = None,
) -> None:
    """Display multiple depth maps side by side.

    Args:
        results: List of ``DepthOutput`` or ``(H, W)`` float32 arrays.
        labels: Optional list of subplot titles. Must match len(results) if given.
        colormap: Matplotlib colormap name.
        save: If given, save the figure to this path instead of calling plt.show().

    Raises:
        ValueError: If ``labels`` is provided but its length differs from ``results``.
    """
    if not results:
        raise ValueError("results must be a non-empty list.")
    if labels is not None and len(labels) != len(results):
        raise ValueError(
            f"len(labels)={len(labels)} must match len(results)={len(results)}"
        )

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (res, ax) in enumerate(zip(results, axes)):
        if isinstance(res, DepthOutput):
            arr = res.colored_depth if res.colored_depth is not None else DepthProcessor._colorize(res.depth, colormap)
        elif isinstance(res, np.ndarray):
            arr = DepthProcessor._colorize(res, colormap)
        else:
            raise TypeError(f"results[{i}]: expected DepthOutput or np.ndarray, got {type(res)}")

        ax.imshow(arr)
        ax.axis("off")
        if labels:
            ax.set_title(labels[i])

    plt.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close(fig)
