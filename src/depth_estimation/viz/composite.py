"""
Composite visualization functions.

overlay_depth  — blend depth colormap over an RGB image
create_anaglyph — red-cyan stereoscopic anaglyph from monocular depth
"""

import cv2
import numpy as np

from ..processing_utils import DepthProcessor


def overlay_depth(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "inferno",
) -> np.ndarray:
    """Blend a depth colormap over an RGB image.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        depth: ``(H, W)`` float32 depth map in [0, 1].
        alpha: Blending weight for the depth layer (0 = image only, 1 = depth only).
        colormap: Matplotlib colormap name.

    Returns:
        ``(H, W, 3)`` uint8 blended image.
    """
    colored = DepthProcessor._colorize(depth, colormap)
    blended = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)
    return blended


def create_anaglyph(
    image: np.ndarray,
    depth: np.ndarray,
    baseline: float = 0.065,
) -> np.ndarray:
    """Create a red-cyan anaglyph stereo image from monocular RGB + depth.

    The depth map is used to derive per-pixel horizontal disparity: near
    objects (higher depth values after inversion) appear with more parallax.
    The left-eye red channel comes from the original image; the right-eye
    cyan (green + blue) channels come from a horizontally shifted version.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        depth: ``(H, W)`` float32 depth map in [0, 1].  Values closer to 1
               are treated as *farther* (metric convention). Near objects
               produce more disparity.
        baseline: Maximum horizontal shift as a fraction of image width.
                  Default 0.065 ≈ 6.5 % of width.

    Returns:
        ``(H, W, 3)`` uint8 anaglyph image.
    """
    H, W = depth.shape

    # Near objects → more disparity. Invert depth so near=1, far=0.
    disparity = ((1.0 - depth) * baseline * W).astype(np.float32)

    # Build remap grids for the right-eye (shifted rightward)
    col_idx = np.arange(W, dtype=np.float32)
    row_idx = np.arange(H, dtype=np.float32)
    map_x, map_y = np.meshgrid(col_idx, row_idx)
    map_x_right = np.clip(map_x + disparity, 0, W - 1).astype(np.float32)
    map_y_right = map_y.astype(np.float32)

    right_eye = cv2.remap(image, map_x_right, map_y_right, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    # Left eye: grayscale in red channel; right eye: color in green+blue
    left_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    anaglyph = np.zeros_like(image)
    anaglyph[:, :, 0] = left_gray           # red channel ← left eye
    anaglyph[:, :, 1] = right_eye[:, :, 1]  # green channel ← right eye
    anaglyph[:, :, 2] = right_eye[:, :, 2]  # blue channel ← right eye
    return anaglyph
