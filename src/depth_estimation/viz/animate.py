"""
3D animation from depth maps.

animate_3d — rotating 3D surface plot exported as GIF or MP4
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_3d(
    image: np.ndarray,
    depth: np.ndarray,
    output_path: str,
    frames: int = 60,
    elevation: float = 20.0,
    fps: int = 15,
) -> None:
    """Produce a rotating 3D surface animation from a depth map textured with RGB colors.

    The output format is inferred from the file extension:
    - ``.gif``  — uses ``PillowWriter`` (no system dependencies).
    - ``.mp4``  — uses ``FFMpegWriter`` (requires ``ffmpeg`` on PATH).

    The depth map is downsampled to a ~128×128 grid for performance; full-
    resolution matplotlib surfaces are too slow for animation.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        depth: ``(H, W)`` float32 depth map in [0, 1].
        output_path: Destination file path (.gif or .mp4).
        frames: Number of rotation frames (azimuth sweeps 0→360°).
        elevation: Fixed elevation angle in degrees.
        fps: Frames per second of the output animation.

    Raises:
        ValueError: If the output extension is not ``.gif`` or ``.mp4``.
    """
    ext = Path(output_path).suffix.lower()
    if ext not in {".gif", ".mp4"}:
        raise ValueError(f"output_path must end in .gif or .mp4, got {ext!r}")

    H, W = depth.shape
    step = max(1, min(H, W) // 128)

    Z = depth[::step, ::step]
    h_, w_ = Z.shape
    X, Y = np.meshgrid(np.arange(w_), np.arange(h_))

    # Texture colors: downsample and convert to float
    colors = image[::step, ::step].astype(np.float32) / 255.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # facecolors must match the number of *quads* (one less than vertices per dim)
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=colors[:-1, :-1],
        rstride=1, cstride=1,
        shade=False,
        antialiased=False,
    )

    ax.set_axis_off()
    ax.view_init(elev=elevation, azim=0)

    def _update(frame_idx):
        ax.view_init(elev=elevation, azim=frame_idx * 360.0 / frames)
        return (surf,)

    anim = FuncAnimation(fig, _update, frames=frames, blit=False, interval=1000 // fps)

    if ext == ".gif":
        from matplotlib.animation import PillowWriter
        anim.save(output_path, writer=PillowWriter(fps=fps))
    else:
        from matplotlib.animation import FFMpegWriter
        anim.save(output_path, writer=FFMpegWriter(fps=fps))

    plt.close(fig)
