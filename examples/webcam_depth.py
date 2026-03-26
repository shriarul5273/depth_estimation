"""
Video depth estimation using Depth Anything V3 Small.

Reads examples/video.mp4 and saves a side-by-side RGB | depth video
to examples/video_depth.mp4.

Usage:
    python examples/webcam_depth.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID   = "depth-anything-v3-small"
INPUT      = os.path.join(os.path.dirname(__file__), "video.mp4")
OUTPUT     = os.path.join(os.path.dirname(__file__), "video_depth.mp4")
COLORMAP   = "inferno"
SMOOTHING  = 0.4   # EMA coefficient — 0.0 = off, 0.9 = heavy
SIDE_BY_SIDE = True
# ──────────────────────────────────────────────────────────────────────────────


def _colorize(depth_norm: np.ndarray, colormap: str) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(colormap)
    return (cmap((depth_norm * 255).astype(np.uint8))[:, :, :3] * 255).astype(np.uint8)


def main():
    print(f"Loading {MODEL_ID} …")
    from depth_estimation import pipeline
    pipe = pipeline("depth-estimation", model=MODEL_ID)
    print("Model ready.\n")

    cap = cv2.VideoCapture(INPUT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {INPUT}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w        = W * 2 if SIDE_BY_SIDE else W
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    writer       = cv2.VideoWriter(OUTPUT, fourcc, fps, (out_w, H))

    print(f"Input  : {INPUT}  ({W}x{H}, {total} frames @ {fps:.1f} fps)")
    print(f"Output : {OUTPUT}")

    prev_depth = None

    for _ in tqdm(range(total), desc="Processing", unit="frame"):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result    = pipe(frame_rgb, colorize=False)
        depth     = result.depth  # (H, W) float32 [0, 1]

        # EMA temporal smoothing
        if SMOOTHING > 0.0 and prev_depth is not None:
            depth = SMOOTHING * prev_depth + (1.0 - SMOOTHING) * depth
        prev_depth = depth

        colored_bgr = cv2.cvtColor(_colorize(depth, COLORMAP), cv2.COLOR_RGB2BGR)

        out_frame = np.concatenate([frame_bgr, colored_bgr], axis=1) if SIDE_BY_SIDE else colored_bgr
        writer.write(out_frame)

    cap.release()
    writer.release()
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
