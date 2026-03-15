"""
Demo: Run inference on ALL models in the depth_estimation package.

Saves colored depth maps to examples/demo_results/<model_name>.png
Uses demo.png as input image.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

# ── Configuration ──────────────────────────────────────────────
INPUT_IMAGE = os.path.join(os.path.dirname(__file__), "demo.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Per-model kwargs forwarded to pipeline() → model.from_pretrained()
MODEL_KWARGS = {}

# All model variant IDs to try
ALL_MODELS = [
    # # Depth Anything v1
    # "depth-anything-v1-vits",
    # "depth-anything-v1-vitb",
    # "depth-anything-v1-vitl",
    # # Depth Anything v2
    "depth-anything-v2-vits",
    # "depth-anything-v2-vitb",
    # "depth-anything-v2-vitl",
    # # Depth Anything v3
    # "depth-anything-v3-small",
    # "depth-anything-v3-base",
    # "depth-anything-v3-large",
    # "depth-anything-v3-giant",
    # "depth-anything-v3-nested-giant-large",
    # "depth-anything-v3-metric-large",
    # "depth-anything-v3-mono-large",
    # ZoeDepth
    # "zoedepth",
    # MiDaS
    # "midas-dpt-large",
    # "midas-dpt-hybrid",
    # "midas-beit-large",
    # DepthPro
    # "depth-pro",
    # Pixel-Perfect Depth
    # "pixel-perfect-depth",
    # Marigold-DC
    # "marigold-dc",
    # MoGe
    # "moge-v1",
    # "moge-v2-vitl",
    # "moge-v2-vitl-normal",
    # "moge-v2-vitb-normal",
    # "moge-v2-vits-normal",
    # OmniVGGT
    # "omnivggt",
    # VGGT
    # "vggt",
    # "vggt-commercial",
]


def run_inference(model_id: str, image_path: str, output_dir: str):
    """Run inference with a single model and save the result."""
    from depth_estimation import pipeline

    print(f"\n{'='*60}")
    print(f"  Model: {model_id}")
    print(f"{'='*60}")

    try:
        t0 = time.time()
        kwargs = MODEL_KWARGS.get(model_id, {})
        pipe = pipeline("depth-estimation", model=model_id, **kwargs)
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.2f}s")

        t1 = time.time()
        result = pipe(image_path)
        infer_time = time.time() - t1
        print(f"  Inference in {infer_time:.2f}s")

        # Save colored depth map
        if result.colored_depth is not None:
            out_path = os.path.join(output_dir, f"{model_id}.png")
            cv2.imwrite(out_path, cv2.cvtColor(result.colored_depth, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path}")
        else:
            print("  WARNING: No colored depth map produced")

        # Save raw depth as numpy
        if result.depth is not None:
            npy_path = os.path.join(output_dir, f"{model_id}_raw.npy")
            np.save(npy_path, result.depth)
            print(f"  Raw depth shape: {result.depth.shape}, "
                  f"range: [{result.depth.min():.4f}, {result.depth.max():.4f}]")

        print(f"  ✅ SUCCESS ({load_time + infer_time:.2f}s total)")
        return True

    except ImportError as e:
        print(f"  ⏭️  SKIPPED — missing dependency: {e}")
        return False
    except Exception as e:
        print(f"  ❌ FAILED — {type(e).__name__}: {e}")
        return False


def main():
    print(f"Input image: {INPUT_IMAGE}")
    print(f"Output dir:  {OUTPUT_DIR}")
    assert os.path.exists(INPUT_IMAGE), f"Input image not found: {INPUT_IMAGE}"

    # Show input image info
    img = Image.open(INPUT_IMAGE)
    print(f"Image size:  {img.size[0]}x{img.size[1]}")
    print(f"\nTotal models to test: {len(ALL_MODELS)}")

    results = {"success": [], "skipped": [], "failed": []}

    for model_id in ALL_MODELS:
        ok = run_inference(model_id, INPUT_IMAGE, OUTPUT_DIR)
        if ok:
            results["success"].append(model_id)
        elif ok is False:
            # Check if it was skipped or failed by looking at what was printed
            results["skipped"].append(model_id)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  ✅ Success:  {len(results['success'])}/{len(ALL_MODELS)}")
    for m in results["success"]:
        print(f"     - {m}")
    print(f"  ⏭️  Skipped:  {len(results['skipped'])}/{len(ALL_MODELS)}")
    for m in results["skipped"]:
        print(f"     - {m}")
    print(f"\n  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single-model mode: python demo_all.py <model_id>
        model_id = sys.argv[1]
        ok = run_inference(model_id, INPUT_IMAGE, OUTPUT_DIR)
        sys.exit(0 if ok else 1)
    main()
