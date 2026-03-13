# Command-Line Interface

The `depth-estimate` command gives you access to the full depth estimation library directly from the terminal — no Python code required. It wraps the [Pipeline API](pipeline.md) for inference and the [Evaluation API](evaluation.md) for benchmarking.

## Installation

The CLI is registered as an entry point when you install the package:

```bash
pip install depth-estimation
```

After installation, the `depth-estimate` command is available on your `PATH`:

```bash
depth-estimate --help
```

For a local / development install:

```bash
git clone https://github.com/shriarul5273/depth_estimation
cd depth_estimation
pip install -e .
depth-estimate --help
```

---

## Command Structure

```
depth-estimate [GLOBAL FLAGS] SUBCOMMAND [SUBCOMMAND FLAGS]
```

**Important:** global flags (`--device`, `--quiet`, `--verbose`) must come **before** the subcommand name.

```bash
# Correct
depth-estimate --quiet predict image.jpg --model depth-anything-v2-vitb

# Wrong — --quiet after the subcommand is not recognised
depth-estimate predict image.jpg --model depth-anything-v2-vitb --quiet
```

---

## Global Flags

These flags apply to every subcommand and must appear before the subcommand name.

| Flag | Description |
|---|---|
| `--device DEVICE` | Device to run inference on: `cuda`, `cpu`, or `mps`. Auto-detected if omitted (prefers CUDA → MPS → CPU). |
| `--quiet` / `-q` | Suppress all progress output. Only errors are printed. |
| `--verbose` / `-v` | Enable debug logging from the underlying library. |

---

## Subcommands

### `predict`

Run depth estimation on a single image, a directory of images, a glob pattern, or a video file.

```
depth-estimate predict SOURCE --model MODEL [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `SOURCE` | Yes | Image path, directory path, quoted glob pattern, or video file. |
| `--model` / `-m` | Yes | Model variant ID. See `depth-estimate list-models` for all options. |
| `--output` / `-o` | No | Output file path. Used for single-image and video predictions. |
| `--output-dir` | No | Output directory for batch predictions (directory / glob source). |
| `--format` | No | Output format: `png` (default), `npy`, or `both`. |
| `--colormap` | No | Matplotlib colormap name (default: `Spectral_r`). |
| `--batch-size` | No | Number of images to process per forward pass (default: `1`). |

#### Output formats

| `--format` value | Files written |
|---|---|
| `png` | `<name>_depth.png` — colorised depth map (RGB, uint8) |
| `npy` | `<name>_depth.npy` — raw float32 depth array, shape `(H, W)` |
| `both` | Both files above |

For relative depth models the `.npy` values are normalised to `[0, 1]`. For metric models (ZoeDepth, DepthPro, Depth Anything v3 metric) values are in **metres**.

#### Colormaps

Any [matplotlib colormap](https://matplotlib.org/stable/gallery/color/colormap_reference.html) name is accepted. Commonly used ones:

| Colormap | Character |
|---|---|
| `Spectral_r` | Default — warm (close) to cool (far) |
| `inferno` | Dark purple → orange → yellow |
| `viridis` | Perceptually uniform, colour-blind safe |
| `plasma` | Purple → pink → yellow |
| `magma` | Dark → light, low contrast |
| `turbo` | Full-spectrum, high contrast |
| `RdYlBu` | Red (close) → yellow → blue (far) |
| `coolwarm` | Cool blues to warm reds |

#### Single image

```bash
# Default output: demo_depth.png in the same directory as the input
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb

# Explicit output path
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb --output results/depth.png

# Save both colourised PNG and raw float32 NPY
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb --format both

# Different colormap
depth-estimate predict examples/demo.png --model depth-anything-v2-vitb --colormap inferno

# Metric depth model — NPY values will be in metres
depth-estimate predict examples/demo.png --model zoedepth --format both

# Run on CPU explicitly
depth-estimate --device cpu predict examples/demo.png --model depth-anything-v2-vits
```

#### Batch (directory)

When `SOURCE` is a directory, every image inside it is processed. Results are written to `--output-dir` using the pattern `<original_stem>_depth.<ext>`.

```bash
# Process all images in a directory
depth-estimate predict images/ --model depth-anything-v2-vitb --output-dir results/

# With larger batch size for faster GPU throughput
depth-estimate predict images/ --model depth-anything-v2-vitb --output-dir results/ --batch-size 4

# Save NPY files instead of PNGs
depth-estimate predict images/ --model depth-anything-v2-vitb --output-dir results/ --format npy

# Suppress per-file progress lines
depth-estimate --quiet predict images/ --model depth-anything-v2-vitb --output-dir results/
```

#### Batch (glob pattern)

Quote the glob pattern to prevent shell expansion.

```bash
# All JPEGs in a directory tree
depth-estimate predict "photos/**/*.jpg" --model depth-anything-v2-vitb --output-dir depth_maps/

# Specific naming pattern
depth-estimate predict "frames/frame_*.png" --model depth-anything-v2-vitb --output-dir depth_maps/
```

#### Video

When `SOURCE` is a video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`), the output is a side-by-side RGB | depth MP4 file. Inference runs frame-by-frame using OpenCV.

```bash
# Default output: video_depth.mp4 in the same directory as the input
depth-estimate predict video.mp4 --model depth-anything-v2-vitb

# Explicit output path
depth-estimate predict video.mp4 --model depth-anything-v2-vitb --output depth_video.mp4

# Different colormap
depth-estimate predict video.mp4 --model depth-anything-v2-vitb --colormap turbo
```

> **Note:** `--format` and `--batch-size` are ignored for video sources. The output is always an MP4 with a side-by-side composite.

---

### `list-models`

Print a table of all registered model variants.

```
depth-estimate list-models [--json]
```

| Flag | Description |
|---|---|
| `--json` | Output as a JSON array instead of a formatted table. |

#### Example output (table)

```
depth-estimate list-models
```

```
Variant                               Model Type                Type      Backbone
------------------------------------  ------------------------  --------  ----------
depth-anything-v1-vitb                depth-anything-v1         relative  vitb
depth-anything-v1-vitl                depth-anything-v1         relative  vitl
depth-anything-v1-vits                depth-anything-v1         relative  vits
depth-anything-v2-vitb                depth-anything-v2         relative  vitb
depth-anything-v2-vitl                depth-anything-v2         relative  vitl
depth-anything-v2-vits                depth-anything-v2         relative  vits
depth-anything-v3-base                depth-anything-v3         relative  base
depth-anything-v3-giant               depth-anything-v3         relative  giant
depth-anything-v3-large               depth-anything-v3         relative  large
depth-anything-v3-metric-large        depth-anything-v3         metric    metric_large
depth-anything-v3-mono-large          depth-anything-v3         relative  mono_large
depth-anything-v3-nested-giant-large  depth-anything-v3-nested  relative  giant
depth-anything-v3-small               depth-anything-v3         relative  small
depth-pro                             depth-pro                 metric    depth-pro
marigold-dc                           marigold-dc               relative  marigold-dc
midas-beit-large                      midas                     relative  beit-large
midas-dpt-hybrid                      midas                     relative  dpt-hybrid
midas-dpt-large                       midas                     relative  dpt-large
moge-v1                               moge                      relative  vitl
moge-v2-vitb-normal                   moge                      metric    vitb
moge-v2-vitl                          moge                      metric    vitl
moge-v2-vitl-normal                   moge                      metric    vitl
moge-v2-vits-normal                   moge                      metric    vits
omnivggt                              omnivggt                  metric    vitl
pixel-perfect-depth                   pixel-perfect-depth       relative  vitl
vggt                                  vggt                      metric    vitl
vggt-commercial                       vggt                      metric    vitl
zoedepth                              zoedepth                  metric    zoedepth-nyu-kitti

28 variants across 12 model families.
```

#### Example output (JSON)

```bash
depth-estimate list-models --json
```

```json
[
  {
    "variant": "depth-anything-v2-vitb",
    "model_type": "depth-anything-v2",
    "is_metric": false,
    "backbone": "vitb"
  },
  ...
]
```

---

### `info`

Show the configuration for a specific model variant — backbone, input size, depth type, and architecture hyperparameters.

```
depth-estimate info MODEL_ID [--json]
```

| Argument | Description |
|---|---|
| `MODEL_ID` | Any variant ID from `list-models`. |
| `--json` | Output as JSON. |

#### Example

```bash
depth-estimate info depth-anything-v2-vitb
```

```
Model: depth-anything-v2-vitb
----------------------------------------
  Model Type       depth-anything-v2
  Backbone         vitb
  Depth Type       relative
  Input Size       518
  Patch Size       14
  Embed Dim        768
  Num Heads        12
  Num Layers       12
```

```bash
depth-estimate info zoedepth
```

```
Model: zoedepth
----------------------------------------
  Model Type       zoedepth
  Backbone         zoedepth-nyu-kitti
  Depth Type       metric
  Input Size       384
  Patch Size       16
  Embed Dim        768
  Num Heads        12
  Num Layers       12
  Max Depth        80.0
  Min Depth        0.001
```

```bash
depth-estimate info depth-anything-v2-vitb --json
```

```json
{
  "variant": "depth-anything-v2-vitb",
  "model_type": "depth-anything-v2",
  "backbone": "vitb",
  "depth_type": "relative",
  "input_size": 518,
  "patch_size": 14,
  "embed_dim": 768,
  "num_heads": 12,
  "num_layers": 12,
  "max_depth": null,
  "min_depth": null
}
```

---

### `evaluate`

Evaluate a model on a standard depth benchmark. Relative-depth models are aligned per-sample (least-squares scale + shift) before metric computation — detected automatically from `config.is_metric`.

```
depth-estimate evaluate --dataset DATASET [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--model` / `-m` | `depth-anything-v2-vitb` | Model variant ID. Ignored with `--compare`. |
| `--dataset` / `-d` | **required** | `nyu_depth_v2` (alias `nyu`), `kitti_eigen` (alias `kitti`), or `diode`. |
| `--split` | `test` | Dataset split: `train`, `val`, or `test`. |
| `--dataset-root` | auto | Local root directory. Auto-downloads where supported. |
| `--compare` | off | Evaluate a preset list of models and print a comparison table. |
| `--compare-models` | preset | Space-separated model IDs to compare. Overrides the built-in preset. |
| `--num-samples N` | all | Limit to N samples for quick checks. |
| `--batch-size B` | `1` | Images per forward pass. |
| `--num-workers W` | `4` | DataLoader workers. Use `0` on Windows with h5py. |
| `--no-align` | off | Disable alignment for relative models. |
| `--scene-type` | `all` | DIODE only: `indoors`, `outdoors`, or `all`. |
| `--max-depth M` | dataset default | Maximum valid depth in metres. |
| `--output` / `-o` | none | Save results to a JSON file. |
| `--json` | off | Print results as JSON instead of a table. |

#### Examples

```bash
# NYU Depth V2 — auto-downloads ~2.8 GB on first run
depth-estimate evaluate --model depth-anything-v2-vitb --dataset nyu_depth_v2

# Quick 50-sample check
depth-estimate evaluate --model depth-pro --dataset nyu --num-samples 50

# KITTI Eigen — manual download required
depth-estimate evaluate --model zoedepth --dataset kitti_eigen --dataset-root /data/kitti

# DIODE indoors subset
depth-estimate evaluate --model depth-anything-v2-vitl --dataset diode --scene-type indoors

# Compare preset models, save to JSON
depth-estimate evaluate --compare --dataset nyu_depth_v2 --output results/compare_nyu.json

# Compare custom model list
depth-estimate evaluate --compare --dataset nyu_depth_v2 \
    --compare-models depth-anything-v2-vits depth-anything-v2-vitb depth-pro

# Machine-readable output
depth-estimate evaluate --model depth-pro --dataset nyu --json

# Disable auto-alignment (metric model, no alignment needed)
depth-estimate evaluate --model zoedepth --dataset kitti --dataset-root /data/kitti --no-align
```

#### Example output (single model)

```
───────────────────────────────────────────────────────
  Results: depth-anything-v2-vitb  [nyu_depth_v2 / test]
───────────────────────────────────────────────────────
  Metric          Value  Direction
  ──────          ─────  ─────────
  abs_rel        0.0431  lower ↓
  sq_rel         0.0124  lower ↓
  rmse           0.3121  lower ↓
  rmse_log       0.0612  lower ↓
  delta1         0.9824  higher ↑
  delta2         0.9971  higher ↑
  delta3         0.9993  higher ↑
───────────────────────────────────────────────────────
  Samples : 654
  Time    : 142.3s
───────────────────────────────────────────────────────
```

#### Example output (`--compare`)

```
-----------------------------------------------------------------
Model                        abs_rel   sq_rel     rmse  rmse_log   delta1   delta2   delta3
                                 (↓)      (↓)      (↓)       (↓)      (↑)      (↑)      (↑)
-----------------------------------------------------------------
depth-anything-v2-vits      0.0512*  0.0143*  0.3541*   0.0712*  0.9723*  0.9961*  0.9992*  n=654
depth-anything-v2-vitb      0.0431   0.0124   0.3121    0.0612   0.9824   0.9971   0.9993   n=654
depth-anything-v2-vitl      0.0378   0.0101   0.2874    0.0571   0.9891   0.9981   0.9996   n=654
-----------------------------------------------------------------
* = best in column
```

> **Datasets requiring download:** NYU Depth V2 (~2.8 GB) and DIODE (~2.6 GB) auto-download on first use and are cached in `~/.cache/depth_estimation/`. KITTI requires registration — pass `--dataset-root` pointing to your local copy.

---

## Model Selection Guide

| Use case | Recommended model | Why |
|---|---|---|
| Fastest inference (CPU / edge) | `depth-anything-v2-vits` | Smallest ViT-S backbone |
| Best relative depth quality | `depth-anything-v2-vitl` | Large backbone, highly accurate |
| Metric depth (metres) | `zoedepth` or `depth-pro` | Produces absolute scale |
| Metric depth + best quality | `depth-anything-v3-metric-large` | Latest architecture |
| Video / real-time | `depth-anything-v2-vits` | Speed priority |
| Diffusion-guided (sparse input) | `marigold-dc` | Depth completion from sparse hints |

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Runtime error (bad image path, unknown model, inference failure) |
| `2` | Argument error (missing required flag, unrecognised argument) |

---

## Machine-Readable Output

Every subcommand supports `--json` for scripting and pipeline integration.

```bash
# Get a JSON array of all variants
depth-estimate list-models --json | jq '.[].variant'

# Check if a model is metric
depth-estimate info zoedepth --json | jq '.depth_type'
```
