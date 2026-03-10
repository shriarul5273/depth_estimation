# v0.0.6 — Command-Line Interface

First minor release. Adds a `depth-estimate` CLI so depth estimation models can be used directly from the terminal without writing Python code.

### New Features

- **`depth-estimate predict`** — run depth estimation on a single image, a directory, a glob pattern, or a video file. Writes colourised PNG, raw float32 NPY, or both. Supports all 20 registered model variants, any matplotlib colormap, configurable batch size, and explicit device selection.
- **`depth-estimate list-models`** — print a formatted table (or `--json`) of all registered variants with model type, depth type, and backbone.
- **`depth-estimate info MODEL_ID`** — show architecture config for any variant (input size, embed dim, patch size, depth type, depth range for metric models). Supports `--json`.
- **`depth-estimate benchmark`** — stubbed subcommand, exits cleanly. Full implementation coming in v0.1.1 with the evaluation suite.
- **`docs/cli.md`** — full CLI reference covering all subcommands, flags, colormaps, output formats, model selection guide, exit codes, and machine-readable output examples.

### Details

- Entry point registered in `pyproject.toml` as `depth-estimate = "depth_estimation.cli:main"`.
- Global flags (`--device`, `--quiet`, `--verbose`) must precede the subcommand name.
- Video sources are auto-detected by file extension; output is a side-by-side RGB | depth MP4.
- Zero new mandatory dependencies — uses `opencv-python` and `matplotlib` already required by the core package.

---

# v0.0.5 — PyPI Link Fix

Patch release correcting broken documentation links on the PyPI package page.

### 🐛 Fixes

- Replaced relative doc links in `README.md` with full GitHub URLs so they resolve correctly from PyPI