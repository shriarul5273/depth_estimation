# Release Notes

## 🎉 depth_estimation v0.0.4

## v0.0.4 — Documentation Restructure


A patch release focused on documentation structure and install command correctness.

## 📚 Documentation Improvements

- **New `docs/models.md`** — all supported models and variant IDs moved out of `README.md` into a dedicated reference file
- **New `docs/dependencies.md`** — dependency tables moved from `README.md` and updated to match `pyproject.toml` exactly (correct version pins, core vs optional split, dev extras)
- **`README.md` slimmed down** — each section now links to the relevant doc file rather than inlining all tables

## 🐛 Fixes

- Install commands corrected from `pip install depth_estimation` → `pip install depth-estimation` (hyphen, PEP 625)
- Optional extras syntax corrected: `pip install "depth-estimation[diffusers]"`
