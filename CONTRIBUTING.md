# Contributing

Thanks for considering a contribution to `depth_estimation`. This document covers the practical setup — for adding a new model family specifically, see [docs/adding_a_model.md](docs/adding_a_model.md).

## Setup

```bash
git clone https://github.com/shriarul5273/depth_estimation.git
cd depth_estimation
pip install -e ".[dev]"
```

`[dev]` pulls in `pytest`, `ruff`, `mypy`, and `onnx`/`onnxruntime` (needed for the export/quantization tests) on top of the core runtime dependencies.

## Running tests

The test suite is split into two tiers:

```bash
# Fast tier — runs on every push/PR, no network access, no real weight downloads
pytest tests -m "not slow"

# Slow tier — downloads real pretrained weights for all 28 registered model
# variants from the Hugging Face Hub (several GB), plus a few offline but
# architecturally heavy cases. Runs weekly in CI, not on every PR.
pytest tests -m slow
```

Run the fast tier before opening a PR. The slow tier is useful when your change touches model loading, weight-mapping, or anything else that only a real checkpoint would exercise — expect it to take a while and to use real disk/bandwidth.

## Linting and type checking

```bash
ruff check .      # must pass — CI gates on this
mypy src/depth_estimation   # advisory only for now, see below — CI reports but does not gate
```

The codebase currently has a pre-existing mypy backlog (~290 errors, mostly missing annotations in older modules — see the `[tool.mypy]` comment in `pyproject.toml`). CI runs mypy but doesn't fail the build on it yet. New code should still aim to be clean under `mypy src/depth_estimation` where practical; don't feel obligated to fix unrelated pre-existing errors in files you're only touching in passing.

## Code style

- No comments that restate what the code does — only add one when the *why* isn't obvious from the code itself (a subtle invariant, a workaround for a specific bug, a non-obvious version constraint).
- Prefer small, focused changes over speculative abstractions. Don't add configurability or error handling for cases that can't currently happen.
- Match the surrounding module's existing patterns (e.g. how other model families structure `configuration_*.py`/`modeling_*.py`, how other CLI subcommands are wired up) rather than introducing a new convention for one file.
- Claims in docstrings and `docs/*.md` should be verified against actual behavior, not assumed — several bugs in this project were caught specifically because a documented number (a file size, an error message, a supported version) didn't match what actually ran.

## Pull requests

- Keep PRs scoped to one concern where possible.
- Include the test command(s) you ran and their result in the PR description.
- CI (lint, typecheck, build smoke test, and the fast test matrix across Python 3.10–3.12 × several torch versions) must be green before merge.

## Reporting bugs / requesting features

Open an issue at [github.com/shriarul5273/depth_estimation/issues](https://github.com/shriarul5273/depth_estimation/issues). For bugs, include the model variant, torch/Python versions, and a minimal reproduction if you can.

## License

By contributing, you agree your contributions are licensed under this project's [MIT License](LICENSE).
