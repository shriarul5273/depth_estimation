"""Tests for top-level depth_estimation package metadata."""

from importlib.metadata import version as installed_version

import depth_estimation


def test_version_matches_installed_metadata():
    """depth_estimation.__version__ must match what pip/PyPI report for the
    installed distribution. This used to be a value hardcoded independently
    in __init__.py, which drifted out of sync with pyproject.toml (0.0.9 vs
    0.1.1) — now it's read dynamically from package metadata instead, so
    there's only one place the version is declared.
    """
    assert depth_estimation.__version__ == installed_version("depth_estimation")


def test_version_is_nonempty_string():
    assert isinstance(depth_estimation.__version__, str)
    assert len(depth_estimation.__version__) > 0
