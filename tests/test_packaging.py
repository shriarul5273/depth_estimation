"""Packaging metadata regression tests."""

from pathlib import Path


def test_pytorch_packages_not_pinned_in_project_dependencies():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert '"torch"' not in content
    assert '"torch==' not in content
    assert '"torchvision"' not in content
