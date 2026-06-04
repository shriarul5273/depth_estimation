"""Packaging metadata regression tests."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from packaging.requirements import Requirement


def test_pytorch_packages_not_in_project_dependencies():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]

    dependency_names = {Requirement(dep).name.lower() for dep in dependencies}

    assert "torch" not in dependency_names
    assert "torchvision" not in dependency_names
