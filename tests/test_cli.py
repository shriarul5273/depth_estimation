"""Tests for the depth-estimate CLI (src/depth_estimation/cli.py)."""

import json
import sys

import numpy as np
import pytest

from depth_estimation import cli
from depth_estimation.output import DepthOutput


# --------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------- #


class TestCollectImages:
    def test_single_file(self, tmp_path):
        f = tmp_path / "a.jpg"
        f.write_bytes(b"fake")
        assert cli._collect_images(str(f)) == [f]

    def test_directory_filters_by_extension(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.png").write_bytes(b"fake")
        (tmp_path / "c.txt").write_bytes(b"fake")
        result = cli._collect_images(str(tmp_path))
        assert sorted(p.name for p in result) == ["a.jpg", "b.png"]

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.jpg").write_bytes(b"fake")
        (tmp_path / "c.png").write_bytes(b"fake")
        result = cli._collect_images(str(tmp_path / "*.jpg"))
        assert sorted(p.name for p in result) == ["a.jpg", "b.jpg"]

    def test_missing_source_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            cli._collect_images(str(tmp_path / "nope"))


class TestResolveOutputPath:
    def test_explicit_output_arg(self, tmp_path):
        source = tmp_path / "img.jpg"
        result = cli._resolve_output_path(source, "custom_out", None)
        assert result == cli.Path("custom_out")

    def test_output_dir_fallback(self, tmp_path):
        source = tmp_path / "img.jpg"
        result = cli._resolve_output_path(source, None, "some_dir")
        assert result == cli.Path("some_dir") / "img_depth"

    def test_default_alongside_source(self, tmp_path):
        source = tmp_path / "img.jpg"
        result = cli._resolve_output_path(source, None, None)
        assert result == source.parent / "img_depth"


class TestPrintTable:
    def test_basic_alignment(self, capsys):
        cli._print_table(["A", "BB"], [["1", "22"], ["333", "4"]])
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        assert lines[0].startswith("A")
        assert "333" in lines[3]


class TestSaveResult:
    def test_png_format(self, tmp_path):
        result = DepthOutput(
            depth=np.zeros((8, 8), dtype=np.float32),
            colored_depth=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        out_path = tmp_path / "out"
        cli._save_result(result, out_path, "png")
        assert (tmp_path / "out.png").exists()
        assert not (tmp_path / "out.npy").exists()

    def test_npy_format(self, tmp_path):
        result = DepthOutput(depth=np.ones((4, 4), dtype=np.float32))
        out_path = tmp_path / "out"
        cli._save_result(result, out_path, "npy")
        assert (tmp_path / "out.npy").exists()
        loaded = np.load(tmp_path / "out.npy")
        assert loaded.shape == (4, 4)

    def test_both_formats(self, tmp_path):
        result = DepthOutput(
            depth=np.zeros((4, 4), dtype=np.float32),
            colored_depth=np.zeros((4, 4, 3), dtype=np.uint8),
        )
        out_path = tmp_path / "out"
        cli._save_result(result, out_path, "both")
        assert (tmp_path / "out.png").exists()
        assert (tmp_path / "out.npy").exists()

    def test_no_colored_depth_skips_png(self, tmp_path):
        result = DepthOutput(depth=np.zeros((4, 4), dtype=np.float32), colored_depth=None)
        out_path = tmp_path / "out"
        cli._save_result(result, out_path, "png")
        assert not (tmp_path / "out.png").exists()


# --------------------------------------------------------------------- #
# Subcommands: list-models / info (real, offline — config-only)
# --------------------------------------------------------------------- #


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["depth-estimate"] + argv)
    cli.main()


class TestListModelsCommand:
    def test_table_output(self, monkeypatch, capsys):
        _run_cli(monkeypatch, ["list-models"])
        out = capsys.readouterr().out
        assert "depth-anything-v2-vits" in out
        assert "variants across" in out

    def test_json_output(self, monkeypatch, capsys):
        _run_cli(monkeypatch, ["list-models", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        variant_ids = {row["variant"] for row in data}
        assert "depth-anything-v2-vits" in variant_ids


class TestInfoCommand:
    def test_known_model_table(self, monkeypatch, capsys):
        _run_cli(monkeypatch, ["info", "depth-anything-v2-vits"])
        out = capsys.readouterr().out
        assert "depth-anything-v2" in out
        assert "Backbone" in out

    def test_known_model_json(self, monkeypatch, capsys):
        _run_cli(monkeypatch, ["info", "depth-anything-v2-vits", "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["model_type"] == "depth-anything-v2"

    def test_unknown_model_exits(self, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, ["info", "not-a-real-model"])
        err = capsys.readouterr().err
        assert "Unknown model identifier" in err


# --------------------------------------------------------------------- #
# predict — pipeline is faked out so no real model weights are needed
# --------------------------------------------------------------------- #


class _FakePipeline:
    """Mimics DepthPipeline.__call__: list in -> list out, single in -> single out."""

    def __call__(self, images, batch_size=1, colorize=True, colormap="Spectral_r"):
        was_list = isinstance(images, list)
        imgs = images if was_list else [images]
        results = [
            DepthOutput(
                depth=np.zeros((8, 8), dtype=np.float32),
                colored_depth=(
                    np.zeros((8, 8, 3), dtype=np.uint8) if colorize else None
                ),
                metadata={},
            )
            for _ in imgs
        ]
        return results if was_list else results[0]


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {}

    def _factory(task, model=None, device=None, **kwargs):
        calls["task"] = task
        calls["model"] = model
        calls["device"] = device
        return _FakePipeline()

    monkeypatch.setattr("depth_estimation.pipeline", _factory)
    return calls


class TestPredictCommand:
    def test_single_image(self, monkeypatch, fake_pipeline, tmp_path, capsys):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        _run_cli(
            monkeypatch,
            ["predict", str(img), "--model", "depth-anything-v2-vits"],
        )
        assert (tmp_path / "a_depth.png").exists()
        assert fake_pipeline["model"] == "depth-anything-v2-vits"

    def test_directory_batch(self, monkeypatch, fake_pipeline, tmp_path):
        for name in ("a.jpg", "b.jpg"):
            (tmp_path / name).write_bytes(b"fake")
        out_dir = tmp_path / "out"
        _run_cli(
            monkeypatch,
            [
                "predict",
                str(tmp_path),
                "--model",
                "depth-anything-v2-vits",
                "--output-dir",
                str(out_dir),
                "--batch-size",
                "2",
            ],
        )
        assert (out_dir / "a_depth.png").exists()
        assert (out_dir / "b_depth.png").exists()

    def test_npy_format(self, monkeypatch, fake_pipeline, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        _run_cli(
            monkeypatch,
            ["predict", str(img), "--model", "depth-anything-v2-vits", "--format", "npy"],
        )
        assert (tmp_path / "a_depth.npy").exists()
        assert not (tmp_path / "a_depth.png").exists()

    def test_missing_source_exits(self, monkeypatch, fake_pipeline, tmp_path):
        with pytest.raises(SystemExit):
            _run_cli(
                monkeypatch,
                [
                    "predict",
                    str(tmp_path / "nope"),
                    "--model",
                    "depth-anything-v2-vits",
                ],
            )

    def test_device_forwarded_to_pipeline(self, monkeypatch, fake_pipeline, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        _run_cli(
            monkeypatch,
            [
                "--device",
                "cpu",
                "predict",
                str(img),
                "--model",
                "depth-anything-v2-vits",
            ],
        )
        assert fake_pipeline["device"] == "cpu"


# --------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------- #


class TestArgParser:
    def test_predict_requires_model(self, monkeypatch):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, ["predict", "some_source"])

    def test_evaluate_requires_dataset(self, monkeypatch):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, ["evaluate"])

    def test_no_subcommand_exits(self, monkeypatch):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, [])

    def test_evaluate_invalid_dataset_choice_exits(self, monkeypatch):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, ["evaluate", "--dataset", "not-a-real-dataset"])
