"""Stereo support beyond basic inference: datasets, augmentations, metrics,
loss, video, evaluation, fine-tuning, ONNX export, quantization and
profiling — exercised with a random-weight S2M2-S (no downloads)."""

from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch

from depth_estimation.data import load_dataset
from depth_estimation.data.stereo import (
    KITTI2015Dataset,
    Middlebury2014Dataset,
    StereoCenterCrop,
    StereoColorJitter,
    StereoFolderDataset,
    StereoNormalize,
    StereoRandomCrop,
    StereoRandomScale,
    get_stereo_train_transforms,
    get_stereo_val_transforms,
    read_disparity,
    read_pfm,
)
from depth_estimation.evaluation import StereoEvaluator, evaluate, profile_latency
from depth_estimation.losses import DisparityLoss
from depth_estimation.models.s2m2.configuration_s2m2 import S2M2Config
from depth_estimation.models.s2m2.modeling_s2m2 import S2M2Model
from depth_estimation.output import DepthOutput
from depth_estimation.video import (
    StereoVideoStream,
    VideoStream,
    open_stream,
    process_video,
    split_stereo_frame,
)

H, W = 64, 96


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pfm(path: Path, data: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{data.shape[1]} {data.shape[0]}\n".encode())
        f.write(b"-1.0\n")
        f.write(np.flipud(data).astype("<f4").tobytes())


def _rgb(seed: int, h: int = H, w: int = W) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _stereo_folder(root: Path, n: int = 2, h: int = H, w: int = W) -> StereoFolderDataset:
    for d in ("left", "right", "disp"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _write_png(root / "left" / f"{i:03d}.png", _rgb(i, h, w))
        _write_png(root / "right" / f"{i:03d}.png", _rgb(100 + i, h, w))
        disp = np.full((h, w), 4.0, dtype=np.float32)
        disp[0, 0] = np.inf  # invalid
        _write_pfm(root / "disp" / f"{i:03d}.pfm", disp)
    return StereoFolderDataset(root / "left", root / "right", root / "disp")


def _small_model() -> S2M2Model:
    torch.manual_seed(0)
    return S2M2Model(S2M2Config(input_size=64)).eval()


# ---------------------------------------------------------------------------
# Readers and datasets
# ---------------------------------------------------------------------------


class TestReaders:
    def test_pfm_roundtrip(self, tmp_path):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        _write_pfm(tmp_path / "d.pfm", data)
        np.testing.assert_array_equal(read_pfm(tmp_path / "d.pfm"), data)

    def test_pfm_inf_is_invalid(self, tmp_path):
        data = np.array([[1.0, np.inf]], dtype=np.float32)
        _write_pfm(tmp_path / "d.pfm", data)
        disp, valid = read_disparity(tmp_path / "d.pfm")
        assert valid.tolist() == [[True, False]]
        assert disp[0, 1] == 0.0

    def test_png_kitti_scale(self, tmp_path):
        raw = np.array([[0, 256, 512]], dtype=np.uint16)
        cv2.imwrite(str(tmp_path / "d.png"), raw)
        disp, valid = read_disparity(tmp_path / "d.png", scale=256.0)
        assert disp.tolist() == [[0.0, 1.0, 2.0]]
        assert valid.tolist() == [[False, True, True]]


class TestStereoDatasets:
    def test_folder_sample(self, tmp_path):
        ds = _stereo_folder(tmp_path)
        assert len(ds) == 2 and ds.is_stereo
        s = ds[0]
        assert s["pixel_values"].shape == (6, H, W)
        assert 0.0 <= s["pixel_values"].min() and s["pixel_values"].max() <= 1.0
        assert s["disparity"].shape == s["valid_mask"].shape == (1, H, W)
        assert not s["valid_mask"][0, 0, 0] and s["disparity"][0, 0, 0] == 0
        assert torch.isfinite(s["disparity"]).all()

    def test_folder_via_load_dataset(self, tmp_path):
        _stereo_folder(tmp_path)
        ds = load_dataset(
            "stereo_folder",
            left_dir=tmp_path / "left",
            right_dir=tmp_path / "right",
            disparity_dir=tmp_path / "disp",
        )
        assert isinstance(ds, StereoFolderDataset)

    def test_max_disparity_masks(self, tmp_path):
        _stereo_folder(tmp_path)
        ds = StereoFolderDataset(
            tmp_path / "left", tmp_path / "right", tmp_path / "disp", max_disparity=3.0
        )
        assert not ds[0]["valid_mask"].any()

    def test_middlebury_layout(self, tmp_path):
        scene = tmp_path / "MiddEval3" / "trainingQ" / "Toy"
        scene.mkdir(parents=True)
        _write_png(scene / "im0.png", _rgb(0))
        _write_png(scene / "im1.png", _rgb(1))
        _write_pfm(scene / "disp0GT.pfm", np.full((H, W), 5.0, dtype=np.float32))
        nocc = np.full((H, W), 255, dtype=np.uint8)
        nocc[:, :10] = 128  # occluded
        cv2.imwrite(str(scene / "mask0nocc.png"), nocc)
        (scene / "calib.txt").write_text(
            "cam0=[1000.5 0 50; 0 1000.5 30; 0 0 1]\ncam1=[1000.5 0 60; 0 1000.5 30; 0 0 1]\n"
            "doffs=10.0\nbaseline=176.252\nwidth=96\nheight=64\n"
        )
        ds = Middlebury2014Dataset(root=tmp_path, download=False)
        assert len(ds) == 1 and ds.scene_names == ["Toy"]
        assert ds[0]["valid_mask"].all()
        noc = Middlebury2014Dataset(root=tmp_path, download=False, occlusion="noc")
        assert not noc[0]["valid_mask"][..., :10].any()
        cal = ds.calibration(0)
        assert cal == pytest.approx(
            {"focal_length_px": 1000.5, "baseline": 0.176252, "doffs": 10.0}
        )

    def test_middlebury_missing_without_download(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Middlebury"):
            Middlebury2014Dataset(root=tmp_path, download=False)

    def test_kitti_2015_layout(self, tmp_path):
        base = tmp_path / "training"
        for d in ("image_2", "image_3", "disp_occ_0"):
            (base / d).mkdir(parents=True)
        _write_png(base / "image_2" / "000000_10.png", _rgb(0))
        _write_png(base / "image_3" / "000000_10.png", _rgb(1))
        cv2.imwrite(
            str(base / "disp_occ_0" / "000000_10.png"),
            np.full((H, W), 256 * 7, dtype=np.uint16),
        )
        ds = load_dataset("kitti_2015", root=str(tmp_path))
        assert isinstance(ds, KITTI2015Dataset) and len(ds) == 1
        assert torch.allclose(ds[0]["disparity"], torch.full((1, H, W), 7.0))

    def test_kitti_2015_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="registration"):
            KITTI2015Dataset(root=tmp_path)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _sample():
    pv = torch.rand(6, H, W)
    disp = torch.full((1, H, W), 10.0)
    valid = torch.ones(1, H, W, dtype=torch.bool)
    return pv, disp, valid


class TestStereoTransforms:
    def test_scale_rescales_disparity_with_width(self):
        pv, disp, valid = StereoRandomScale((2.0, 2.0))(*_sample())
        assert pv.shape == (6, 2 * H, 2 * W)
        assert torch.allclose(disp, torch.full_like(disp, 20.0))
        assert valid.dtype == torch.bool and valid.all()

    def test_crop_is_consistent_across_views(self):
        pv = torch.arange(H * W, dtype=torch.float32).reshape(1, H, W).repeat(6, 1, 1)
        disp = pv[:1].clone()
        out_pv, out_disp, _ = StereoRandomCrop((32, 32))(pv, disp, torch.ones(1, H, W, dtype=torch.bool))
        assert out_pv.shape == (6, 32, 32)
        assert torch.equal(out_pv[0], out_pv[3])  # same window in both views
        assert torch.equal(out_pv[0], out_disp[0])

    def test_crop_pads_small_inputs_as_invalid(self):
        pv, disp, valid = StereoRandomCrop((H + 8, W))(*_sample())
        assert pv.shape == (6, H + 8, W)
        assert not valid[:, H:].any()

    def test_center_crop(self):
        _, _, valid = StereoCenterCrop((32, 64))(*_sample())
        assert valid.shape == (1, 32, 64)

    def test_symmetric_jitter_matches_views(self):
        pv = torch.rand(3, H, W).repeat(2, 1, 1)  # identical views
        out, _, _ = StereoColorJitter(asymmetric_prob=0.0)(pv, *_sample()[1:])
        assert torch.allclose(out[:3], out[3:])

    def test_normalize_to_minus_one_one(self):
        pv = torch.ones(6, 4, 4)
        out, _, _ = StereoNormalize()(pv, torch.zeros(1, 4, 4), torch.ones(1, 4, 4, dtype=torch.bool))
        assert torch.allclose(out, torch.ones_like(out))

    def test_presets(self):
        pv, disp, valid = get_stereo_train_transforms((32, 64))(*_sample())
        assert pv.shape == (6, 32, 64) and disp.shape == valid.shape == (1, 32, 64)
        pv, _, _ = get_stereo_val_transforms()(*_sample())
        assert pv.shape == (6, H, W) and pv.min() >= -1.0


# ---------------------------------------------------------------------------
# Metrics and loss
# ---------------------------------------------------------------------------


class TestStereoMetricsAndLoss:
    def test_evaluator_values(self):
        gt = torch.tensor([[10.0, 10.0, 100.0, 50.0]])
        pred = torch.tensor([[10.5, 12.5, 104.0, 0.0]])
        valid = torch.tensor([[True, True, True, False]])
        ev = StereoEvaluator()
        ev.update(pred, gt, valid)
        r = ev.compute()
        assert r["n_pixels"] == 3
        assert r["epe"] == pytest.approx((0.5 + 2.5 + 4.0) / 3)
        assert r["bad1"] == pytest.approx(2 / 3)
        assert r["bad2"] == pytest.approx(2 / 3)
        assert r["bad3"] == pytest.approx(1 / 3)
        assert r["d1"] == pytest.approx(0.0)  # 4 px error is < 5 % of 100

    def test_evaluator_empty(self):
        assert StereoEvaluator().compute()["epe"] == 0.0

    def test_disparity_loss(self):
        pred = torch.zeros(1, 2, 2, requires_grad=True)
        target = torch.tensor([[[2.0, 0.5], [9.0, 9.0]]])
        valid = torch.tensor([[[True, True], [False, False]]])
        l1 = DisparityLoss("l1")(pred, target, valid)["loss"]
        assert l1.item() == pytest.approx((2.0 + 0.5) / 2)
        smooth = DisparityLoss()(pred, target, valid)["loss"]
        assert smooth.item() == pytest.approx((1.5 + 0.125) / 2)
        smooth.backward()
        assert pred.grad is not None

    def test_disparity_loss_no_valid_pixels(self):
        pred = torch.zeros(1, 2, 2, requires_grad=True)
        out = DisparityLoss()(pred, torch.ones(1, 2, 2), torch.zeros(1, 2, 2, dtype=torch.bool))
        assert out["loss"].item() == 0.0
        out["loss"].backward()


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _write_video(path: Path, frames) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _frames_glob(directory: Path, prefix: str, n: int, seed: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _write_png(directory / f"{prefix}_{i:03d}.png", _rgb(seed + i))
    return str(directory / f"{prefix}_*.png")


class TestStereoVideo:
    def test_split_layouts(self):
        frame = np.zeros((4, 7, 3), dtype=np.uint8)
        frame[:, 3:] = 1
        left, right = split_stereo_frame(frame, "sbs")
        assert left.shape == right.shape == (4, 3, 3) and right.min() == 1
        top, bottom = split_stereo_frame(np.zeros((5, 4, 3), dtype=np.uint8), "tab")
        assert top.shape == bottom.shape == (2, 4, 3)
        with pytest.raises(ValueError):
            split_stereo_frame(frame, "diagonal")

    def test_two_sources(self, tmp_path):
        left = _frames_glob(tmp_path / "l", "l", 3, 0)
        right = _frames_glob(tmp_path / "r", "r", 2, 10)  # shorter
        with StereoVideoStream((left, right)) as stream:
            assert stream.total_frames == 2
            pairs = list(stream)
        assert len(pairs) == 2
        (l0, r0), meta = pairs[0]
        assert l0.shape == r0.shape == (H, W, 3) and meta["frame_index"] == 0

    def test_packed_source(self, tmp_path):
        d = tmp_path / "sbs"
        d.mkdir()
        _write_png(d / "f_000.png", np.concatenate([_rgb(0), _rgb(1)], axis=1))
        (left, right), _ = next(iter(StereoVideoStream(str(d / "f_*.png"), layout="sbs")))
        assert left.shape == right.shape == (H, W, 3)

    def test_open_stream_validation(self, tmp_path):
        glob = _frames_glob(tmp_path, "f", 1, 0)
        assert isinstance(open_stream(glob), VideoStream)
        with pytest.raises(ValueError, match="stereo model"):
            open_stream(glob, is_stereo=True)
        with pytest.raises(ValueError, match="monocular"):
            open_stream((glob, glob))
        with pytest.raises(ValueError, match="layout"):
            StereoVideoStream(glob, layout=None)

    def test_process_video_two_sources(self, tmp_path):
        left, right = tmp_path / "l.mp4", tmp_path / "r.mp4"
        _write_video(left, [_rgb(i) for i in range(3)])
        _write_video(right, [_rgb(10 + i) for i in range(3)])
        seen = []

        def fake_pipe(images, **kwargs):
            seen.append(images)
            return DepthOutput(depth=np.random.rand(H, W).astype(np.float32))

        pipe = mock.MagicMock(side_effect=fake_pipe)
        pipe.model.config.is_stereo = True
        out = tmp_path / "out.mp4"
        process_video(pipe, (str(left), str(right)), str(out))
        assert all(isinstance(x, tuple) and len(x) == 2 for x in seen) and len(seen) == 3
        cap = cv2.VideoCapture(str(out))
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * W  # left view | depth
        cap.release()

    def test_pipeline_stream_with_s2m2(self, tmp_path):
        from depth_estimation.pipeline_utils import DepthPipeline
        from depth_estimation.processing_utils import DepthProcessor

        config = S2M2Config(input_size=64)
        pipe = DepthPipeline(_small_model(), DepthProcessor.from_config(config), device="cpu")
        left = _frames_glob(tmp_path / "l", "l", 2, 0)
        right = _frames_glob(tmp_path / "r", "r", 2, 10)
        results = list(pipe.stream((left, right)))
        assert len(results) == 2
        assert results[0].depth.shape == (H, W)
        assert results[1].metadata["frame_index"] == 1


# ---------------------------------------------------------------------------
# Evaluation and fine-tuning
# ---------------------------------------------------------------------------


class TestStereoEvaluate:
    def test_evaluate_s2m2_on_folder(self, tmp_path):
        ds = _stereo_folder(tmp_path)
        results = evaluate(_small_model(), ds, device="cpu", num_workers=0)
        for key in ("epe", "bad1", "bad2", "bad3", "d1", "n_pixels"):
            assert key in results
        assert results["n_samples"] == 2
        assert results["n_pixels"] == 2 * (H * W - 1)

    def test_num_samples(self, tmp_path):
        ds = _stereo_folder(tmp_path, n=3)
        results = evaluate(_small_model(), ds, device="cpu", num_workers=0, num_samples=1)
        assert results["n_samples"] == 1

    def test_mismatch_raises(self, tmp_path):
        ds = _stereo_folder(tmp_path)
        from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (
            DepthAnythingV2Config,
        )
        from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import (
            DepthAnythingV2Model,
        )

        mono = DepthAnythingV2Model(DepthAnythingV2Config(backbone="vits"))
        with pytest.raises(ValueError, match="Monocular model"):
            evaluate(mono, ds, device="cpu", num_workers=0)


class TestStereoTraining:
    def test_s2m2_backbone_helpers(self):
        model = _small_model()
        model.freeze_backbone()
        assert not any(p.requires_grad for p in model.net.cnn_backbone.parameters())
        assert all(p.requires_grad for p in model.net.refiner.parameters())
        model.unfreeze_top_k_backbone_layers(1)
        assert all(p.requires_grad for p in model.net.transformer.parameters())
        assert not any(p.requires_grad for p in model.net.feat_pyramid.parameters())
        decoder, backbone = model.get_parameter_groups()
        assert decoder["params"] and backbone["params"]

    def test_train_mode_runs_in_float32(self):
        model = _small_model().train()
        # No forced float16 autocast in train mode: the trainer decides.
        assert isinstance(model._compute_context(torch.device("cuda")), nullcontext)
        out = model(torch.rand(1, 6, H, W) * 2 - 1)
        assert out.dtype == torch.float32
        out.mean().backward()
        assert model.net.refiner.disp_update[0].weight.grad is not None

    def test_trainer_one_epoch(self, tmp_path):
        from depth_estimation import DepthTrainer, DepthTrainingArguments

        ds = _stereo_folder(tmp_path / "data")
        ds.transform = get_stereo_val_transforms()
        args = DepthTrainingArguments(
            output_dir=str(tmp_path / "ckpt"),
            num_epochs=1,
            batch_size=1,
            dataloader_num_workers=0,
            save_every_n_epochs=1,
        )
        trainer = DepthTrainer(_small_model(), args, train_dataset=ds, eval_dataset=ds)
        assert trainer._eval_metric == "epe"  # depth default swapped for stereo
        trainer.train()
        assert (tmp_path / "ckpt" / "final" / "model.pt").exists()
        assert (tmp_path / "ckpt" / "best_model" / "model.pt").exists()

    def test_trainer_rejects_mono_dataset_for_stereo(self, tmp_path):
        from depth_estimation import DepthTrainer, DepthTrainingArguments
        from depth_estimation.data import FolderDataset

        (tmp_path / "rgb").mkdir()
        _write_png(tmp_path / "rgb" / "a.png", _rgb(0))
        mono = FolderDataset(tmp_path / "rgb")
        args = DepthTrainingArguments(output_dir=str(tmp_path / "ckpt"))
        with pytest.raises(ValueError, match="stereo model"):
            DepthTrainer(_small_model(), args, train_dataset=mono)

    def test_stereo_metric_for_mono_model_raises(self, tmp_path):
        from depth_estimation import DepthTrainer, DepthTrainingArguments

        model = torch.nn.Linear(1, 1)
        args = DepthTrainingArguments(output_dir=str(tmp_path), eval_metric="epe")
        with pytest.raises(ValueError, match="stereo metric"):
            DepthTrainer(model, args, train_dataset=[])

    def test_disparity_loss_arg_validated(self, tmp_path):
        from depth_estimation import DepthTrainingArguments

        with pytest.raises(ValueError, match="disparity_loss"):
            DepthTrainingArguments(output_dir=str(tmp_path), disparity_loss="l2")


# ---------------------------------------------------------------------------
# ONNX export, quantization, profiling
# ---------------------------------------------------------------------------


class TestStereoExportAndOptimize:
    def test_onnx_export_two_inputs_and_verify(self, tmp_path):
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        from depth_estimation.export import export_onnx

        model = _small_model()
        path = export_onnx(model, tmp_path / "s2m2.onnx", input_size=64, verify=True)
        assert model.onnx_compatible_mode is False  # restored after export
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        assert [i.name for i in sess.get_inputs()] == ["left", "right"]
        x = np.random.rand(2, 3, 64, 64).astype(np.float32) * 2 - 1
        out = sess.run(None, {"left": x, "right": x[::-1].copy()})[0]
        assert out.shape == (2, 64, 64)  # dynamic batch

    def test_onnx_dynamic_spatial_rejected(self, tmp_path):
        from depth_estimation.export import export_onnx

        with pytest.raises(ValueError, match="dynamic_spatial"):
            export_onnx(_small_model(), tmp_path / "x.onnx", input_size=64, dynamic_spatial=True)

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_reduced_precision_cpu(self, dtype):
        from depth_estimation import quantize_model

        model = quantize_model(_small_model(), dtype)
        out = model(torch.rand(1, 6, H, W) * 2 - 1)
        assert out.shape == (1, H, W) and torch.isfinite(out).all()

    def test_int8(self):
        from depth_estimation import quantize_model

        out = quantize_model(_small_model(), "int8")(torch.rand(1, 6, H, W) * 2 - 1)
        assert out.shape == (1, H, W)

    def test_profile_latency_uses_pairs(self):
        p = profile_latency(_small_model(), input_size=64, num_warmup=0, num_runs=1, device="cpu")
        assert p["input_shape"] == (1, 6, 64, 64)
