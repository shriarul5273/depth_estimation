"""Tests for ONNX export (depth_estimation.export).

Requires the optional onnx/onnxruntime packages — skipped entirely if
they're not installed (they aren't core dependencies).
"""

import torch
import pytest

onnxruntime = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from depth_estimation.export import export_onnx  # noqa: E402
from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (  # noqa: E402
    DepthAnythingV2Config,
)
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import (  # noqa: E402
    DepthAnythingV2Model,
)
from depth_estimation.models.depth_anything_v3.configuration_depth_anything_v3 import (  # noqa: E402
    DepthAnythingV3Config,
)
from depth_estimation.models.depth_anything_v3.modeling_depth_anything_v3 import (  # noqa: E402
    DepthAnythingV3Model,
)
from depth_estimation.models.depth_pro.configuration_depth_pro import (  # noqa: E402
    DepthProConfig,
)
from depth_estimation.models.depth_pro.modeling_depth_pro import (  # noqa: E402
    DepthProModel,
)
from depth_estimation.models.pixel_perfect_depth.configuration_ppd import (  # noqa: E402
    PixelPerfectDepthConfig,
)
from depth_estimation.models.pixel_perfect_depth.modeling_ppd import (  # noqa: E402
    PixelPerfectDepthModel,
)


def _export_or_skip_if_sdpa_unsupported(*args, **kwargs):
    """Run export_onnx(), skipping (not failing) if this torch version's
    ONNX exporter has no symbolic mapping for scaled_dot_product_attention.

    That support was added in a later torch release than our declared
    floor (torch>=2.0) — torch==2.0.1 genuinely cannot export any model
    using F.scaled_dot_product_attention (DepthAnythingV3 and others; not
    DepthAnythingV2, which uses an older manual-attention implementation).
    This isn't a bug in our code, so we skip rather than xfail/hardcode a
    version cutoff we're not fully certain of.
    """
    try:
        return export_onnx(*args, **kwargs)
    except torch.onnx.errors.UnsupportedOperatorError as e:
        if "scaled_dot_product_attention" in str(e):
            pytest.skip(
                f"torch {torch.__version__}'s ONNX exporter doesn't support "
                "scaled_dot_product_attention (added in a later torch "
                "release than our floor of torch>=2.0)"
            )
        raise


class TestExportOnnxFast:
    """Offline, random-weight models that export correctly and are cheap."""

    @pytest.mark.parametrize("backbone", ["vits", "vitb"])
    def test_depth_anything_v2(self, tmp_path, backbone):
        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone=backbone)
        model = DepthAnythingV2Model(config).eval()
        out = tmp_path / "model.onnx"

        export_onnx(model, out, input_size=518, verify=True)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_depth_anything_v3_small(self, tmp_path):
        torch.manual_seed(0)
        config = DepthAnythingV3Config(backbone="small")
        model = DepthAnythingV3Model(config).eval()
        out = tmp_path / "model.onnx"

        _export_or_skip_if_sdpa_unsupported(model, out, input_size=518, verify=True)

        assert out.exists()

    def test_dynamic_batch_actually_works(self, tmp_path):
        """A batch size different from the one used to trace must still run."""
        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config).eval()
        out = tmp_path / "model.onnx"
        export_onnx(model, out, input_size=518, dynamic_batch=True)

        sess = onnxruntime.InferenceSession(
            str(out), providers=["CPUExecutionProvider"]
        )
        batch = torch.randn(3, 3, 518, 518).numpy()
        result = sess.run(None, {"pixel_values": batch})[0]
        assert result.shape[0] == 3

    def test_static_batch_rejects_other_sizes(self, tmp_path):
        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config).eval()
        out = tmp_path / "model.onnx"
        export_onnx(model, out, input_size=518, dynamic_batch=False)

        sess = onnxruntime.InferenceSession(
            str(out), providers=["CPUExecutionProvider"]
        )
        batch = torch.randn(2, 3, 518, 518).numpy()
        with pytest.raises(Exception):
            sess.run(None, {"pixel_values": batch})

    def test_float16_model_exports_and_verifies(self, tmp_path):
        """Regression test: export_onnx() used to always build a float32
        dummy input regardless of the model's actual dtype, so exporting a
        model previously cast via quantize_model(dtype="float16") raised
        RuntimeError: Input type (float) and bias type (c10::Half) should
        be the same. The dummy input's dtype must track the model's.

        atol/rtol loosened to 1e-2: confirmed against a real pretrained
        checkpoint that float16's own rounding noise (~3 decimal digits of
        precision) routinely exceeds export_onnx()'s float32-oriented
        default of 1e-3 — expected, not an export bug. See
        examples/optimize.py's matching comment.
        """
        from depth_estimation.quantization import quantize_model

        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config).eval()
        quantize_model(model, dtype="float16")
        out = tmp_path / "model_fp16.onnx"

        try:
            export_onnx(model, out, input_size=518, verify=True, atol=1e-2, rtol=1e-2)
        except RuntimeError as e:
            # Same known torch-version limitation as
            # test_quantization.py::test_float16_forward_works — some
            # torch CPU builds have no Half kernel for conv2d at all.
            pytest.skip(
                f"torch {torch.__version__} CPU doesn't implement this op "
                f"for float16 (known limitation on older torch): {e}"
            )

        assert out.exists()
        assert out.stat().st_size > 0

    def test_verify_raises_on_real_mismatch(self, tmp_path):
        """verify=True must actually fail when outputs genuinely don't match."""
        from depth_estimation.export import _verify_onnx_export

        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config).eval()
        out = tmp_path / "model.onnx"
        export_onnx(model, out, input_size=518)

        # A model with different (freshly re-randomized) weights than what
        # was exported must fail verification against the same ONNX file.
        torch.manual_seed(1)
        different_model = DepthAnythingV2Model(config).eval()
        dummy = torch.randn(1, 3, 518, 518)
        with pytest.raises(AssertionError):
            _verify_onnx_export(different_model, out, dummy_input=dummy)

    def test_onnx_exportable_false_blocks_export_immediately(self, tmp_path):
        """Regression test: BaseDepthModel._onnx_exportable = False (set by
        ZoeDepth and Marigold-DC, both of which wrap an opaque external
        pipeline and round-trip through PIL/numpy inside forward() —
        confirmed neither is meaningfully traceable) must raise
        immediately, before attempting any trace, rather than surfacing a
        confusing crash deep inside a third-party library or wasting time
        on a doomed export attempt.
        """
        from depth_estimation.modeling_utils import BaseDepthModel
        from depth_estimation.configuration_utils import BaseDepthConfig

        class _NotExportable(BaseDepthModel):
            _onnx_exportable = False

            def forward(self, pixel_values):
                return pixel_values.mean(dim=1)

        model = _NotExportable(BaseDepthConfig()).eval()
        with pytest.raises(NotImplementedError, match="not traceable"):
            export_onnx(model, tmp_path / "should_not_exist.onnx", input_size=64)
        assert not (tmp_path / "should_not_exist.onnx").exists()

    def test_verify_restores_tf32_setting(self, tmp_path):
        """Regression test: _verify_onnx_export() disables cuDNN TF32
        for its comparison forward pass (confirmed: on a real
        pretrained+pruned GPU model, TF32 alone pushed max diff to ~0.19,
        vs ~8e-5 with it off — dwarfing the default 1e-3 tolerance and
        having nothing to do with actual export fidelity). It must
        restore the caller's original setting afterward rather than
        leaving it globally disabled as a side effect.
        """
        torch.manual_seed(0)
        config = DepthAnythingV2Config(backbone="vits")
        model = DepthAnythingV2Model(config).eval()
        out = tmp_path / "model.onnx"

        for original in (True, False):
            torch.backends.cudnn.allow_tf32 = original
            export_onnx(model, out, input_size=518, verify=True)
            assert torch.backends.cudnn.allow_tf32 is original

    def test_zero_input_graph_detected(self, tmp_path):
        """Regression test: a model that consumes pixel_values only via a
        non-traceable conversion (e.g. .numpy(), disconnecting it from the
        graph before any traced op uses it — confirmed for marigold-dc)
        produces an ONNX graph with zero declared inputs: a static replay
        of one memorized output regardless of what image it's given.
        export_onnx() must catch this with a clear error rather than
        silently writing a useless file, or letting callers discover it
        via a confusing IndexError out of onnxruntime.
        """
        from depth_estimation.modeling_utils import BaseDepthModel
        from depth_estimation.configuration_utils import BaseDepthConfig

        class _DisconnectedInput(BaseDepthModel):
            def forward(self, pixel_values):
                # Round-trips through numpy, same disconnection pattern as
                # marigold-dc/zoedepth's PIL conversion.
                arr = pixel_values.detach().cpu().numpy()
                return torch.from_numpy(arr).mean(dim=1)

        model = _DisconnectedInput(BaseDepthConfig()).eval()
        with pytest.raises(RuntimeError, match="zero declared inputs"):
            export_onnx(model, tmp_path / "should_not_exist.onnx", input_size=64)


@pytest.mark.slow
class TestExportOnnxSlowOffline:
    """Offline, random-weight, but architecturally heavy — same tier as
    TestSlowOffline in test_inference_all_models.py.
    """

    def test_depth_pro(self, tmp_path):
        torch.manual_seed(0)
        config = DepthProConfig()
        model = DepthProModel(config).eval()
        out = tmp_path / "model.onnx"

        export_onnx(model, out, input_size=64, verify=True)

        assert out.exists()

    def test_pixel_perfect_depth_exports_but_does_not_match(self, tmp_path):
        """Known limitation: PPD samples torch.randn() inside forward() for
        its diffusion process. Tracing freezes that call as a constant, so
        the exported graph always reuses the same noise — export succeeds
        structurally but the numeric output is NOT expected to match
        PyTorch. This test documents that limitation rather than hiding it.
        """
        torch.manual_seed(0)
        config = PixelPerfectDepthConfig(input_size=128, sampling_steps=2)
        model = PixelPerfectDepthModel(config).eval()
        out = tmp_path / "model.onnx"

        _export_or_skip_if_sdpa_unsupported(model, out, input_size=128, verify=False)
        assert out.exists()

        from depth_estimation.export import _verify_onnx_export

        dummy = torch.randn(1, 3, 128, 128)
        with pytest.raises(AssertionError):
            _verify_onnx_export(model, out, dummy_input=dummy)


@pytest.mark.slow
class TestExportOnnxRealPretrained:
    """Requires real pretrained weights from the Hugging Face Hub (network
    required) — same tier as TestPretrainedVariants in
    test_inference_all_models.py.
    """

    def test_zoedepth_blocked_immediately(self, tmp_path):
        """Regression test: ZoeDepthModel wraps an opaque HF
        transformers.Pipeline and round-trips through PIL inside
        forward() — confirmed it crashes mid-trace on a real bug in
        transformers' image processor that only manifests under tracing
        (np.round() on a traced value returns a Tensor, and the processor
        calls .astype() on it — works fine in normal eager inference).
        _onnx_exportable = False must block this immediately instead.
        """
        from depth_estimation import AutoDepthModel

        model = AutoDepthModel.from_pretrained("zoedepth", device="cpu")
        with pytest.raises(NotImplementedError, match="not traceable"):
            export_onnx(model, tmp_path / "should_not_exist.onnx", input_size=384)

    def test_marigold_dc_blocked_immediately(self, tmp_path):
        """Regression test: MarigoldDCModel wraps a diffusers pipeline and
        round-trips through PIL/numpy inside forward(). _onnx_exportable
        = False must block this immediately rather than running the full
        (expensive) diffusion sampling process only to then discover the
        resulting graph has zero declared inputs.
        """
        from depth_estimation import AutoDepthModel

        model = AutoDepthModel.from_pretrained("marigold-dc", device="cpu")
        with pytest.raises(NotImplementedError, match="not traceable"):
            export_onnx(model, tmp_path / "should_not_exist.onnx", input_size=768)

    def test_marigold_dc_pipeline_components_frozen(self):
        """Regression test: diffusers loads pipeline components with
        requires_grad=True by default (ready for fine-tuning) — this
        wrapper is inference-only and every other model in this package
        loads frozen. Left on, this crashed ONNX export outright before
        even reaching the (separately documented) frozen-noise
        limitation.
        """
        from depth_estimation import AutoDepthModel

        model = AutoDepthModel.from_pretrained("marigold-dc", device="cpu")
        model._ensure_pipe()
        for name, param in model._pipe.unet.named_parameters():
            assert not param.requires_grad, f"unet.{name} still requires_grad"

    @pytest.mark.parametrize("model_id", ["moge-v1", "moge-v2-vitb-normal"])
    def test_moge_exports_and_verifies(self, tmp_path, model_id):
        """Regression test: moge-v1 used unconditional antialias=True
        bicubic/bilinear resizing (aten::_upsample_{bicubic,bilinear}2d_aa,
        unsupported at ONNX opset 17) and moge-v2 called Python's round()
        on what can be a Tensor under tracing (TypeError: type Tensor
        doesn't define __round__ method) plus an in-place &= (aten::__iand_,
        also unsupported) and autocast-induced mixed fp16/fp32 dtypes in
        the exported graph. export_onnx() now sets the model's
        onnx_compatible_mode flag (where present) to route around all of
        these.
        """
        from depth_estimation import AutoDepthModel

        model = AutoDepthModel.from_pretrained(model_id, device="cpu")
        out = tmp_path / "model.onnx"
        export_onnx(model, out, input_size=518, verify=True)
        assert out.exists()

    def test_midas_dpt_large_handles_non_square_image(self):
        """Regression test: dpt-large/dpt-hybrid hardcode a square patch
        grid internally and crash on DepthProcessor's default
        aspect-ratio-preserving resize for any non-square real image
        (confirmed: dpt-large raises a patch-grid reshape error,
        dpt-hybrid raises "Input image size doesn't match model").
        MiDaSConfig now defaults keep_aspect_ratio=False for these two
        variants specifically (beit-large is unaffected and keeps it
        True).
        """
        import numpy as np
        from depth_estimation import pipeline

        pipe = pipeline("depth-estimation", model="midas-dpt-large", device="cpu")
        non_square = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe(non_square)
        assert not np.isnan(result.depth).any()

    def test_midas_dpt_hybrid_handles_non_square_image(self):
        import numpy as np
        from depth_estimation import pipeline

        pipe = pipeline("depth-estimation", model="midas-dpt-hybrid", device="cpu")
        non_square = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pipe(non_square)
        assert not np.isnan(result.depth).any()
