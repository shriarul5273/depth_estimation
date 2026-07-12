"""Tests for depth_estimation.quantization."""

import torch
import pytest

from depth_estimation.quantization import quantize_model, quantize_onnx
from depth_estimation.models.depth_anything_v2.configuration_depth_anything_v2 import (
    DepthAnythingV2Config,
)
from depth_estimation.models.depth_anything_v2.modeling_depth_anything_v2 import (
    DepthAnythingV2Model,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = DepthAnythingV2Config(backbone="vits")
    return DepthAnythingV2Model(config).eval()


class TestQuantizeModel:
    def test_float16_casts_in_place(self, model):
        result = quantize_model(model, dtype="float16")
        assert result is model
        assert next(model.parameters()).dtype == torch.float16

    def test_bfloat16_casts_in_place(self, model):
        result = quantize_model(model, dtype="bfloat16")
        assert result is model
        assert next(model.parameters()).dtype == torch.bfloat16

    def test_float16_forward_works(self, model):
        """float16 CPU conv2d isn't implemented on some torch versions
        (confirmed: torch==2.0.1 CPU raises RuntimeError from
        aten::slow_conv2d/thnn_conv2d — no Half CPU kernel). This is a
        real torch-version limitation, not a quantize_model() bug — the
        cast itself (test_float16_casts_in_place) always works; only
        actually running a CPU forward pass afterward is version-gated.
        docs/quantization.md already documents float16 as GPU-oriented for
        exactly this reason. Skip rather than fail on old torch.
        """
        quantize_model(model, dtype="float16")
        dummy = torch.randn(1, 3, 518, 518).half()
        try:
            with torch.no_grad():
                out = model(dummy)
        except RuntimeError as e:
            pytest.skip(
                f"torch {torch.__version__} CPU doesn't implement this op for "
                f"float16 (known limitation on older torch): {e}"
            )
        assert out.shape == (1, 518, 518)
        assert not torch.isnan(out).any()

    def test_int8_returns_new_object_same_type(self, model):
        result = quantize_model(model, dtype="int8")
        assert result is not model
        assert type(result) is type(model)

    def test_int8_forward_works(self, model):
        qmodel = quantize_model(model, dtype="int8")
        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = qmodel(dummy)
        assert out.shape == (1, 518, 518)
        assert not torch.isnan(out).any()

    def test_original_model_unaffected_by_int8(self, model):
        original_dtype = next(model.parameters()).dtype
        quantize_model(model, dtype="int8")
        assert next(model.parameters()).dtype == original_dtype

    def test_int8_result_always_on_cpu(self, model):
        qmodel = quantize_model(model, dtype="int8")
        assert next(qmodel.parameters()).device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
    def test_int8_from_cuda_model_works_and_leaves_original_on_cuda(self, model):
        """Regression test: torch's dynamic quantization only has CPU
        kernels for the quantized linear op it produces. Quantizing a
        CUDA-resident model directly used to either crash immediately
        (NotImplementedError: 'quantized::linear_dynamic' ... 'CUDA'
        backend) or, if "fixed" by naively calling model.to("cpu") in
        quantize_model() itself (an earlier, wrong version of this fix),
        silently move the CALLER's original model to CPU as a side
        effect too — nn.Module.to() mutates and returns self. Confirmed
        both failure modes by hand before landing the deepcopy-based fix
        below. Only runs where a GPU is actually available (CI is
        CPU-only), so this won't execute there — verified locally.
        """
        cuda_model = model.to("cuda")
        qmodel = quantize_model(cuda_model, dtype="int8")

        assert next(cuda_model.parameters()).device.type == "cuda"

        dummy = torch.randn(1, 3, 518, 518)
        with torch.no_grad():
            out = qmodel(dummy)
        assert out.shape == (1, 518, 518)
        assert not torch.isnan(out).any()

    def test_int16_raises_with_helpful_message(self, model):
        with pytest.raises(ValueError, match="not supported by PyTorch"):
            quantize_model(model, dtype="int16")

    def test_uint16_raises_with_helpful_message(self, model):
        with pytest.raises(ValueError, match="not supported by PyTorch"):
            quantize_model(model, dtype="uint16")

    def test_unknown_dtype_raises(self, model):
        with pytest.raises(ValueError, match="Unknown dtype"):
            quantize_model(model, dtype="not_a_real_dtype")

    def test_int8_result_cannot_export_to_onnx(self, model, tmp_path):
        """Documented, confirmed incompatibility: torch's ONNX exporter has
        no symbolic mapping for the quantized::linear_dynamic op that
        quantize_model(dtype="int8") produces, at any opset. The correct
        path for a quantized ONNX file is always export first, then
        quantize_onnx() — never quantize_model(dtype="int8") then export.
        """
        pytest.importorskip("onnx")
        from depth_estimation.export import export_onnx

        qmodel = quantize_model(model, dtype="int8")
        with pytest.raises(Exception, match="quantized::linear_dynamic"):
            export_onnx(qmodel, tmp_path / "should_not_work.onnx", input_size=518)


class TestQuantizeOnnx:
    """Requires the optional onnx/onnxruntime packages — skipped entirely if
    they're not installed (they aren't core dependencies).
    """

    @pytest.fixture
    def onnx_path(self, model, tmp_path):
        pytest.importorskip("onnx")
        from depth_estimation.export import export_onnx

        out = tmp_path / "base.onnx"
        export_onnx(model, out, input_size=518)
        return out

    def test_int8_quantizes_and_verifies(self, onnx_path, tmp_path):
        """int8 quantization of Conv2d layers (every model here has at least
        one, in its patch embedding) produces a ConvInteger op. Older
        onnxruntime CPU builds don't implement it at all — confirmed:
        onnxruntime==1.23.2 (the newest available for Python 3.10 at time
        of writing) raises "NOT_IMPLEMENTED: ... ConvInteger(10)"; verified
        working on onnxruntime==1.26.0+. uint8 (test_uint8_quantizes_and_verifies)
        is unaffected — it produces different, more broadly-supported ops.
        Skip rather than fail on old onnxruntime.
        """
        pytest.importorskip("onnxruntime")
        out = tmp_path / "quant_int8.onnx"
        try:
            result = quantize_onnx(onnx_path, out, weight_type="int8", verify=True)
        except Exception as e:
            if "ConvInteger" in str(e) or "NOT_IMPLEMENTED" in str(e):
                import onnxruntime as ort

                pytest.skip(
                    f"onnxruntime {ort.__version__}'s CPU execution provider "
                    f"doesn't implement an op needed for int8 Conv2d "
                    f"quantization (known limitation on older onnxruntime): {e}"
                )
            raise
        assert result == out
        assert out.exists()

    def test_uint8_quantizes_and_verifies(self, onnx_path, tmp_path):
        pytest.importorskip("onnxruntime")
        out = tmp_path / "quant_uint8.onnx"
        quantize_onnx(onnx_path, out, weight_type="uint8", verify=True)
        assert out.exists()

    def test_int8_actually_shrinks_file(self, onnx_path, tmp_path):
        pytest.importorskip("onnxruntime")
        out = tmp_path / "quant_int8.onnx"
        quantize_onnx(onnx_path, out, weight_type="int8")
        assert out.stat().st_size < onnx_path.stat().st_size

    def test_int16_verify_raises_known_limitation(self, onnx_path, tmp_path):
        """Documented, confirmed limitation: int16 quantization "succeeds"
        (writes a file) but the result fails to load in onnxruntime's CPU
        execution provider. verify=True must surface this, not hide it.
        """
        pytest.importorskip("onnxruntime")
        out = tmp_path / "quant_int16.onnx"
        with pytest.raises(Exception):
            quantize_onnx(onnx_path, out, weight_type="int16", verify=True)

    def test_unknown_weight_type_raises(self, onnx_path, tmp_path):
        with pytest.raises(ValueError, match="Unknown weight_type"):
            quantize_onnx(onnx_path, tmp_path / "out.onnx", weight_type="fp8")
