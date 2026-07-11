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
        quantize_model(model, dtype="float16")
        dummy = torch.randn(1, 3, 518, 518).half()
        with torch.no_grad():
            out = model(dummy)
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

    def test_int16_raises_with_helpful_message(self, model):
        with pytest.raises(ValueError, match="not supported by PyTorch"):
            quantize_model(model, dtype="int16")

    def test_uint16_raises_with_helpful_message(self, model):
        with pytest.raises(ValueError, match="not supported by PyTorch"):
            quantize_model(model, dtype="uint16")

    def test_unknown_dtype_raises(self, model):
        with pytest.raises(ValueError, match="Unknown dtype"):
            quantize_model(model, dtype="not_a_real_dtype")


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
        pytest.importorskip("onnxruntime")
        out = tmp_path / "quant_int8.onnx"
        result = quantize_onnx(onnx_path, out, weight_type="int8", verify=True)
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
