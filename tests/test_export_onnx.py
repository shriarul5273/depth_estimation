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
