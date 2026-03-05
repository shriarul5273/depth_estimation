"""
AutoProcessor — Automatic processor resolution via the global registry.
"""

from typing import Any, Optional

from ...registry import MODEL_REGISTRY
from ...processing_utils import DepthProcessor


class AutoProcessor:
    """Resolves a model identifier to a correctly configured DepthProcessor.

    Usage::

        from depth_estimation import AutoProcessor

        processor = AutoProcessor.from_pretrained("depth-anything-v2-vitb")
    """

    def __init__(self):
        raise RuntimeError(
            "AutoProcessor is not meant to be instantiated directly. "
            "Use AutoProcessor.from_pretrained(model_id) instead."
        )

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any) -> DepthProcessor:
        """Create a DepthProcessor configured for the given model.

        Args:
            model_id: Model identifier (e.g. "depth-anything-v2-vitb").
            **kwargs: Additional args passed to the processor.

        Returns:
            DepthProcessor configured with the correct model config.
        """
        config_cls = MODEL_REGISTRY.get_config_cls(model_id)

        # Resolve variant → backbone for from_variant()
        if hasattr(config_cls, "from_variant"):
            try:
                config = config_cls.from_variant(model_id)
            except (ValueError, KeyError):
                config = config_cls()
        else:
            config = config_cls()

        return DepthProcessor.from_config(config)
