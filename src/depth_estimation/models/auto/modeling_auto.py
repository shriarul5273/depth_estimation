"""
AutoDepthModel — Automatic model resolution via the global registry.
"""

from typing import Any, Optional

from ...registry import MODEL_REGISTRY


class AutoDepthModel:
    """Resolves a model identifier to the correct DepthModel subclass and loads weights.

    Usage::

        from depth_estimation import AutoDepthModel

        model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")
        # Internally instantiates DepthAnythingV2Model
    """

    def __init__(self):
        raise RuntimeError(
            "AutoDepthModel is not meant to be instantiated directly. "
            "Use AutoDepthModel.from_pretrained(model_id) instead."
        )

    @staticmethod
    def from_pretrained(
        model_id: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """Load a pretrained depth estimation model.

        Args:
            model_id: Model identifier (e.g. "depth-anything-v2-vitb").
            device: Device to load to. Auto-detected if None.
            **kwargs: Additional args passed to the model's from_pretrained().

        Returns:
            Instantiated model with pretrained weights.
        """
        model_cls = MODEL_REGISTRY.get_model_cls(model_id)
        return model_cls.from_pretrained(model_id, device=device, **kwargs)
