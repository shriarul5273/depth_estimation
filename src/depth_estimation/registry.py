"""
Model Registry — Central singleton for registering and resolving model families.

New models register themselves via:
    MODEL_REGISTRY.register("model-type", config_cls, model_cls)

Auto classes use this registry to resolve model identifiers at runtime.
"""

from typing import Dict, Tuple, Type, Optional


class ModelRegistry:
    """Singleton registry mapping model type strings to their config and model classes.

    Every model family added to the library must register:
        1. A BaseDepthConfig subclass
        2. A BaseDepthModel subclass

    The auto-loading layer reads this registry at runtime — no conditional
    import chains or manual if-else mapping required.
    """

    def __init__(self):
        self._configs: Dict[str, Type] = {}
        self._models: Dict[str, Type] = {}
        self._variant_to_type: Dict[str, str] = {}

    def register(
        self,
        model_type: str,
        config_cls: Type,
        model_cls: Type,
        variant_ids: Optional[list] = None,
    ) -> None:
        """Register a model family.

        Args:
            model_type: Unique model family identifier (e.g. "depth-anything-v1").
            config_cls: The DepthConfig subclass for this family.
            model_cls: The DepthModel subclass for this family.
            variant_ids: Optional list of variant identifier strings
                (e.g. ["depth-anything-v1-vits", "depth-anything-v1-vitb"]).
                Each variant is mapped to this model_type for auto-resolution.
        """
        self._configs[model_type] = config_cls
        self._models[model_type] = model_cls

        if variant_ids:
            for vid in variant_ids:
                self._variant_to_type[vid] = model_type

    def resolve_model_type(self, model_id: str) -> str:
        """Resolve a model identifier (variant or type) to its canonical model_type."""
        if model_id in self._configs:
            return model_id
        if model_id in self._variant_to_type:
            return self._variant_to_type[model_id]
        raise ValueError(
            f"Unknown model identifier '{model_id}'. "
            f"Available types: {list(self._configs.keys())}. "
            f"Available variants: {list(self._variant_to_type.keys())}."
        )

    def get_config_cls(self, model_id: str) -> Type:
        """Get the config class for a model identifier."""
        model_type = self.resolve_model_type(model_id)
        return self._configs[model_type]

    def get_model_cls(self, model_id: str) -> Type:
        """Get the model class for a model identifier."""
        model_type = self.resolve_model_type(model_id)
        return self._models[model_type]

    def list_model_types(self) -> list:
        """List all registered model type strings."""
        return list(self._configs.keys())

    def list_variants(self) -> list:
        """List all registered variant identifier strings."""
        return list(self._variant_to_type.keys())


# Global singleton — import this from anywhere
MODEL_REGISTRY = ModelRegistry()
