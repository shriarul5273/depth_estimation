# Adding a New Model

This guide shows how to add a new depth estimation model family to the library. The process follows the **Transformers modular pattern**: inherit from base classes, override only what's different, and register.

## Step 1: Create the directory

```
src/depth_estimation/models/your_model/
├── __init__.py
├── configuration_your_model.py
└── modeling_your_model.py
```

## Step 2: Configuration

Inherit from `BaseDepthConfig` and override default values:

```python
# configuration_your_model.py
from ...configuration_utils import BaseDepthConfig

_VARIANT_MAP = {
    "your-model-small": "small",
    "your-model-large": "large",
}

class YourModelConfig(BaseDepthConfig):
    model_type = "your-model"

    def __init__(self, backbone="large", input_size=512, **kwargs):
        super().__init__(backbone=backbone, input_size=input_size, **kwargs)

    @classmethod
    def from_variant(cls, variant_id):
        backbone = _VARIANT_MAP[variant_id]
        return cls(backbone=backbone)
```

## Step 3: Model (single file)

Inherit from `BaseDepthModel`. Inline all architecture in one file. Implement `forward()` and `_load_pretrained_weights()`:

```python
# modeling_your_model.py
from ...modeling_utils import BaseDepthModel
from .configuration_your_model import YourModelConfig

class YourModel(BaseDepthModel):
    config_class = YourModelConfig

    def __init__(self, config):
        super().__init__(config)
        # Build your architecture here

    def forward(self, pixel_values):
        # Return depth tensor (B, H, W)
        ...

    @classmethod
    def _load_pretrained_weights(cls, model_id, device="cpu", **kwargs):
        config = YourModelConfig.from_variant(model_id)
        model = cls(config)
        # Load your weights here
        return model
```

## Step 4: Register

In `__init__.py`, register with the global registry:

```python
# __init__.py
from .configuration_your_model import YourModelConfig, _VARIANT_MAP
from .modeling_your_model import YourModel
from ...registry import MODEL_REGISTRY

MODEL_REGISTRY.register(
    model_type="your-model",
    config_cls=YourModelConfig,
    model_cls=YourModel,
    variant_ids=list(_VARIANT_MAP.keys()),
)
```

## Step 5: Import in top-level

Add one line to `src/depth_estimation/__init__.py`:

```python
from .models import your_model  # noqa: F401
```

## Done!

Your model now works with:

```python
from depth_estimation import pipeline, AutoDepthModel, AutoProcessor

pipe = pipeline("depth-estimation", model="your-model-large")
model = AutoDepthModel.from_pretrained("your-model-large")
processor = AutoProcessor.from_pretrained("your-model-large")
```

No changes to any core files needed. The shared `DepthProcessor` handles preprocessing/postprocessing automatically.
