"""
depth_estimation — A Transformers-style Python library for monocular depth estimation.

Provides a unified, modular API for running, comparing, and integrating
depth estimation models.
"""

from .output import DepthOutput
from .configuration_utils import BaseDepthConfig
from .registry import MODEL_REGISTRY

# Auto classes
from .models.auto import AutoDepthModel, AutoProcessor

# Ensure model modules are imported so they self-register.
# These are now torch-free — modeling classes are loaded lazily on first use.
from .models import depth_anything_v1
from .models import depth_anything_v2
from .models import depth_anything_v3
from .models import zoedepth
from .models import midas
from .models import depth_pro
from .models import pixel_perfect_depth
from .models import marigold_dc
from .models import moge
from .models import omnivggt
from .models import vggt


def load_dataset(name, split="train", root=None, download=True, transform=None, **kwargs):
    """Load a depth dataset by name. See :mod:`depth_estimation.data` for details."""
    from .data import load_dataset as _load
    return _load(name, split=split, root=root, download=download, transform=transform, **kwargs)


def __getattr__(name):
    """Defer torch-heavy imports until first use."""
    if name == "BaseDepthModel":
        from .modeling_utils import BaseDepthModel
        globals()["BaseDepthModel"] = BaseDepthModel
        return BaseDepthModel
    if name == "DepthProcessor":
        from .processing_utils import DepthProcessor
        globals()["DepthProcessor"] = DepthProcessor
        return DepthProcessor
    if name == "DepthPipeline":
        from .pipeline_utils import DepthPipeline
        globals()["DepthPipeline"] = DepthPipeline
        return DepthPipeline
    if name == "pipeline":
        from .pipeline_utils import pipeline
        globals()["pipeline"] = pipeline
        return pipeline
    # Training symbols
    if name == "DepthTrainer":
        from .trainer import DepthTrainer
        globals()["DepthTrainer"] = DepthTrainer
        return DepthTrainer
    if name == "DepthTrainingArguments":
        from .training_args import DepthTrainingArguments
        globals()["DepthTrainingArguments"] = DepthTrainingArguments
        return DepthTrainingArguments
    if name == "CombinedDepthLoss":
        from .losses import CombinedDepthLoss
        globals()["CombinedDepthLoss"] = CombinedDepthLoss
        return CombinedDepthLoss
    if name == "ScaleInvariantLoss":
        from .losses import ScaleInvariantLoss
        globals()["ScaleInvariantLoss"] = ScaleInvariantLoss
        return ScaleInvariantLoss
    if name == "GradientLoss":
        from .losses import GradientLoss
        globals()["GradientLoss"] = GradientLoss
        return GradientLoss
    if name == "BerHuLoss":
        from .losses import BerHuLoss
        globals()["BerHuLoss"] = BerHuLoss
        return BerHuLoss
    if name == "get_train_transforms":
        from .data.transforms import get_train_transforms
        globals()["get_train_transforms"] = get_train_transforms
        return get_train_transforms
    if name == "get_val_transforms":
        from .data.transforms import get_val_transforms
        globals()["get_val_transforms"] = get_val_transforms
        return get_val_transforms
    # Video & streaming
    if name == "VideoStream":
        from .video import VideoStream
        globals()["VideoStream"] = VideoStream
        return VideoStream
    if name == "process_video":
        from .video import process_video
        globals()["process_video"] = process_video
        return process_video
    # Visualization toolkit
    if name == "viz":
        from . import viz
        globals()["viz"] = viz
        return viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DepthOutput",
    "BaseDepthConfig",
    "BaseDepthModel",
    "DepthProcessor",
    "DepthPipeline",
    "pipeline",
    "AutoDepthModel",
    "AutoProcessor",
    "MODEL_REGISTRY",
    "load_dataset",
    # Training
    "DepthTrainer",
    "DepthTrainingArguments",
    "CombinedDepthLoss",
    "ScaleInvariantLoss",
    "GradientLoss",
    "BerHuLoss",
    "get_train_transforms",
    "get_val_transforms",
    # Video & streaming
    "VideoStream",
    "process_video",
    # Visualization
    "viz",
]

__version__ = "0.0.9"
