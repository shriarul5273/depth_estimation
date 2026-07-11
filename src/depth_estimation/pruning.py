"""Weight pruning for depth estimation models.

Uses PyTorch's built-in ``torch.nn.utils.prune`` — pure CPU/GPU-agnostic,
no special hardware or SDK required (unlike, say, TensorRT).

Usage::

    from depth_estimation import AutoDepthModel
    from depth_estimation.pruning import prune_model, compute_sparsity

    model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")
    prune_model(model, amount=0.3)
    print(compute_sparsity(model)["overall"])  # ~0.3

    # Pruned models export to ONNX exactly like any other model — pruning
    # bakes zeroed weights directly into the tensors (see make_permanent
    # below), so export_onnx() needs no special handling.
    model.export_onnx("pruned.onnx", verify=True)

Pruning zeros out individual weight *values* — it does not change tensor
shapes, so a pruned model is not smaller on disk or faster on generic
hardware/runtimes by itself (dense zeros still take the same space and
FLOPs as dense non-zeros). What it buys you is a smaller *effective*
parameter count for further compression (e.g. sparse-aware runtimes,
quantization stacked on top) and, for structured pruning, an actual
reduction in tensor size if you follow up by slicing out the pruned
channels — that follow-up step isn't implemented here.
"""

import logging
from typing import Dict, Optional, Sequence, Tuple, Type

import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)

_METHOD_MAP = {
    "l1_unstructured": prune.l1_unstructured,
    "random_unstructured": prune.random_unstructured,
}


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = "l1_unstructured",
    module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    exclude: Optional[Sequence[str]] = None,
    make_permanent: bool = True,
) -> nn.Module:
    """Prune a fraction of weights in every matching submodule, in-place.

    Args:
        model: Any ``nn.Module`` — typically a ``BaseDepthModel`` subclass,
            but this works on any model since it only depends on standard
            ``nn.Linear``/``nn.Conv2d`` layers.
        amount: Fraction of weights to zero out per layer, in ``[0, 1)``.
            ``0.3`` zeros the smallest-magnitude 30% of each layer's
            weights (for the default ``"l1_unstructured"`` method).
        method: ``"l1_unstructured"`` (magnitude-based — zeros the
            smallest-|weight| entries, generally preserves accuracy
            better) or ``"random_unstructured"`` (zeros a random subset —
            useful as a baseline/sanity check, not for real deployment).
        module_types: Which layer types to prune. Defaults to every
            ``Linear`` and ``Conv2d`` in the model (covers attention
            projections, MLP layers, and conv-based patch embeddings /
            decoders across every model family in this package).
        exclude: Optional list of substrings — any submodule whose
            dotted name contains one of these is skipped. E.g.
            ``exclude=["patch_embed"]`` to leave the input projection
            dense.
        make_permanent: If True (default), bakes the zeroed weights
            directly into each layer's ``.weight`` tensor and removes
            PyTorch's pruning reparameterization (the ``weight_orig`` /
            ``weight_mask`` buffers and forward pre-hook that
            ``torch.nn.utils.prune`` installs). This is required for a
            clean `export_onnx()` — an un-removed reparameterization adds
            an extra mask-multiply op and mask buffers to the traced
            graph — and is also what you want once you're done
            prune-aware fine-tuning. Set False only if you plan to keep
            training with the mask actively enforced (each optimizer
            step would otherwise "unprune" masked weights via the
            underlying ``weight_orig``).

    Returns:
        ``model``, mutated in-place (returned for chaining).

    Raises:
        ValueError: If ``method`` isn't a recognized pruning method, or
            ``amount`` is outside ``[0, 1)``.
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown pruning method {method!r}. Available: {list(_METHOD_MAP)}"
        )
    if not (0.0 <= amount < 1.0):
        raise ValueError(f"amount must be in [0, 1), got {amount}")

    prune_fn = _METHOD_MAP[method]
    exclude = exclude or []

    pruned_modules = []
    for name, module in model.named_modules():
        if not isinstance(module, module_types):
            continue
        if any(pattern in name for pattern in exclude):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        prune_fn(module, name="weight", amount=amount)
        pruned_modules.append((name, module))

    if not pruned_modules:
        logger.warning(
            "prune_model: no modules matched (module_types=%s, exclude=%s) — "
            "nothing was pruned.",
            module_types,
            exclude,
        )

    if make_permanent:
        for _, module in pruned_modules:
            prune.remove(module, "weight")

    logger.info(
        "Pruned %d modules at amount=%.2f (method=%s, permanent=%s)",
        len(pruned_modules),
        amount,
        method,
        make_permanent,
    )
    return model


def make_pruning_permanent(
    model: nn.Module,
    module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
) -> nn.Module:
    """Bake any active pruning reparameterization into ``.weight`` directly.

    For finishing a prune-aware fine-tuning workflow: call
    ``prune_model(model, ..., make_permanent=False)``, fine-tune with the
    mask actively enforced, then call this once training is done — before
    ``export_onnx()``, which needs the reparameterization removed for a
    clean graph.

    Note: iterates only over ``module_types`` before checking
    ``torch.nn.utils.prune.is_pruned()`` — checking arbitrary container
    modules (rather than the exact ``nn.Linear``/``nn.Conv2d`` instances
    pruning was applied to) can raise ``ValueError: ... has to be pruned
    before pruning can be removed``, since ``is_pruned()`` isn't reliably
    scoped to "this exact module has a pruned 'weight' parameter".

    Args:
        model: Any model that may have modules with active pruning
            reparameterization (i.e. ``prune_model(..., make_permanent=False)``
            was used on it, or on a subset of it).
        module_types: Which layer types to check. Must match (or be a
            superset of) whatever ``module_types`` was used when pruning
            was originally applied.

    Returns:
        ``model``, mutated in-place (returned for chaining).
    """
    for _, module in model.named_modules():
        if isinstance(module, module_types) and prune.is_pruned(module):
            prune.remove(module, "weight")
    return model


def compute_sparsity(
    model: nn.Module,
    module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, float]:
    """Report the fraction of zero-valued weights, per layer and overall.

    Reports on every module matching ``module_types`` (default: every
    ``Linear``/``Conv2d``, matching :func:`prune_model`'s default) —
    not just ones that were actually pruned. A layer that was never
    pruned simply shows ~0.0. This is deliberately not restricted to
    "layers prune_model touched", since there's no reliable way to
    detect that after ``make_permanent=True`` has removed pruning's own
    bookkeeping (``prune.is_pruned()`` returns False once removed) —
    guessing from "does this layer merely contain a zero" would
    misclassify any never-pruned layer that happens to have one.

    Works whether or not pruning was made permanent — it just counts
    zeros in the *effective* weight (``module.weight``, which already
    reflects the mask if pruning is still reparameterized).

    Returns:
        Dict mapping each matching submodule's dotted name to its
        sparsity fraction, plus an ``"overall"`` key for the model-wide
        fraction (weighted by parameter count across all matching layers).
    """
    result: Dict[str, float] = {}
    total_params = 0
    total_zeros = 0

    for name, module in model.named_modules():
        if not isinstance(module, module_types):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        weight = module.weight.detach()
        n = weight.numel()
        if n == 0:
            continue
        zeros = int((weight == 0).sum().item())
        result[name] = zeros / n
        total_params += n
        total_zeros += zeros

    result["overall"] = (total_zeros / total_params) if total_params else 0.0
    return result
