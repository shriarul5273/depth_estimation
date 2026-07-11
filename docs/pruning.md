# Pruning

Zero out a fraction of a model's weights using PyTorch's built-in `torch.nn.utils.prune` — pure CPU/GPU-agnostic, no special hardware or SDK required (unlike, say, TensorRT).

## Quick Example

```python
from depth_estimation import AutoDepthModel, prune_model, compute_sparsity

model = AutoDepthModel.from_pretrained("depth-anything-v2-vitb")
prune_model(model, amount=0.3)
print(compute_sparsity(model)["overall"])  # ~0.3
```

Or via `BaseDepthModel.prune()`:

```python
model.prune(amount=0.3)
```

Pruned models export to ONNX exactly like any other model — no special handling needed in `export_onnx()`:

```python
model.prune(amount=0.3).export_onnx("pruned.onnx", verify=True)
```

## What pruning actually does (and doesn't do)

Pruning zeros out individual weight *values* — it does not change tensor shapes. A pruned model is **not** smaller on disk or faster on generic hardware/runtimes by itself: dense zeros still take the same space and FLOPs as dense non-zeros. What it buys you:

- A smaller *effective* parameter count, useful as a foundation for further compression (sparse-aware runtimes, quantization stacked on top).
- For structured pruning, an actual size/speed reduction — but only if you follow up by slicing out the pruned channels yourself. That follow-up step isn't implemented here; `prune_model()` only does unstructured (per-weight) pruning.

## `prune_model()`

```python
prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = "l1_unstructured",
    module_types: tuple[type, ...] = (nn.Linear, nn.Conv2d),
    exclude: list[str] | None = None,
    make_permanent: bool = True,
) -> nn.Module
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | **required** | Any model — typically a `BaseDepthModel` subclass, but this works on any model since it only depends on standard `nn.Linear`/`nn.Conv2d` layers. |
| `amount` | `float` | `0.3` | Fraction of weights to zero out per layer, in `[0, 1)`. `0.3` zeros the smallest-magnitude 30% of each layer's weights (for the default `"l1_unstructured"` method). |
| `method` | `str` | `"l1_unstructured"` | `"l1_unstructured"` (magnitude-based — zeros the smallest-\|weight\| entries, generally preserves accuracy better) or `"random_unstructured"` (zeros a random subset — useful as a baseline/sanity check, not for real deployment). |
| `module_types` | `tuple[type, ...]` | `(nn.Linear, nn.Conv2d)` | Which layer types to prune. Defaults to every `Linear` and `Conv2d` in the model — covers attention projections, MLP layers, and conv-based patch embeddings/decoders across every model family in this package. |
| `exclude` | `list[str]` or `None` | `None` | Submodule dotted-name substrings to skip. E.g. `exclude=["patch_embed"]` to leave the input projection dense. |
| `make_permanent` | `bool` | `True` | Bakes the zeroed weights directly into each layer's `.weight` tensor and removes PyTorch's pruning reparameterization. **Required for a clean `export_onnx()`** — an un-removed reparameterization adds an extra mask-multiply op and mask buffers to the traced graph. Set `False` only if you plan to keep training with the mask actively enforced (see below). |

Returns `model`, mutated in-place (also returned, for chaining — `model.prune(0.3).export_onnx(...)`).

Raises `ValueError` for an unrecognized `method` or an `amount` outside `[0, 1)`.

## `compute_sparsity()`

```python
compute_sparsity(
    model: nn.Module,
    module_types: tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> dict[str, float]
```

Reports the fraction of zero-valued weights, per layer and overall (`"overall"` key, weighted by parameter count). Reports on **every** module matching `module_types` — not just ones `prune_model()` touched, since there's no reliable way to detect that after `make_permanent=True` removes pruning's own bookkeeping. A layer that was never pruned simply shows ~0.0.

Works whether or not pruning was made permanent.

## `make_pruning_permanent()`

```python
make_pruning_permanent(
    model: nn.Module,
    module_types: tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> nn.Module
```

Bakes any active pruning reparameterization into `.weight` directly — the standalone version of what `prune_model(..., make_permanent=True)` does automatically. Use this to finish a prune-aware fine-tuning workflow (below). Returns `model`, mutated in-place.

Note it deliberately only checks `module_types` instances for `torch.nn.utils.prune.is_pruned()`, not every submodule — checking arbitrary container modules can raise `ValueError: ... has to be pruned before pruning can be removed`, since `is_pruned()` isn't reliably scoped to "this exact module has a pruned `weight` parameter."

## Prune-Aware Fine-Tuning

To keep training with the sparsity mask actively enforced (each optimizer step re-applies the mask via the underlying `weight_orig`, so pruned weights stay at zero through training rather than drifting):

```python
from depth_estimation import make_pruning_permanent

model.prune(amount=0.3, make_permanent=False)
# ... fine-tune with DepthTrainer as normal — see docs/training.md ...

# Once training is done, bake the mask in before export:
make_pruning_permanent(model)
model.export_onnx("pruned_finetuned.onnx", verify=True)
```
