"""Top-level evaluation functions: evaluate() and compare()."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .metrics import Evaluator, align_least_squares

logger = logging.getLogger(__name__)

_METRIC_HEADERS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
_DATASET_ALIASES = {
    "nyu":   "nyu_depth_v2",
    "kitti": "kitti_eigen",
}


# ---------------------------------------------------------------------------
# Internal wrapper dataset
# ---------------------------------------------------------------------------

class _EvalDataset(Dataset):
    """Wraps a BaseDepthDataset and applies model-specific preprocessing.

    Stores both:
    - ``pixel_values``: image preprocessed by the model's DepthProcessor
      (correct input_size, ImageNet normalisation).
    - ``depth_map`` / ``valid_mask``: ground truth at native resolution.
    """

    def __init__(self, base_dataset, processor) -> None:
        self.base = base_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        image_np, depth_np = self.base._load_sample(index)  # HxWx3, HxW

        # Model input — properly resized and normalised
        inputs = self.processor.preprocess(image_np)
        pixel_values = inputs["pixel_values"][0]          # (3, H_m, W_m)
        orig_h, orig_w = inputs["original_sizes"][0]

        # Ground-truth tensors at native resolution
        depth_t = torch.from_numpy(depth_np).float()       # (H, W)
        valid_mask = (
            (depth_t > self.base.min_depth)
            & (depth_t < self.base.max_depth)
        )

        return {
            "pixel_values": pixel_values,         # (3, H_m, W_m)
            "depth_map":    depth_t,              # (H, W)
            "valid_mask":   valid_mask,           # (H, W) bool
            "orig_h":       orig_h,
            "orig_w":       orig_w,
        }


def _collate(batch: list) -> dict:
    """Custom collate that handles variable-size depth maps."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    orig_h = [b["orig_h"] for b in batch]
    orig_w = [b["orig_w"] for b in batch]
    depth_maps   = [b["depth_map"]   for b in batch]
    valid_masks  = [b["valid_mask"]  for b in batch]
    return {
        "pixel_values": pixel_values,
        "depth_maps":   depth_maps,
        "valid_masks":  valid_masks,
        "orig_h":       orig_h,
        "orig_w":       orig_w,
    }


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def evaluate(
    model: Union[str, "BaseDepthModel"],  # noqa: F821
    dataset: Union[str, "BaseDepthDataset"],  # noqa: F821
    split: str = "test",
    dataset_root: Optional[str] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    num_workers: int = 4,
    align: bool = True,
    num_samples: Optional[int] = None,
    **dataset_kwargs: Any,
) -> Dict[str, float]:
    """Evaluate a depth model on a dataset and return standard metrics.

    Args:
        model:          Model variant ID (e.g. ``"depth-anything-v2-vitb"``) or a
                        loaded :class:`BaseDepthModel` instance.
        dataset:        Dataset name (``"nyu_depth_v2"``, ``"diode"``,
                        ``"kitti_eigen"``) or a :class:`BaseDepthDataset` instance.
                        Shorthand aliases ``"nyu"`` and ``"kitti"`` are also accepted.
        split:          ``"train"``, ``"val"``, or ``"test"``. Default ``"test"``.
        dataset_root:   Root directory for the dataset (when *dataset* is a string).
        batch_size:     Images per forward pass. Default 1.
        device:         Device string. Auto-detected if ``None``.
        num_workers:    DataLoader workers. Default 4. Use 0 on Windows with h5py.
        align:          For relative models (``config.is_metric=False``), apply
                        per-sample least-squares scale+shift alignment before
                        computing metrics. Default ``True``.
        num_samples:    Limit the number of evaluated samples. Useful for quick
                        sanity checks. ``None`` evaluates all samples.
        **dataset_kwargs: Extra kwargs forwarded to :func:`load_dataset` when
                        *dataset* is a string.

    Returns:
        Dict with keys ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log``,
        ``delta1``, ``delta2``, ``delta3``, ``n_pixels``, ``n_samples``.

    Example::

        from depth_estimation.evaluation import evaluate

        # Quick check on 100 samples
        results = evaluate("depth-anything-v2-vitb", "nyu_depth_v2",
                           num_samples=100)
        print(f"AbsRel: {results['abs_rel']:.4f}  δ₁: {results['delta1']:.4f}")
    """
    from depth_estimation.models.auto.modeling_auto import AutoDepthModel
    from depth_estimation.models.auto.processing_auto import AutoProcessor
    from depth_estimation.modeling_utils import _auto_detect_device
    from depth_estimation.data import load_dataset
    from depth_estimation.data.base_dataset import BaseDepthDataset

    # ------------------------------------------------------------------
    # Resolve device
    # ------------------------------------------------------------------
    if device is None:
        device = _auto_detect_device()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if isinstance(model, str):
        logger.info("Loading model %s …", model)
        loaded_model = AutoDepthModel.from_pretrained(model, device=device)
        processor = AutoProcessor.from_pretrained(model)
    else:
        loaded_model = model.to(device)
        processor = AutoProcessor.from_pretrained(loaded_model.config.model_type)

    loaded_model.eval()
    is_metric = getattr(loaded_model.config, "is_metric", False)
    do_align = align and not is_metric

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    if isinstance(dataset, str):
        name = _DATASET_ALIASES.get(dataset, dataset)
        kwargs = dict(split=split)
        if dataset_root is not None:
            kwargs["root"] = dataset_root
        kwargs.update(dataset_kwargs)
        logger.info("Loading dataset %s (split=%s) …", name, split)
        base_ds = load_dataset(name, **kwargs)
    else:
        base_ds = dataset

    if num_samples is not None:
        # Wrap with a Subset
        from torch.utils.data import Subset
        indices = list(range(min(num_samples, len(base_ds))))
        base_ds = _SubsetWrapper(base_ds, indices)

    eval_ds = _EvalDataset(base_ds, processor)
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(device.startswith("cuda")),
    )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    ev = Evaluator()
    n_samples = 0

    try:
        from tqdm import tqdm
        iterable = tqdm(loader, desc=f"Evaluating", unit="batch", leave=True)
    except ImportError:
        iterable = loader

    with torch.no_grad():
        for batch in iterable:
            pixel_values = batch["pixel_values"].to(device)
            depth_maps   = batch["depth_maps"]    # list of (H_orig, W_orig) tensors
            valid_masks  = batch["valid_masks"]
            orig_hs      = batch["orig_h"]
            orig_ws      = batch["orig_w"]

            # Forward pass
            raw_preds = loaded_model(pixel_values)   # (B, H_m, W_m) or (B, 1, H_m, W_m)
            if raw_preds.dim() == 4:
                raw_preds = raw_preds.squeeze(1)

            B = raw_preds.shape[0]
            for b in range(B):
                oh, ow = int(orig_hs[b]), int(orig_ws[b])

                # Resize prediction to GT resolution
                pred_resized = F.interpolate(
                    raw_preds[b].unsqueeze(0).unsqueeze(0),
                    size=(oh, ow),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]  # (H_orig, W_orig)

                gt   = depth_maps[b].to(device)    # (H_orig, W_orig)
                mask = valid_masks[b].to(device)   # (H_orig, W_orig) bool

                # Align relative predictions to GT scale
                if do_align and mask.sum() >= 2:
                    pred_np = pred_resized.cpu().numpy()
                    gt_np   = gt.cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    scale, shift = align_least_squares(pred_np, gt_np, mask_np)
                    pred_resized = (scale * pred_resized + shift).clamp(min=1e-6)

                ev.update(
                    pred_resized.unsqueeze(0),
                    gt.unsqueeze(0),
                    mask.unsqueeze(0),
                )
                n_samples += 1

    results = ev.compute()
    results["n_samples"] = n_samples
    return results


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

def compare(
    models: List[str],
    dataset: Union[str, "BaseDepthDataset"],  # noqa: F821
    split: str = "test",
    dataset_root: Optional[str] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    num_workers: int = 4,
    align: bool = True,
    num_samples: Optional[int] = None,
    print_table: bool = True,
    **dataset_kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple models on the same dataset and print a comparison table.

    Args:
        models:       List of model variant ID strings.
        dataset:      Dataset name or :class:`BaseDepthDataset` instance.
        split:        Dataset split. Default ``"test"``.
        dataset_root: Root directory (when *dataset* is a string).
        batch_size:   Images per forward pass.
        device:       Device string. Auto-detected if ``None``.
        num_workers:  DataLoader workers.
        align:        Align relative-depth predictions before metrics.
        num_samples:  Limit evaluated samples per model.
        print_table:  Print a formatted comparison table. Default ``True``.
        **dataset_kwargs: Forwarded to :func:`load_dataset`.

    Returns:
        ``{model_id: metrics_dict}`` — same structure as :func:`evaluate`.

    Example::

        from depth_estimation.evaluation import compare

        results = compare(
            ["depth-anything-v2-vits",
             "depth-anything-v2-vitb",
             "depth-anything-v2-vitl"],
            dataset="nyu_depth_v2",
        )
    """
    from depth_estimation.data import load_dataset
    from depth_estimation.data.base_dataset import BaseDepthDataset

    # Load dataset once and reuse across models
    if isinstance(dataset, str):
        name = _DATASET_ALIASES.get(dataset, dataset)
        kwargs: Dict[str, Any] = dict(split=split)
        if dataset_root is not None:
            kwargs["root"] = dataset_root
        kwargs.update(dataset_kwargs)
        logger.info("Loading dataset %s (split=%s) …", name, split)
        base_ds = load_dataset(name, **kwargs)
    else:
        base_ds = dataset

    all_results: Dict[str, Dict[str, float]] = {}

    for model_id in models:
        logger.info("Evaluating %s …", model_id)
        try:
            result = evaluate(
                model=model_id,
                dataset=base_ds,
                split=split,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                align=align,
                num_samples=num_samples,
            )
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", model_id, exc)
            result = {k: float("nan") for k in _METRIC_HEADERS}
            result["n_samples"] = 0

        all_results[model_id] = result

    if print_table:
        _print_table(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _print_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table to stdout."""
    lower = ["abs_rel", "sq_rel", "rmse", "rmse_log"]
    higher = ["delta1", "delta2", "delta3"]
    headers = lower + higher

    col_w = max(len(m) for m in results) + 2
    num_w = 10

    # Header row
    header = f"{'Model':<{col_w}}" + "".join(f"{h:>{num_w}}" for h in headers)
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    # Direction hint row
    hint = f"{'':>{col_w}}" + "".join(f"{'(↓)':>{num_w}}" if h in lower else f"{'(↑)':>{num_w}}" for h in headers)
    print(hint)
    print(sep)

    # Find best value per metric
    best: Dict[str, float] = {}
    for h in headers:
        vals = [r.get(h, float("nan")) for r in results.values() if not np.isnan(r.get(h, float("nan")))]
        if vals:
            best[h] = min(vals) if h in lower else max(vals)

    # Data rows
    for model_id, metrics in results.items():
        row = f"{model_id:<{col_w}}"
        for h in headers:
            v = metrics.get(h, float("nan"))
            cell = f"{v:.4f}" if not np.isnan(v) else "   n/a"
            # Mark best with *
            if h in best and not np.isnan(v) and abs(v - best[h]) < 1e-9:
                cell = cell + "*"
            else:
                cell = cell + " "
            row += f"{cell:>{num_w}}"
        row += f"  n={metrics.get('n_samples', '?')}"
        print(row)

    print(sep)
    print("* = best in column")
    print()


# ---------------------------------------------------------------------------
# Subset helper
# ---------------------------------------------------------------------------

class _SubsetWrapper:
    """Lightweight subset that preserves BaseDepthDataset interface."""

    def __init__(self, dataset, indices: list) -> None:
        self._ds = dataset
        self._indices = indices
        self.min_depth = dataset.min_depth
        self.max_depth = dataset.max_depth

    def __len__(self) -> int:
        return len(self._indices)

    def _load_sample(self, index: int):
        return self._ds._load_sample(self._indices[index])
