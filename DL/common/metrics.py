"""Optional reference-based evaluation independent of training normalization."""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch

from common.losses import ReconstructionLoss

REFERENCE_PROTOCOL = "reference_minmax_v1"


EVALUATION_SSIM_WINDOW = 7
EVALUATION_SSIM_SIGMA = 1.5


@lru_cache(maxsize=1)
def _reference_criterion():
    return ReconstructionLoss(SimpleNamespace(
        l1_weight=1.0, ssim_weight=1.0, ssim_window=EVALUATION_SSIM_WINDOW,
        ssim_sigma=EVALUATION_SSIM_SIGMA, ssim_data_range=1.0)).cpu()


def evaluation_scale(item):
    """Read the original finite reference range retained by load_case."""
    limits = item.get("reference_range")
    if limits is None:
        raise ValueError("reference_minmax_v1 requires the original reference_range from load_case")
    return dict(limits, method=REFERENCE_PROTOCOL, statistics_scope="reference_all_finite",
                prediction_clipped_for_metrics=True, data_range=1.0)


@torch.inference_mode()
def reference_metrics(prediction_raw, target_raw, valid, limits):
    """Use a shared reference scale; clip temporary predictions for SSIM/PSNR only."""
    if prediction_raw.shape != target_raw.shape or valid.shape != target_raw.shape:
        raise ValueError("Evaluation prediction, reference, and mask must have identical shapes")
    if not valid.any() or not np.isfinite(prediction_raw[valid]).all() or not np.isfinite(target_raw[valid]).all():
        raise ValueError("Evaluation region is empty or prediction/reference contains NaN/Inf")
    low, span = limits["lower"], limits["scale"]
    if not np.isfinite(low) or not np.isfinite(span) or span <= 0:
        raise ValueError("Reference evaluation scale must be finite and positive")

    x = np.where(valid, (prediction_raw.astype(np.float64) - low) / span, 0)
    y = np.where(valid, (target_raw.astype(np.float64) - low) / span, 0)
    clipped_fraction = float(np.mean((x[valid] < 0) | (x[valid] > 1)))
    x = np.clip(x, 0, 1)
    _, parts = _reference_criterion()(
        torch.from_numpy(x.astype(np.float32)[None, None]),
        torch.from_numpy(y.astype(np.float32)[None, None]),
        torch.from_numpy(valid.astype(np.float32)[None, None]))
    mse = float(np.mean((x[valid] - y[valid]) ** 2))
    return {"ssim": float(parts["ssim"]),

            "psnr": float(-10 * np.log10(max(mse, 1e-12))),
            "evaluation_clipped_fraction": clipped_fraction}
