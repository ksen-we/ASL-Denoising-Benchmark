"""Volume inference, Gaussian tile blending, and case-level evaluation."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import torch

from common.data import adjacent_slices, load_case, save_prediction
from common.losses import ReconstructionLoss
from common.metrics import REFERENCE_PROTOCOL, evaluation_scale, reference_metrics
from common.utils import save_preview, write_csv, write_json


def tile_starts(size, tile, overlap):
    if size <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


@torch.inference_mode()
def predict_volume(model, volume, cfg, device):
    """Map a D,H,W NumPy volume to a prediction of the same shape."""
    model.eval()
    amp = cfg.amp and device.type == "cuda"
    def forward(x):
        with torch.autocast(device_type=device.type, enabled=amp):
            result = model(torch.from_numpy(np.ascontiguousarray(x)).to(device))
        if not torch.isfinite(result).all():
            raise FloatingPointError("Prediction contains NaN/Inf; check model state and input")
        return result.float().cpu().numpy()
    if cfg.mode == "2.5d":
        out = np.empty_like(volume)
        for first in range(0, len(volume), cfg.inference_batch_size):
            last = min(first + cfg.inference_batch_size, len(volume))
            inputs = np.stack([adjacent_slices(volume, z) for z in range(first, last)])
            out[first:last] = forward(inputs)[:, 0]
        return out
    if cfg.inference_patch_size is None:
        return forward(volume[None, None])[0, 0]

    tile = tuple(cfg.inference_patch_size)
    padding = tuple((0, max(0, t - n)) for t, n in zip(tile, volume.shape))
    padded = np.pad(volume, padding, mode="edge")
    weight = np.ones(tile, dtype=np.float32)
    for axis, n in enumerate(tile):
        shape = [1, 1, 1]
        shape[axis] = n
        weight *= np.exp(-0.5 * (np.linspace(-1, 1, n) / 0.5) ** 2).reshape(shape)
    weight = np.maximum(weight, 1e-4)
    result, total = np.zeros_like(padded), np.zeros_like(padded)
    starts = [tile_starts(n, t, cfg.inference_overlap) for n, t in zip(padded.shape, tile)]
    for pos in product(*starts):
        region = tuple(slice(s, s + t) for s, t in zip(pos, tile))
        result[region] += forward(padded[region][None, None])[0, 0] * weight
        total[region] += weight
    region = tuple(slice(0, n) for n in volume.shape)
    return (result / total)[region]


def volume_metrics(prediction, item, criterion, evaluation_protocol="legacy"):
    target, valid = item["target"], item["valid"]
    if target is None:
        return {}
    loss, parts = criterion(torch.from_numpy(prediction[None, None]),
                            torch.from_numpy(target[None, None]),
                            torch.from_numpy(valid[None, None].astype(np.float32)))
    delta = prediction[valid].astype(np.float64) - target[valid]
    mse = float(np.mean(delta ** 2))

    raw_target = target[valid].astype(np.float64) * item["scale"] + item["offset"]
    raw_mse = mse * item["scale"] ** 2
    energy = float(np.mean(raw_target ** 2))
    result = {"loss": float(loss), "mae_normalized": float(parts["l1"]),
            "ssim": float(parts["ssim"]),
            "psnr": float(10 * np.log10(criterion.data_range ** 2 / max(mse, 1e-12))),
            "mae_raw": float(np.mean(np.abs(delta)) * item["scale"]),
            "rmse_raw": float(np.sqrt(raw_mse)), "nmse_raw": raw_mse / max(energy, 1e-12)}
    if evaluation_protocol == "legacy":
        return result
    if evaluation_protocol != REFERENCE_PROTOCOL:
        raise ValueError(f"Unknown evaluation protocol: {evaluation_protocol}")

    result["ssim_training_scale"] = float(parts["ssim"])
    result.update(reference_metrics(
        prediction.astype(np.float64) * item["scale"] + item["offset"],
        target.astype(np.float64) * item["scale"] + item["offset"],
        valid, evaluation_scale(item)))
    return result


def summarize(rows):
    summary = {"n_cases": len(rows)}
    if rows:
        for key in rows[0]:
            if key == "case":
                continue
            values = np.asarray([r[key] for r in rows], dtype=np.float64)
            summary[key + "_mean"] = float(values.mean())
            summary[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def evaluate(model, cases, cfg, device, logger=None, export_dir=None, preview_path=None):

    criterion = ReconstructionLoss(cfg).cpu()
    rows, normalization = [], []
    for index, case in enumerate(cases):
        item = load_case(case, cfg.normalization)
        prediction = predict_volume(model, item["input"], cfg, device)
        if item["target"] is not None:
            metrics = volume_metrics(prediction, item, criterion, cfg.evaluation_protocol)
            baseline = volume_metrics(item["input"], item, criterion, cfg.evaluation_protocol)
            rows.append({"case": case.name, **metrics, **{"input_" + k: v for k, v in baseline.items()}})
        normalization.append({"case": case.name, "offset": item["offset"], "scale": item["scale"],
                              **item["normalization"],
                              "source_valid_voxels": int(item["source_valid"].sum()),
                              "evaluation_valid_voxels": int(item["valid"].sum())})
        if item["target"] is not None and cfg.evaluation_protocol == REFERENCE_PROTOCOL:
            normalization[-1]["evaluation"] = evaluation_scale(item)
        if export_dir is not None:
            folder = Path(export_dir) / "nifti" / case.name
            save_prediction(prediction, item, folder / "denoised.nii.gz")

            mask_item = dict(item, scale=1.0, offset=0.0)
            name = "evaluation_mask.nii.gz" if case.mask is not None else "valid_fov_mask.nii.gz"
            save_prediction(item["valid"].astype(np.float32), mask_item, folder / name)
            if index < 3:
                save_preview(item, prediction, Path(export_dir) / "previews" / f"{case.name}.png", cfg.show_plots)
        if preview_path is not None and index == 0:
            save_preview(item, prediction, preview_path, cfg.show_plots)
        if logger and ((index + 1) % 10 == 0 or index + 1 == len(cases)):
            logger.info("Validation/inference case %d/%d", index + 1, len(cases))
    summary = summarize(rows)
    summary["n_predicted_cases"] = len(cases)
    if export_dir is not None:
        write_csv(Path(export_dir) / "metrics.csv", rows)
        write_json(Path(export_dir) / "summary.json", summary)
        write_json(Path(export_dir) / "normalization.json", normalization)
    return summary, rows
