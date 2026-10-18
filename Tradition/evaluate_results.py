"""Evaluate classical predictions with the shared DL metrics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

DL_ROOT = Path(__file__).resolve().parents[1] / "DL"
sys.path.insert(0, str(DL_ROOT))
from common.config import DATA_ROOT, ExperimentConfig
from common.data import discover_cases, load_case, read_volume
from common.evaluation import summarize, volume_metrics
from common.losses import ReconstructionLoss
from common.masking import metric_protocol
from common.metrics import REFERENCE_PROTOCOL, evaluation_scale
from common.utils import file_hash, fingerprint, save_preview, write_csv, write_json

DATASET_ROOT = DATA_ROOT
SPLITS = ("val",)                    # Add "train" to evaluate training cases.
RUN_NAME = "wtm0mask_seed42"  # Evaluation label; predictions are unchanged.
NORMALIZATION = "minmax"
EVALUATION_PROTOCOL = "legacy"  # Matches the final wtm0mask_ori DL evaluation.
MASK_MODE = "m0"                    # Evaluation mask: m0 or none.
MASK_NAME = "m0_foreground_mask.nii"
SOURCE_NAME = "rpwi_ave_4.nii"
TARGET_NAME = "rpwi_ave_max.nii"
METHODS = {
    "BM4D": {"root": DATA_ROOT / "TraditionResult" / "BM4D", "filename": "rpwi_bm4d.nii"},
    "DTCWT_NLM": {"root": DATA_ROOT / "TraditionResult" / "DTCWT_NLM", "filename": "rpwi_dtcwt_nlm.nii"},
}
CPU_THREADS = 4
PREVIEW_CASES = 3


def validate_prediction(path, item):
    """Map raw predictions to the shared source scale without clipping."""
    prediction, image = read_volume(path)
    if prediction.shape != item["input"].shape or not np.allclose(image.affine, item["image"].affine, rtol=0, atol=1e-4):
        raise ValueError("Prediction shape or affine differs from the input; resampling is not allowed.")
    if not np.isfinite(prediction[item["source_valid"]]).all():
        raise ValueError("Prediction contains NaN/Inf in the valid source region.")
    normalized = np.where(item["source_valid"], (prediction - item["offset"]) / item["scale"], 0)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def evaluate_method(method, result_root, prediction_name, dataset_root=DATASET_ROOT,
                    split="val", run_name=RUN_NAME, preview_cases=PREVIEW_CASES,
                    source_name=SOURCE_NAME, target_name=TARGET_NAME,
                    mask_mode=MASK_MODE, mask_name=MASK_NAME, normalization=NORMALIZATION,
                    evaluation_protocol=EVALUATION_PROTOCOL):
    if mask_mode not in {"none", "m0"}:
        raise ValueError("Evaluation mask_mode must be none or m0.")
    torch.set_num_threads(CPU_THREADS)
    cfg = ExperimentConfig(data_root=Path(dataset_root), run_name=run_name,
                           source_name=source_name, target_name=target_name,
                           mask_mode=mask_mode, mask_name=mask_name,
                           normalization=normalization, evaluation_protocol=evaluation_protocol).validate()
    cases, audit = discover_cases(Path(dataset_root) / split, cfg)
    root = Path(result_root)
    folder = root / "evaluation" / split
    if evaluation_protocol == REFERENCE_PROTOCOL:
        folder = folder / ("wtM0mask_reference_minmax_v1" if mask_mode == "m0" else "reference_minmax_v1")
    elif normalization == "p1p99":
        folder = folder / ("wtM0mask_p1p99" if mask_mode == "m0" else "p1p99")
    elif mask_mode == "m0":
        folder = folder / "wtM0mask"  # Keep evaluation protocols in separate directories.
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / ".evaluation.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Evaluation lock exists: {lock}") from exc
    try:
        # A failed/incomplete rerun must never retain a previous success marker.
        for name in ("completed.json", "metrics.csv", "summary.json", "metrics.partial.csv", "summary.partial.json"):
            (folder / name).unlink(missing_ok=True)
        criterion = ReconstructionLoss(cfg).cpu()
        rows, baseline_rows, statuses, predictions = [], [], [], []
        normalization_records = []
        run_config_path = root / "run_config.json"
        run_record = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else None
        for index, case in enumerate(cases):
            item = load_case(case, cfg.normalization)
            normalization_records.append({"case": case.name, **item["normalization"],
                                          "offset": item["offset"], "scale": item["scale"],
                                          "source_valid_voxels": int(item["source_valid"].sum()),
                                          "evaluation_valid_voxels": int(item["valid"].sum())})
            if evaluation_protocol == REFERENCE_PROTOCOL:
                normalization_records[-1]["evaluation"] = evaluation_scale(item)
            baseline = volume_metrics(item["input"], item, criterion, evaluation_protocol)
            baseline_rows.append({"case": case.name, **baseline})
            path = root / split / case.name / prediction_name
            status = {"case": case.name, "prediction": str(path), "status": "missing", "message": ""}
            if path.is_file():
                try:
                    output_hash = file_hash(path)
                    # BM4D must have been generated by the repaired, recorded
                    # implementation; old untracked outputs are not scored as it.
                    if method == "BM4D":
                        sidecar = Path(str(path) + ".json")
                        record = json.loads(sidecar.read_text(encoding="utf-8"))
                        if (run_record is None or record.get("signature") != run_record["signature"]
                                or record.get("input_sha256") != file_hash(case.source)
                                or record.get("output_sha256") != output_hash):
                            raise ValueError("BM4D algorithm/input/output fingerprints are missing or inconsistent.")
                        denoising_mask = run_record.get("config", {}).get("denoising_mask")
                        if denoising_mask is not None:
                            mask_path = case.source.parent / denoising_mask["name"]
                            if record.get("mask_sha256") != file_hash(mask_path):
                                raise ValueError("The BM4D denoising mask has changed; regenerate the prediction.")
                    prediction = validate_prediction(path, item)
                    values = volume_metrics(prediction, item, criterion, evaluation_protocol)
                    if not all(np.isfinite(v) for v in values.values()):
                        raise ValueError("Evaluation produced nonfinite metrics.")
                    rows.append({"case": case.name, **values,
                                 **{"input_" + key: value for key, value in baseline.items()},
                                 "delta_ssim": values["ssim"] - baseline["ssim"],
                                 "delta_psnr": values["psnr"] - baseline["psnr"],
                                 "mae_reduction_raw": baseline["mae_raw"] - values["mae_raw"]})
                    predictions.append({"case": case.name, "path": str(path.resolve()), "sha256": output_hash})
                    status["status"] = "evaluated"
                    if len(rows) <= preview_cases:
                        save_preview(item, prediction, folder / "previews" / f"{case.name}.png")
                except Exception as exc:
                    status.update(status="invalid", message=f"{type(exc).__name__}: {exc}")
            statuses.append(status)
            if (index + 1) % 10 == 0 or index + 1 == len(cases):
                print(f"{method}/{split}: checked {index + 1}/{len(cases)}, valid {len(rows)}", flush=True)
        write_csv(folder / "case_status.csv", statuses)
        write_json(folder / "normalization.json", normalization_records)
        write_csv(folder / "input_metrics.csv", baseline_rows)
        write_json(folder / "input_summary.json", summarize(baseline_rows))
        complete = len(rows) == len(cases)
        summary = summarize(rows)
        summary.update(expected_cases=len(cases), complete=complete,
                       missing_cases=sum(s["status"] == "missing" for s in statuses),
                       invalid_cases=sum(s["status"] == "invalid" for s in statuses))
        suffix = "" if complete else ".partial"
        write_csv(folder / f"metrics{suffix}.csv", rows)
        write_json(folder / f"summary{suffix}.json", summary)
        digest, manifest = fingerprint(cases)
        metric_config = cfg.to_dict()
        metric_config.update(method=method, mode="3d", use_gan=False, seed=None, inference_patch_size=None)
        write_json(folder / "provenance.json", {
            "family": "traditional", "config": metric_config,
            "checkpoint_epoch": None, "metric_protocol": metric_protocol(cfg),
            "denoising_root": str(root.resolve()),
            "data_fingerprint": digest, "manifest": manifest, "audit": audit,
            "prediction_files": predictions, "denoising_run": run_record,
            "denoising_provenance": "verified" if method == "BM4D" else "legacy output; parameters not verified",
        })
        if complete:
            write_json(folder / "completed.json", {"n_cases": len(rows), "data_fingerprint": digest})
            print(f"{method}/{split}: SSIM={summary['ssim_mean']:.5f}, PSNR={summary['psnr_mean']:.3f} dB, "
                  f"MAE={summary['mae_raw_mean']:.5g}; output: {folder}", flush=True)
        else:
            print(f"{method}/{split}: incomplete ({len(rows)}/{len(cases)}); partial results are excluded from comparisons.", flush=True)
        return summary
    finally:
        lock.unlink(missing_ok=True)


def main():
    for method, settings in METHODS.items():
        if (Path(settings["root"]) / ".denoising.lock").exists():
            print(f"{method} is being denoised; skipping evaluation.", flush=True)
            continue
        for split in SPLITS:
            evaluate_method(method, settings["root"], settings["filename"], DATASET_ROOT, split, RUN_NAME,
                            normalization=NORMALIZATION, evaluation_protocol=EVALUATION_PROTOCOL)


if __name__ == "__main__":
    main()
