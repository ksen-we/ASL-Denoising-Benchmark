"""Batch-denoise Dataset001/rpwi_ave_4.nii with CPU or CUDA BM4D.

For every case, sigma is estimated from the input alone with the robust MAD
of one-level 3-D Haar HHH coefficients. The high-SNR target is never used.

Edit the configuration section below, then run this file directly in the
IDE/editor. No command-line arguments are required.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import time
from importlib import metadata
from pathlib import Path

import nibabel as nib
import numpy as np
from run_support import ensure_config, file_hash, verified_existing, write_json, write_rows


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parents[1] / "Data" / "Dataset001"

# -------------------------------------------------------------------------
# User-adjustable configuration
# -------------------------------------------------------------------------
SPLITS = ["train", "val"]
INPUT_NAME = "rpwi_ave_4.nii"
TARGET_NAME = "rpwi_ave_max.nii"  # Evaluation only; excluded from noise estimation and denoising.
OUTPUT_NAME = "rpwi_bm4d.nii"
OUTPUT_ROOT = DATASET_ROOT / "TraditionResult" / "BM4D"  # May also be an absolute custom Path
REPORT_NAME = "bm4d_sigma_report.csv"
CASE_FILTER = ""              # Example: "case_001"; empty means all cases
MAX_CASES = None              # Example: 3 for a short test; None means all
OVERWRITE = False
DRY_RUN = False               # True: estimate sigma only; do not run BM4D
VERBOSE = False               # True: print CUDA kernel details
BACKEND = "cuda"               # No automatic backend fallback.
CPU_THREADS = 4               # Use 1 thread for stricter CPU reproducibility.
GPU_ID = 0                    # Index among GPUs visible to this process.
SIGMA_MULTIPLIER = 1.0        # Global correction of automatic sigma estimates
RUN_NAME = "baseline_seed42"  # Denoising record label.
EVALUATE_AFTER_RUN = True     # Evaluate the complete validation split after denoising.
EVALUATION_SPLITS = ("val",)
EVALUATION_RUN_NAME = "wtm0mask_seed42"  # Shared evaluation label.
MASK_MODE = "none"            # none / output / input_output; denoise the full FOV by default.
MASK_NAME = "m0_foreground_mask.nii"
CHECK_ONLY = False           # True: run backend checks without processing cases.
# CUDA parameters; the optional CPU backend uses the official np profile.
BM4D_PARAMETERS = dict(bs_ht=4, step_ht=3, max_stack_ht=16, search_ht=7,
                       tau_ht=2.9527, lambda_thr=3.0, bs_wiener=5,
                       step_wiener=3, max_stack_wiener=32, search_wiener=7,
                       tau_wiener=0.7689)

MAD_NORMAL_QUANTILE = 0.6744897501960817
HAAR_NORMALIZATION = np.sqrt(2.0)


def load_3d_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}: {path}")
    if not np.any(np.isfinite(volume)):
        raise ValueError(f"Volume contains no finite voxels: {path}")
    return image, np.ascontiguousarray(volume)


def robust_scale(volume: np.ndarray, percentile: float = 99.5) -> float:
    """Return a robust positive scale while preserving signed ASL values."""
    values = np.abs(volume[np.isfinite(volume)])
    values = values[values > np.finfo(np.float32).eps]
    if values.size == 0:
        return 1.0  # Unit scale for an all-zero input.
    scale = float(np.percentile(values, percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid normalization scale: {scale}")
    return scale


def estimate_sigma_haar(volume: np.ndarray) -> tuple[float, int]:
    """Estimate AWGN sigma using robust 3-D Haar HHH coefficients.

    The normalized Haar transform preserves white-noise variance. Therefore
    sigma = MAD(coefficients) / 0.67448975. Coefficients touching NaN/Inf or
    zero-padded outside-FOV voxels are excluded.
    """
    finite = np.isfinite(volume)
    if not finite.any():
        raise ValueError("Cannot estimate sigma without finite voxels")
    values = volume[finite]
    if np.all(values == values[0]):
        return 0.0, 0  # An exactly constant input has zero estimated noise.
    support = finite & (np.abs(volume) > np.finfo(np.float32).eps)
    detail = np.where(finite, volume, 0.0).astype(np.float64, copy=False)

    for axis in range(3):
        even_length = (detail.shape[axis] // 2) * 2
        if even_length < 2:
            raise ValueError(
                f"Axis {axis} is too short for Haar estimation: {volume.shape}"
            )
        even = [slice(None)] * 3
        odd = [slice(None)] * 3
        even[axis] = slice(0, even_length, 2)
        odd[axis] = slice(1, even_length, 2)
        even_index = tuple(even)
        odd_index = tuple(odd)
        detail = (detail[even_index] - detail[odd_index]) / HAAR_NORMALIZATION
        support = support[even_index] & support[odd_index]

    coefficients = detail[support]
    if coefficients.size < 100:
        raise ValueError(
            "Too few valid Haar coefficients for a stable sigma estimate: "
            f"{coefficients.size}"
        )
    center = float(np.median(coefficients))
    mad = float(np.median(np.abs(coefficients - center)))
    sigma = mad / MAD_NORMAL_QUANTILE
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Invalid estimated sigma: {sigma}")
    return sigma, int(coefficients.size)


def save_like_source(
    output_path: Path,
    volume: np.ndarray,
    source_image: nib.Nifti1Image,
) -> None:
    shape = source_image.shape[:3]
    if volume.shape != shape or not np.isfinite(volume).all():
        raise ValueError("Refusing to save a wrong-shape or non-finite BM4D prediction")
    if len(source_image.shape) == 4:
        volume = volume[..., None]
    header = source_image.header.copy()
    header.set_data_dtype(np.float32)
    output_image = nib.Nifti1Image(
        volume.astype(np.float32, copy=False),
        source_image.affine,
        header=header,
    )
    output_image.set_qform(source_image.get_qform(), int(source_image.header["qform_code"]))
    output_image.set_sform(source_image.get_sform(), int(source_image.header["sform_code"]))
    output_image.header.set_slope_inter(1.0, 0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(".tmp_" + output_path.name)
    nib.save(output_image, str(temporary))
    temporary.replace(output_path)


def discover_inputs(
    dataset_root: Path,
    splits: list[str],
    input_name: str,
    case_name: str | None,
) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    for split in splits:
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {split_root}")
        for input_path in sorted(split_root.glob(f"*/{input_name}")):
            if case_name is None or input_path.parent.name == case_name:
                inputs.append((split, input_path))
    if not inputs:
        suffix = f" for case {case_name!r}" if case_name else ""
        raise FileNotFoundError(
            f"No {input_name} files found under {dataset_root}{suffix}."
        )
    return inputs


def main() -> None:
    from safety import validate_parameters
    from cpu_runtime import validate_cpu_input
    dataset_root = DATASET_ROOT.expanduser().resolve()
    output_root = OUTPUT_ROOT.expanduser().resolve()
    if MASK_MODE not in {"none", "output", "input_output"}:
        raise ValueError("MASK_MODE must be none, output or input_output.")
    if MASK_MODE != "none":
        output_root = output_root.with_name(output_root.name + "_mask_" + MASK_MODE)
        sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "DL"))
        from common.masking import load_foreground_mask
    if MAX_CASES is not None and MAX_CASES < 1:
        raise ValueError("MAX_CASES must be at least 1 or None")
    if not np.isfinite(SIGMA_MULTIPLIER) or SIGMA_MULTIPLIER <= 0:
        raise ValueError("SIGMA_MULTIPLIER must be finite and positive")
    if BACKEND not in ("cpu", "cuda"):
        raise ValueError('BACKEND must be "cpu" or "cuda".')
    if BACKEND == "cuda":
        validate_parameters((8, 8, 8), 0.1, **BM4D_PARAMETERS)
    else:
        validate_cpu_input((8, 8, 8), 0.1, CPU_THREADS)
    inputs = discover_inputs(dataset_root, SPLITS, INPUT_NAME, CASE_FILTER or None)
    if MAX_CASES is not None:
        inputs = inputs[:MAX_CASES]
    output_root.mkdir(parents=True, exist_ok=True)
    lock = output_root / ".denoising.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Denoising lock exists: {lock}. Remove it only after the previous task ends.") from exc
    rows = []
    try:
        bm4d, environment = None, {"status": "dry_run"}
        if not DRY_RUN:
            try:
                if BACKEND == "cuda":
                    from cuda_runtime import preflight
                    bm4d, environment = preflight(GPU_ID)
                else:
                    from cpu_runtime import preflight
                    bm4d, environment = preflight(CPU_THREADS)
            except Exception as exc:
                write_json(output_root / "preflight.json", {"status": "failed", "error": str(exc),
                                                           "python": sys.executable})
                raise
        environment["python"] = sys.executable
        write_json(output_root / "preflight.json", environment)
        if CHECK_ONLY:
            print(f"Preflight completed: {environment}", flush=True)
            return
        if EVALUATE_AFTER_RUN and not DRY_RUN:
            sys.path.insert(0, str(SCRIPT_DIR.parent))
            from evaluate_results import evaluate_method
        # Selection/GPU/logging changes do not change the denoising signature.
        # Numeric preprocessing and CUDA implementation changes do.
        implementation_files = (("bm4d_cuda.py", "safety.py", "cuda_runtime.py", "run_support.py")
                                if BACKEND == "cuda" else ("cpu_runtime.py", "run_support.py"))
        config = {"method": "BM4D", "backend": "local_cuda" if BACKEND == "cuda" else "official_cpu", "run_name": RUN_NAME,
                  "input_name": INPUT_NAME, "output_name": OUTPUT_NAME,
                  "sigma_multiplier": SIGMA_MULTIPLIER,
                  "parameters": BM4D_PARAMETERS if BACKEND == "cuda" else {"profile": "np", "num_threads": CPU_THREADS},
                  "cpu_profile_settings": environment.get("profile_settings"),
                  "normalization": "source_abs_p99.5_no_clipping", "sigma": "source_Haar_HHH_MAD",
                  "code": {name: file_hash(SCRIPT_DIR / name) for name in
                           implementation_files},
                  "preprocessing_sha256": hashlib.sha256("".join(inspect.getsource(f) for f in
                           (load_3d_nifti, robust_scale, estimate_sigma_haar, save_like_source)).encode()).hexdigest(),
                  "packages": ({name: metadata.version(name) for name in ("numpy", "scipy", "PyWavelets")}
                               if not DRY_RUN else {}),
                  "backend_packages": environment.get("packages")}
        if MASK_MODE != "none":
            config["denoising_mask"] = dict(mode=MASK_MODE, name=MASK_NAME,
                loader_sha256=file_hash(SCRIPT_DIR.parents[1] / "DL" / "common" / "masking.py"))
        signature = None if DRY_RUN else ensure_config(output_root, config, OVERWRITE)
        if not DRY_RUN:
            for split in {s for s, _ in inputs}:
                (output_root / "evaluation" / split / "completed.json").unlink(missing_ok=True)
                (output_root / "evaluation" / split / "m0" / "completed.json").unlink(missing_ok=True)
                (output_root / "evaluation" / split / "wtM0mask" / "completed.json").unlink(missing_ok=True)
        report = output_root / ("bm4d_sigma_estimates.csv" if DRY_RUN else REPORT_NAME)
        device = environment.get("gpu_name", environment.get("device", "sigma estimation only"))
        print(f"BM4D | backend={BACKEND} | {len(inputs)} cases | {device} | {output_root}", flush=True)
        for index, (split, input_path) in enumerate(inputs, 1):
            output = output_root / split / input_path.parent.name / OUTPUT_NAME
            row = dict(split=split, case=input_path.parent.name, input=str(input_path), output=str(output),
                       shape="", scale_p99_5=None, sigma_raw=None, variance_raw=None,
                       sigma_normalized=None, haar_coefficients=None, seconds=None, status="", message="")
            tick = time.perf_counter()
            try:
                image, volume = load_3d_nifti(input_path)
                foreground, mask_hash = None, None
                if MASK_MODE != "none":
                    foreground = load_foreground_mask(input_path.parent / MASK_NAME, image)
                    mask_hash = file_hash(input_path.parent / MASK_NAME)
                if BACKEND == "cuda":
                    validate_parameters(volume.shape, 0.1, **BM4D_PARAMETERS)
                else:
                    validate_cpu_input(volume.shape, 0.1, CPU_THREADS)
                scale = robust_scale(volume)
                sigma, count = estimate_sigma_haar(volume)
                sigma *= SIGMA_MULTIPLIER
                row.update(shape="x".join(map(str, volume.shape)), scale_p99_5=scale,
                           sigma_raw=sigma, variance_raw=sigma**2, sigma_normalized=sigma / scale,
                           haar_coefficients=count)
                if DRY_RUN:
                    row["status"] = "estimated_only"
                elif output.exists() and not OVERWRITE:
                    saved = verified_existing(output, input_path, signature)
                    if MASK_MODE != "none" and saved.get("mask_sha256") != mask_hash:
                        raise ValueError("The mask for an existing output has changed; refusing reuse.")
                    row.update(status="skipped_verified", seconds=saved["seconds"])
                else:
                    finite = np.isfinite(volume)
                    normalized = np.ascontiguousarray(np.where(finite, volume / scale, 0), dtype=np.float32)
                    if MASK_MODE == "input_output":
                        normalized[~foreground] = 0
                    kwargs = BM4D_PARAMETERS if BACKEND == "cuda" else {}
                    denoised = np.asarray(bm4d(normalized, sigma / scale, verbose=VERBOSE, **kwargs), dtype=np.float32)
                    if denoised.shape != volume.shape or not np.isfinite(denoised).all():
                        raise FloatingPointError("BM4D returned wrong shape or NaN/Inf")
                    denoised = denoised * scale
                    if not np.isfinite(denoised).all():
                        raise FloatingPointError("BM4D output overflowed after restoring raw signal units")
                    denoised[~finite] = 0
                    if foreground is not None:
                        denoised[~foreground] = 0
                    save_like_source(output, denoised, image)
                    row.update(status="processed", seconds=time.perf_counter() - tick)
                    write_json(Path(str(output) + ".json"), {"signature": signature,
                               "input_sha256": file_hash(input_path), "output_sha256": file_hash(output),
                               "sigma_raw": sigma, "scale": scale, "seconds": row["seconds"],
                               "mask_sha256": mask_hash})
            except Exception as exc:
                row.update(status="error", message=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - tick)
            rows.append(row)
            write_rows(report, rows)
            print(f"[{index}/{len(inputs)}] {split}/{row['case']}: {row['status']} | sigma={row['sigma_raw']}"
                  + (f" | {row['message']}" if row['message'] else ""), flush=True)
    finally:
        lock.unlink(missing_ok=True)
    if EVALUATE_AFTER_RUN and not DRY_RUN:
        for split in EVALUATION_SPLITS:
            evaluate_method("BM4D", output_root, OUTPUT_NAME, dataset_root, split, EVALUATION_RUN_NAME,
                            source_name=INPUT_NAME, target_name=TARGET_NAME,
                            normalization="minmax", evaluation_protocol="legacy")
    counts = {status: sum(r["status"] == status for r in rows) for status in {r["status"] for r in rows}}
    print(f"Finished: {counts}; report: {report}", flush=True)
    if counts.get("error", 0):
        raise SystemExit(1)
    return rows


if __name__ == "__main__":
    main()
