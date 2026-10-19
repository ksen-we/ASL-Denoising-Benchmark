"""Generate qualitative PWI comparisons from saved, completed predictions.

Example:
    python qualitative_comparison.py --dataset-root /path/to/Dataset001 \
        --output-dir /path/to/figures

The selection manifest contains private case identifiers. Keep it with the
local figures, not in a public code repository.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asl_figure_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker


METHODS = ("UNet", "Pix2Pix", "SwinUNet", "SwinIR")
LABELS = ("Low-SNR\ninput", "BM4D\n(3D)", "DT-CWT-NLM\n(3D)",
          "U-Net", "Pix2Pix", "Swin U-Net", "SwinIR", "Reference")
MODES = (("2p5D", "2.5d", "2.5D"), ("3D", "3d", "3D"))
INPUT_METRICS = ("input_ssim", "input_psnr", "input_mae_raw", "input_rmse_raw", "input_nmse_raw")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_metrics(folder):
    if not (folder / "completed.json").is_file():
        raise ValueError(f"Incomplete inference: {folder}")
    if any((folder / name).exists() for name in (".inference.lock", ".evaluation.lock")):
        raise ValueError(f"Inference or evaluation is still running: {folder}")
    with (folder / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {row["case"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate cases: {folder}")
    return indexed


def check_runs(root, split, prefix, seed, expected_cases):
    folders, baseline, identities, baseline_metrics = {}, None, [], None
    fingerprint = None

    def protocol(provenance):
        cfg = provenance["config"]
        return (provenance["data_fingerprint"], provenance["metric_protocol"],
                *(cfg[key] for key in ("source_name", "target_name", "mask_name",
                                      "ssim_window", "ssim_sigma", "ssim_data_range")))

    def check_baseline(records):
        current = {c: [float(r[key]) for key in INPUT_METRICS] for c, r in records.items()}
        if len(current) != expected_cases:
            raise ValueError(f"Expected {expected_cases} cases, found {len(current)}.")
        if not np.isfinite(list(current.values())).all():
            raise ValueError("Nonfinite input metrics.")
        if baseline_metrics is not None and (set(current) != set(baseline_metrics) or any(
                not np.allclose(current[c], baseline_metrics[c], rtol=1e-6, atol=1e-7)
                for c in baseline_metrics)):
            raise ValueError("Case sets or input baselines differ across methods.")
        return current

    for method in METHODS:
        for disk_mode, config_mode, _ in MODES:
            for gres in (0, 1):
                run = f"{prefix}_gres{gres}_seed{seed}"
                folder = root / "DLResult" / method / disk_mode / run / "inference" / split
                records = read_metrics(folder)
                provenance = json.loads((folder / "provenance.json").read_text())
                cfg = provenance["config"]
                expected = (method, config_mode, run, bool(gres), "minmax", "legacy")
                actual = tuple(cfg[key] for key in ("method", "mode", "run_name", "global_residual",
                                                    "normalization", "evaluation_protocol"))
                if actual != expected:
                    raise ValueError(f"Unexpected final-run configuration: {folder}")
                identity = protocol(provenance)
                if fingerprint is None:
                    fingerprint = identity
                elif identity != fingerprint:
                    raise ValueError("Final runs use different datasets or evaluation protocols.")
                current_metrics = check_baseline(records)
                current = {case: values[0] for case, values in current_metrics.items()}
                if baseline is None:
                    baseline = current
                    baseline_metrics = current_metrics
                folders[(method, disk_mode, gres)] = folder
                identities.append({"folder": str(folder.relative_to(root)),
                                   "metrics_sha256": sha256(folder / "metrics.csv"),
                                   "provenance_sha256": sha256(folder / "provenance.json")})
    for method in ("BM4D", "DTCWT_NLM"):
        folder = root / "TraditionResult" / method / "evaluation" / split / "wtM0mask"
        records = read_metrics(folder)
        provenance = json.loads((folder / "provenance.json").read_text())
        cfg = provenance["config"]
        if (cfg["method"], cfg["mode"], cfg["mask_mode"]) != (method, "3d", "m0"):
            raise ValueError(f"Unexpected classical evaluation configuration: {folder}")
        if protocol(provenance) != fingerprint:
            raise ValueError("Classical and DL evaluation protocols differ.")
        check_baseline(records)
        identities.append({"folder": str(folder.relative_to(root)),
                           "metrics_sha256": sha256(folder / "metrics.csv"),
                           "provenance_sha256": sha256(folder / "provenance.json")})
    return folders, baseline, identities


def select_cases(baseline):
    candidates = sorted(baseline)
    values = np.array([baseline[c] for c in candidates])
    selected = []
    for label, quantile in zip(("Lower SSIM", "Higher SSIM"), (0.25, 0.75)):
        threshold = float(np.quantile(values, quantile))
        case = min(candidates, key=lambda c: (abs(baseline[c] - threshold), c))
        selected.append({"label": label, "case": case, "input_ssim_quantile": quantile,
                         "quantile_value": threshold, "input_ssim": baseline[case]})
    if len({item["case"] for item in selected}) != 2:
        raise ValueError("The input quartiles do not select two distinct cases.")
    return selected


def load_case(root, split, selected, folders):
    case = selected["case"]
    input_path = root / split / case / "rpwi_ave_4.nii"
    source_image = nib.load(input_path)
    paths = []

    def load(path):
        img = nib.load(path)
        if img.shape != source_image.shape or not np.allclose(
                img.affine, source_image.affine, rtol=1e-5, atol=1e-5):
            raise ValueError(f"NIfTI geometry mismatch: {path}")
        paths.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
        return nib.as_closest_canonical(img).get_fdata(dtype=np.float32)

    source = load(input_path)
    reference = load(root / split / case / "rpwi_ave_max.nii")
    mask = load(root / split / case / "m0_foreground_mask.nii") > 0
    mask &= np.isfinite(source) & np.isfinite(reference)
    if not mask.any():
        raise ValueError("Empty evaluation mask.")
    predictions = {}
    for key, folder in folders.items():
        saved_mask = load(folder / "nifti" / case / "evaluation_mask.nii.gz") > 0
        if not np.array_equal(mask, saved_mask):
            raise ValueError(f"Evaluation masks differ: {folder}")
        prediction = load(folder / "nifti" / case / "denoised.nii.gz")
        if not np.isfinite(prediction[mask]).all():
            raise ValueError(f"Nonfinite prediction within mask: {folder}")
        predictions[key] = prediction
    classical = []
    for method, filename in (("BM4D", "rpwi_bm4d.nii"), ("DTCWT_NLM", "rpwi_dtcwt_nlm.nii")):
        volume = load(root / "TraditionResult" / method / split / case / filename)
        if not np.isfinite(volume[mask]).all():
            raise ValueError(f"Nonfinite classical output: {method}")
        classical.append(volume)

    slice_mass = np.sum(mask, axis=(0, 1))
    z = int(np.searchsorted(np.cumsum(slice_mass), np.sum(slice_mass) / 2.0))
    pwi_limits = np.percentile(reference[mask], [1, 99]).astype(float)
    error_max = float(np.percentile(np.abs(source[mask] - reference[mask]), 95))
    if pwi_limits[1] <= pwi_limits[0] or error_max <= 0:
        raise ValueError("Degenerate display window.")
    selected.update({"slice_index_canonical_zero_based": z,
                     "volume_shape_canonical": list(source.shape),
                     "pwi_limits_raw": pwi_limits.tolist(), "error_max_raw": error_max,
                     "input_slice_mae_raw": float(np.mean(np.abs(source[:, :, z] - reference[:, :, z])[mask[:, :, z]])),
                     "files": paths})
    return {"selection": selected, "input": source, "reference": reference,
            "mask": mask, "classical": classical, "predictions": predictions,
            "z": z, "pwi_limits": pwi_limits, "error_max": error_max}


def volumes(case, mode, gres):
    return [case["input"], *case["classical"],
            *(case["predictions"][(m, mode, gres)] for m in METHODS), case["reference"]]


def draw_figure(groups, title, output_stem, dpi):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    nrows = len(groups) * 2
    fig = plt.figure(figsize=(22.0, 1.5 + nrows * 2.5), facecolor="white")
    grid = fig.add_gridspec(nrows, 9, width_ratios=[1] * 8 + [0.055],
                           left=0.08, right=0.96, top=0.90, bottom=0.035,
                           wspace=0.035, hspace=0.075)
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("black")
    gray_cmap = plt.get_cmap("gray").copy()
    gray_cmap.set_bad("black")
    for group_index, (case, mode, gres, group_label) in enumerate(groups):
        images = volumes(case, mode, gres)
        z = case["z"]
        plane_mask = np.rot90(case["mask"][:, :, z])
        for error in (False, True):
            row = 2 * group_index + int(error)
            cmap = error_cmap if error else gray_cmap
            norm = colors.Normalize(0, case["error_max"]) if error else colors.Normalize(*case["pwi_limits"])
            for col, volume in enumerate(images):
                ax = fig.add_subplot(grid[row, col])
                if error:
                    plane = np.rot90(np.abs(volume[:, :, z] - case["reference"][:, :, z]))
                    plane = np.ma.masked_where(~plane_mask, plane)
                else:
                    plane = np.ma.masked_invalid(np.rot90(volume[:, :, z]))
                artist = ax.imshow(plane, cmap=cmap, norm=norm, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(LABELS[col], fontsize=13, fontweight="bold", pad=9)
                if col == 0:
                    label = "Absolute\nerror" if error else group_label + "\nPWI"
                    ax.set_ylabel(label, fontsize=13, labelpad=11, rotation=0,
                                  va="center", ha="right", fontweight="bold")
            cax = fig.add_subplot(grid[row, 8])
            cbar = fig.colorbar(artist, cax=cax)
            cbar.outline.set_linewidth(0.5)
            cbar.ax.tick_params(labelsize=11, width=0.5, length=3)
            cbar.locator = ticker.MaxNLocator(nbins=3)
            cbar.update_ticks()
            cbar.ax.set_title("a.u.", fontsize=10, pad=7)
    fig.text(0.08, 0.972, title, fontsize=19, fontweight="bold", ha="left", va="top")
    for extension in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix("." + extension), dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="wtm0mask_ori")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-cases", type=int, default=54)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folders, baseline, identities = check_runs(root, args.split, args.run_prefix, args.seed, args.expected_cases)
    selection = select_cases(baseline)
    cases = [load_case(root, args.split, selected, folders) for selected in selection]
    captions = {}
    common = ("Neural networks use 3D residual prediction. Within each case, all methods share "
              "the reference-foreground P1–P99 intensity window and the input-error P95 upper limit. "
              "Error maps use the common foreground mask; color bars indicate original PWI intensity units.")
    stem = "qualitative_low_high"
    draw_figure([(case, "3D", 1, case["selection"]["label"]) for case in cases],
                "PWI denoising with 3D residual prediction",
                args.output_dir / stem, args.dpi)
    captions[stem] = ("PWI and absolute error for cases nearest the 25th and 75th percentiles of input SSIM "
                      f"({cases[0]['selection']['input_ssim']:.3f} and "
                      f"{cases[1]['selection']['input_ssim']:.3f}). " + common)
    (args.output_dir / "qualitative_captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")
    manifest = {"private": True, "run_prefix": args.run_prefix, "seed": args.seed,
                "split": args.split, "number_of_cases": len(baseline), "expected_cases": args.expected_cases,
                "selected_cases": selection,
                "selection_rule": "Nearest input SSIM to numpy linear quantiles 0.25 and 0.75; ties by case identifier.",
                "slice_rule": "Median cumulative foreground voxel count along canonical slice axis.",
                "runs": identities, "generator_sha256": sha256(__file__)}
    (args.output_dir / "qualitative_selection.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified 16 DL and 2 classical evaluations with {len(baseline)} matched input baselines; "
          "wrote 1 figure and a private selection manifest for 2 cases.")


if __name__ == "__main__":
    main()
