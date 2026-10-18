"""Plot completed case-level comparisons. Run DL/compare_results.py first."""
from pathlib import Path
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ftasl_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ftasl_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib


ROOT = Path(__file__).resolve().parents[1] / "Data" / "Dataset001" / "DLResult"
RUN_NAME = "wtm0mask_ori_gres0_seed42"  
SPLIT = "val"
SUMMARY_CSV = ROOT / "_comparison" / RUN_NAME / SPLIT / "all_results.csv"
OUTPUT_DIR = SUMMARY_CSV.parent / "figures"

METRICS = ("ssim", "psnr", "mae_raw")

METHOD_ORDER = ("BM4D", "DTCWT_NLM", "UNet", "Pix2Pix", "SwinUNet", "SwinIR")
INCLUDE_INPUT = True             
SHOW_CASE_POINTS = True          
SHOW_FIGURES = False             
SAVE_FORMATS = ("png", "pdf")   
DPI = 300
FIGURE_WIDTH = 10.5              
HEIGHT_PER_METRIC = 3.0
FONT_SIZE = 11
JITTER_SEED = 42                 

LABELS = {
    "Input": "Low-SNR\ninput", "BM4D": "BM4D", "DTCWT_NLM": "DT-CWT\nNLM",
    "UNet": "U-Net", "Pix2Pix": "Pix2Pix", "SwinUNet": "Swin-Unet",
    "SwinIR": "SwinIR",
}
COLORS = {
    "Input": "#929AA3", "BM4D": "#B69B7D", "DTCWT_NLM": "#CC79A7",
    "UNet": "#56B4E9", "Pix2Pix": "#E69F00", "SwinUNet": "#009E73",
    "SwinIR": "#0072B2",
}
METRIC_INFO = {
    "ssim": "SSIM",
    "psnr": "PSNR (dB)",
    "mae_raw": "MAE",   
    "rmse_raw": "RMSE", 
    "nmse_raw": "NMSE",
    "loss": "L1 + (1 - SSIM)",
    "mae_normalized": "Normalized MAE",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_group(rows):
    """Align cases and verify the evaluation protocol and input baseline."""
    methods = [row["method"] for row in rows]
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate methods in a group; select one run per method and seed.")
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    rows = sorted(rows, key=lambda row: (order.get(row["method"], len(order)), row["method"]))
    cases, protocol, baseline = None, None, None
    distributions = {}
    for row in rows:
        folder = Path(row["result_dir"])
        if not folder.is_absolute():
            folder = ROOT / folder
        if not (folder / "completed.json").is_file():
            raise ValueError(f"Evaluation is incomplete: {folder}")
        if any((folder / name).exists() for name in (".inference.lock", ".evaluation.lock")):
            raise RuntimeError(f"Results are being updated; wait before plotting: {folder}")
        info = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        cfg = info["config"]
        current_protocol = (info["data_fingerprint"], info["metric_protocol"], cfg["mode"],
                            *(cfg[key] for key in ("source_name", "target_name", "l1_weight",
                              "ssim_weight", "ssim_window", "ssim_sigma", "ssim_data_range")))
        if (cfg["method"], cfg["mode"], cfg["run_name"]) != (row["method"], row["mode"], row["run_name"]):
            raise ValueError(f"Summary and provenance differ; rerun comparison: {folder}")
        if cfg["method"] == "Pix2Pix" and not cfg["use_gan"]:
            raise ValueError("Pix2Pix comparisons require a complete GAN.")
        if protocol is None:
            protocol = current_protocol
        elif protocol != current_protocol:
            raise ValueError("A plot group contains incompatible evaluation protocols.")
        records = read_csv(folder / "metrics.csv")
        indexed = {record["case"]: record for record in records}
        ids = sorted(indexed)
        if len(ids) != len(records) or len(ids) != int(row["n_cases"]):
            raise ValueError(f"Case count mismatch or duplicate cases: {folder}")
        if cases is None:
            cases = ids
        elif cases != ids:
            raise ValueError("Methods have different evaluation case sets.")
        values, inputs = {}, {}
        for metric in METRICS:
            values[metric] = np.array([float(indexed[c][metric]) for c in cases])
            inputs[metric] = np.array([float(indexed[c]["input_" + metric]) for c in cases])
            if not np.isfinite(values[metric]).all() or not np.isfinite(inputs[metric]).all():
                raise ValueError(f"Nonfinite metric values: {folder} / {metric}")
            if baseline is not None and not np.allclose(inputs[metric], baseline[metric], rtol=1e-6, atol=1e-7):
                raise ValueError(f"Input baselines differ: {metric}")
        if baseline is None:
            baseline = inputs
        distributions[row["method"]] = values
    if INCLUDE_INPUT:
        distributions = {"Input": baseline, **distributions}
    return distributions, len(cases), rows[0]["evaluation_mask"]


def draw_group(plt, distributions, n_cases, mode, mask, output_stem):
    methods = list(distributions)
    positions = np.arange(1, len(methods) + 1)
    width = max(FIGURE_WIDTH, 1.35 * len(methods))
    fig, axes = plt.subplots(len(METRICS), 1, sharex=True, squeeze=False,
                             figsize=(width, HEIGHT_PER_METRIC * len(METRICS) + 0.7))

    jitter = np.random.default_rng(JITTER_SEED).uniform(-0.17, 0.17, n_cases)
    for index, (ax, metric) in enumerate(zip(axes[:, 0], METRICS)):
        values = [distributions[method][metric] for method in methods]
        artists = ax.boxplot(values, positions=positions, widths=0.52, patch_artist=True, manage_ticks=False,
                            showmeans=True, showfliers=not SHOW_CASE_POINTS,
                            meanprops=dict(marker="D", markerfacecolor="#222222", markeredgecolor="white", markersize=5),
                            medianprops=dict(color="#222222", linewidth=1.6),
                            whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"),
                            flierprops=dict(marker="o", markersize=3, alpha=0.45))
        for patch, method in zip(artists["boxes"], methods):
            patch.set_facecolor(COLORS.get(method, "#999999"))
            patch.set_alpha(0.68)
            patch.set_edgecolor("#444444")
        if SHOW_CASE_POINTS:
            for x, array in zip(positions, values):
                ax.scatter(x + jitter, array, s=11, color="#333333", alpha=0.22, zorder=2)
        ax.set_ylabel(METRIC_INFO[metric])
        ax.set_title(f"({chr(97 + index)})", loc="left", fontsize=FONT_SIZE, fontweight="bold")
        ax.grid(axis="y", color="#E5E8EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.12)

    axes[-1, 0].set_xticks(positions)
    axes[-1, 0].set_xticklabels([LABELS.get(m, m) for m in methods], fontsize=FONT_SIZE)
    axes[-1, 0].tick_params(axis="x", length=0, pad=9)

    fig.suptitle("3D" if mode == "3d" else "2.5D", fontsize=FONT_SIZE + 2, y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.08, hspace=0.24)
    for extension in SAVE_FORMATS:
        fig.savefig(output_stem.with_suffix("." + extension), dpi=DPI, facecolor="white", bbox_inches="tight")
    if not SHOW_FIGURES:
        plt.close(fig)


def main():
    if not METRICS or any(metric not in METRIC_INFO for metric in METRICS):
        raise ValueError(f"Supported metrics: {tuple(METRIC_INFO)}")
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f"Missing summary: {SUMMARY_CSV}; run DL/compare_results.py first")
    if not SHOW_FIGURES:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_SIZE,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    groups = defaultdict(list)
    for row in read_csv(SUMMARY_CSV):

        
        groups[(row["mode"], row["comparison_group"])].append(row)
    if not groups:
        raise ValueError(f"Empty summary: {SUMMARY_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode_counts = Counter(mode for mode, _ in groups)
    for (mode, key), rows in sorted(groups.items()):
        values, count, mask = load_group(rows)
        print("Selected runs: " + "; ".join(f"{r['method']}={r['run_name']}" for r in rows), flush=True)
        stem = "3D" if mode == "3d" else "2p5D"
        if mode_counts[mode] > 1:

            stem = key if key.startswith(stem + "_protocol") else stem + "_" + key
        draw_group(plt, values, count, mode, mask, OUTPUT_DIR / stem)
        print(f"{stem}: {len(rows)} methods, {count} cases; saved {' / '.join(SAVE_FORMATS)}", flush=True)
    print(f"Figure directory: {OUTPUT_DIR}", flush=True)
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
