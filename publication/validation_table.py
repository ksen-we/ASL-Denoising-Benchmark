"""Render the final validation summary as a monochrome three-line table."""
import argparse
import csv
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asl_manuscript_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "asl_manuscript_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"Input": "Low-SNR input", "BM4D": "BM4D", "DTCWT_NLM": "DT-CWT NLM",
          "UNet": "U-Net", "Pix2Pix": "Pix2Pix", "SwinUNet": "Swin U-Net", "SwinIR": "SwinIR"}
METHOD_ORDER = {name: index for index, name in enumerate(LABELS)}
METRICS = (("ssim", 4, max), ("psnr", 2, max), ("rmse_raw", 2, min))


def sort_key(row):
    if row["method"] in ("Input", "BM4D", "DTCWT_NLM"):
        return (0, METHOD_ORDER[row["method"]], 0)
    return (1 if row["mode"] == "2.5d" else 2,
            METHOD_ORDER[row["method"]], row["global_residual"] == "True")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=sort_key)
    if len(rows) != 19 or any(int(row["n_cases"]) != 54 for row in rows):
        raise ValueError("Expected 19 final configurations with 54 validation cases each.")
    for row in rows:
        if row["normalization"] != "minmax" or row["evaluation_protocol"] != "legacy":
            raise ValueError("The final summary must use minmax + legacy.")
        for metric, _, _ in METRICS:
            if not all(math.isfinite(float(row[metric + suffix])) for suffix in ("_mean", "_std")):
                raise ValueError("Nonfinite table values are not supported.")
    best = {metric: direction(float(row[metric + "_mean"]) for row in rows)
            for metric, _, direction in METRICS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.5, 8.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.8, len(rows) + 1.6)
    ax.axis("off")
    x = (0.015, 0.26, 0.39, 0.565, 0.75, 0.93)
    headers = ("Method", "Dimension", "Output", "SSIM ↑", "PSNR (dB) ↑", "RMSE ↓")
    for column, (position, label) in enumerate(zip(x, headers)):
        ax.text(position, len(rows) + 0.65, label,
                ha="left" if column == 0 else "center", va="center", fontweight="bold")
    ax.hlines((len(rows) + 1.25, len(rows) + 0.12, -0.12),
              xmin=0, xmax=1, color="black", linewidth=(1.2, 0.8, 1.2))
    for index, row in enumerate(rows):
        y = len(rows) - index - 0.5
        is_dl = row["method"] not in ("Input", "BM4D", "DTCWT_NLM")
        dimension = "2.5D" if row["mode"] == "2.5d" else ("—" if row["method"] == "Input" else "3D")
        output = ("Residual" if row["global_residual"] == "True" else "Direct") if is_dl else "—"
        for column, value in enumerate((LABELS[row["method"]], dimension, output)):
            ax.text(x[column], y, value, ha="left" if column == 0 else "center", va="center")
        for column, (metric, digits, _) in enumerate(METRICS, start=3):
            mean, std = float(row[metric + "_mean"]), float(row[metric + "_std"])
            ax.text(x[column], y, f"{mean:.{digits}f} ± {std:.{digits}f}",
                    ha="center", va="center",
                    fontweight="bold" if mean == best[metric] else "normal")
    ax.text(0, -0.72, "Values are mean ± SD across 54 validation cases. Bold indicates the best overall mean.",
            ha="left", va="center", fontsize=10)
    ax.text(0, -1.28, "Direct: predicted image; residual: output includes the input image. RMSE uses original PWI intensity units.",
            ha="left", va="center", fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.98, bottom=0.025)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("validation_table." + extension),
                    dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved validation_table.png and validation_table.pdf (19 rows, three horizontal rules).")


if __name__ == "__main__":
    main()
