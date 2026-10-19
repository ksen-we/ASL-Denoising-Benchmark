"""Render the final validation summary with monochrome group separators."""
import argparse
import csv
import math
import os
from pathlib import Path
import tempfile
from itertools import groupby

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asl_manuscript_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "asl_manuscript_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"Input": "Noisy PWI", "BM4D": "BM4D", "DTCWT_NLM": "DT-CWT-NLM",
          "UNet": "U-Net", "Pix2Pix": "Pix2Pix", "SwinUNet": "Swin U-Net", "SwinIR": "SwinIR"}
METHOD_ORDER = {name: index for index, name in enumerate(LABELS)}
METRICS = (("ssim", 4, max), ("psnr", 2, max), ("rmse_raw", 2, min), ("nmse_raw", 4, min))


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
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(13.4, 8.1))
    group_gap = 0.35
    body_height = len(rows) + 2 * group_gap
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, body_height + 1.6)
    ax.axis("off")
    x = (0.015, 0.20, 0.30, 0.44, 0.60, 0.75, 0.925)
    headers = ("Method", "Dimension", "Output", "SSIM ↑", "PSNR (dB) ↑", "RMSE ↓", "NMSE ↓")
    for column, (position, label) in enumerate(zip(x, headers)):
        ax.text(position, body_height + 0.65, label,
                ha="left" if column == 0 else "center", va="center", fontweight="bold")
    ax.hlines((body_height + 1.25, body_height + 0.12, -0.12),
              xmin=0, xmax=1, color="black", linewidth=(1.2, 0.8, 1.2))
    positions = [body_height - index - 0.5 - group_gap * sort_key(row)[0]
                 for index, row in enumerate(rows)]
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        y = (positions[index - 1] + positions[index]) / 2
        if previous["method"] == "Input" or sort_key(previous)[0] != sort_key(current)[0]:
            ax.hlines(y, xmin=0, xmax=1, color="black", linewidth=0.75)
        elif previous["method"] != current["method"]:
            # Leave the vertically merged dimension label uninterrupted.
            ax.hlines((y, y), xmin=(0, 0.265), xmax=(0.15, 1), color="black", linewidth=0.375)
    for index, row in enumerate(rows):
        y = positions[index]
        is_dl = row["method"] not in ("Input", "BM4D", "DTCWT_NLM")
        output = ("Residual" if row["global_residual"] == "True" else "Direct") if is_dl else "—"
        ax.text(x[2], y, output, ha="center", va="center")
        for column, (metric, digits, _) in enumerate(METRICS, start=3):
            mean, std = float(row[metric + "_mean"]), float(row[metric + "_std"])
            mean_text = f"{mean:.{digits}f}"
            if mean == best[metric]:
                mean_text = r"$\mathbf{" + mean_text + "}$"
            ax.text(x[column], y, f"{mean_text} ± {std:.{digits}f}", ha="center", va="center")
    for _, members in groupby(enumerate(rows), key=lambda item: (sort_key(item[1])[0], item[1]["method"])):
        members = list(members)
        y = (positions[members[0][0]] + positions[members[-1][0]]) / 2
        ax.text(x[0], y, LABELS[members[0][1]["method"]], ha="left", va="center")
    for _, members in groupby(enumerate(rows), key=lambda item: (sort_key(item[1])[0], item[1]["method"] == "Input")):
        members = list(members)
        row = members[0][1]
        dimension = "—" if row["method"] == "Input" else "2.5D" if row["mode"] == "2.5d" else "3D"
        y = (positions[members[0][0]] + positions[members[-1][0]]) / 2
        ax.text(x[1], y, dimension, ha="center", va="center")
    ax.text(0, -0.72, "Values are mean ± SD across 54 validation cases.",
            ha="left", va="center", fontsize=10)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.98, bottom=0.025)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / ("validation_table." + extension),
                    dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved validation_table.png and validation_table.pdf (19 rows, group and method separators).")


if __name__ == "__main__":
    main()
