"""Render direct and residual PWI denoising results as separate tables."""

import argparse
import csv
import math
from itertools import groupby
from pathlib import Path


LABELS = {"Input": "Noisy PWI", "BM4D": "BM4D", "DTCWT_NLM": "DT-CWT-NLM",
          "UNet": "U-Net", "Pix2Pix": "Pix2Pix", "SwinUNet": "Swin U-Net", "SwinIR": "SwinIR"}
METHOD_ORDER = {name: index for index, name in enumerate(LABELS)}
METRICS = (("ssim", 4, max), ("psnr", 2, max), ("rmse_raw", 2, min), ("nmse_raw", 4, min))
HEADERS = ("Method", "Dimension", "SSIM ↑", "PSNR\n(dB) ↑", "RMSE ↓", "NMSE ↓")
COLUMN_FRACTIONS = (0.22, 0.14, 0.16, 0.16, 0.16, 0.16)
TABLE_TITLES = {
    "direct": "Table 1. Direct prediction and conventional denoising methods.",
    "residual": "Table 2. Residual prediction.",
}
SUMMARY_NOTE = "Values are mean ± SD across 54 validation cases."


def group_key(row):
    if row["method"] == "Input":
        return 0
    if row["method"] in ("BM4D", "DTCWT_NLM"):
        return 1
    return 2 if row["mode"] == "2.5d" else 3


def sort_key(row):
    return group_key(row), METHOD_ORDER[row["method"]]


def dimension_label(row):
    return "—" if row["method"] == "Input" else "2.5D" if row["mode"] == "2.5d" else "3D"


def dimension_groups(rows):
    for _, members in groupby(enumerate(rows), key=lambda item: group_key(item[1])):
        yield list(members)


def read_summary(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 19 or any(int(row["n_cases"]) != 54 for row in rows):
        raise ValueError("Expected 19 final configurations with 54 validation cases each.")
    keys = set()
    for row in rows:
        if row["normalization"] != "minmax" or row["evaluation_protocol"] != "legacy":
            raise ValueError("The final summary must use minmax + legacy.")
        key = row["method"], row["mode"], row["global_residual"]
        if key in keys:
            raise ValueError("Duplicate configuration in the validation summary.")
        keys.add(key)
        if row["method"] not in LABELS:
            raise ValueError("Unknown denoising method in the validation summary.")
        for metric, _, _ in METRICS:
            if not all(math.isfinite(float(row[metric + suffix])) for suffix in ("_mean", "_std")):
                raise ValueError("Nonfinite table values are not supported.")
    return rows


def partition_summary(rows):
    direct = sorted((row for row in rows if group_key(row) < 2 or row["global_residual"] == "False"),
                    key=sort_key)
    residual = sorted((row for row in rows if group_key(row) >= 2 and row["global_residual"] == "True"),
                      key=sort_key)
    if len(direct) != 11 or len(residual) != 8:
        raise ValueError("Expected 11 direct/baseline rows and 8 residual rows.")
    for selection in (direct[3:], residual):
        if {(row["method"], row["mode"]) for row in selection} != {
            (method, mode) for method in ("UNet", "Pix2Pix", "SwinUNet", "SwinIR")
            for mode in ("2.5d", "3d")
        }:
            raise ValueError("Each output formulation must contain all four networks in both dimensions.")
    return (("direct", direct), ("residual", residual))


def best_means(rows):
    return {metric: direction(float(row[metric + "_mean"]) for row in rows)
            for metric, _, direction in METRICS}


def render_table(kind, rows, output_dir):
    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asl_manuscript_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "asl_manuscript_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in installed else "Liberation Serif" if "Liberation Serif" in installed else "DejaVu Serif"
    plt.rcParams.update({"font.family": serif, "font.size": 10.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    page_width, page_height = 210 / 25.4, 297 / 25.4
    left = right = 1.25
    top = bottom = 1.0
    usable_width = (page_width - left - right) * 72
    usable_height = (page_height - top - bottom) * 72
    fig = plt.figure(figsize=(page_width, page_height))
    ax = fig.add_axes((left / page_width, bottom / page_height,
                       usable_width / (page_width * 72), usable_height / (page_height * 72)))
    ax.set_xlim(0, usable_width)
    ax.set_ylim(usable_height, 0)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    caption_fonts = {bold: font_manager.FontProperties(family=serif, size=12,
                                                       weight="bold" if bold else "normal")
                     for bold in (False, True)}

    def width(text, font):
        return renderer.get_text_width_height_descent(text, font, ismath=False)[0] * 72 / fig.dpi

    caption_words = ([(word, True) for word in TABLE_TITLES[kind].split()]
                     + [(word, False) for word in SUMMARY_NOTE.split()])
    lines, line, occupied = [], [], 0.0
    for word, bold in caption_words:
        word_width = width(word, caption_fonts[bold])
        space = width(" ", caption_fonts[bold])
        if line and occupied + space + word_width > usable_width:
            lines.append(line)
            line, occupied = [], 0.0
        line.append((word, bold, word_width))
        occupied += word_width + (space if len(line) > 1 else 0)
    if line:
        lines.append(line)
    for number, line in enumerate(lines):
        occupied = sum(item[2] for item in line)
        gap = ((usable_width - occupied) / (len(line) - 1)
               if number < len(lines) - 1 and len(line) > 1
               else width(" ", caption_fonts[False]))
        x = 0.0
        for word, bold, word_width in line:
            ax.text(x, number * 16.2, word, ha="left", va="top",
                    fontproperties=caption_fonts[bold])
            x += word_width + gap

    table_top = len(lines) * 16.2 + 8
    header_height, row_height = 34.0, 34.0
    table_bottom = table_top + header_height + len(rows) * row_height
    if table_bottom > usable_height:
        raise ValueError("The portrait table exceeds the template's printable height.")
    boundaries = [0.0]
    for fraction in COLUMN_FRACTIONS:
        boundaries.append(boundaries[-1] + fraction * usable_width)
    centers = [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
    centers[0] = 3.0
    for column, (position, label) in enumerate(zip(centers, HEADERS)):
        ax.text(position, table_top + header_height / 2, label,
                ha="left" if column == 0 else "center", va="center",
                fontweight="bold", fontsize=10.5, linespacing=1.0)
    ax.hlines((table_top, table_top + header_height, table_bottom), 0, usable_width,
              color="black", linewidth=(1.2, 0.8, 1.2))
    positions = [table_top + header_height + (index + 0.5) * row_height
                 for index in range(len(rows))]
    for index in range(1, len(rows)):
        y = table_top + header_height + index * row_height
        if group_key(rows[index - 1]) != group_key(rows[index]):
            ax.hlines(y, 0, usable_width, color="black", linewidth=0.8)
        else:
            ax.hlines((y, y), (0, boundaries[2]), (boundaries[1], usable_width),
                      color="black", linewidth=0.35)
    best = best_means(rows)
    for index, row in enumerate(rows):
        y = positions[index]
        ax.text(centers[0], y, LABELS[row["method"]], ha="left", va="center", fontsize=10.5)
        for column, (metric, digits, _) in enumerate(METRICS, start=2):
            mean, std = float(row[metric + "_mean"]), float(row[metric + "_std"])
            ax.text(centers[column], y - 6.3, f"{mean:.{digits}f}",
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if mean == best[metric] else "normal")
            ax.text(centers[column], y + 6.3, f"± {std:.{digits}f}",
                    ha="center", va="center", fontsize=10.5, fontweight="normal")
    for members in dimension_groups(rows):
        y = (positions[members[0][0]] + positions[members[-1][0]]) / 2
        ax.text(centers[1], y, dimension_label(members[0][1]), ha="center", va="center", fontsize=10.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"validation_table_{kind}.{extension}",
                    dpi=300, facecolor="white")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for kind, rows in partition_summary(read_summary(args.summary)):
        render_table(kind, rows, args.output_dir)
        print(f"Saved validation_table_{kind}.png and .pdf ({len(rows)} rows, {len(METRICS)} metrics).")


if __name__ == "__main__":
    main()
