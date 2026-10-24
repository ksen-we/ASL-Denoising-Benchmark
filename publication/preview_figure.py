"""Create the separate ISMRM 2027 program preview, not a replacement for Figure 1.

The current call requires a preview; the submission guide specifies 3:2 and 1 MB.
Only the PNG is intended for upload. The PDF is a viewing companion.
"""

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from qualitative_comparison import check_runs, load_case, select_cases
import matplotlib.pyplot as plt
from matplotlib import colors


def draw_preview(cases, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig = plt.figure(figsize=(12, 8), dpi=100, facecolor="white")
    gray = plt.get_cmap("gray").copy()
    gray.set_bad("black")
    for column, label in enumerate(("Input", "Denoised", "Reference")):
        fig.text(0.20 + column * 0.30, 0.975, label, fontsize=32,
                 fontweight="bold", ha="center", va="top")
    for row, case in enumerate(cases):
        volumes = (case["input"], case["predictions"][("SwinUNet", "3D", 1)],
                   case["reference"])
        norm = colors.Normalize(*case["pwi_limits"])
        for column, volume in enumerate(volumes):
            ax = fig.add_axes([0.06 + column * 0.30, 0.48 - row * 0.44, 0.28, 0.42])
            plane = np.ma.masked_invalid(np.rot90(volume[:, :, case["z"]]))
            ax.imshow(plane, cmap=gray, norm=norm, interpolation="nearest")
            ax.set_axis_off()
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "ismrm2027_preview.png"
    pdf = output_dir / "ismrm2027_preview.pdf"
    fig.savefig(png, dpi=100, facecolor="white",
                pil_kwargs={"compress_level": 9, "optimize": True})
    fig.savefig(pdf, facecolor="white", metadata={"Title": "ASL PWI denoising preview"})
    plt.close(fig)
    data = png.read_bytes()
    if struct.unpack(">II", data[16:24]) != (1200, 800) or len(data) > 1_000_000:
        raise ValueError("The preview must be 1200 × 800 pixels and no larger than 1 MB.")
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--run-prefix", default="wtm0mask_ori")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-cases", type=int, default=54)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    folders, baseline, identities, _ = check_runs(
        root, args.split, args.run_prefix, args.seed, args.expected_cases)
    selected = select_cases(baseline)
    cases = [load_case(root, args.split, case, folders) for case in selected]
    if args.selection_manifest:
        previous = json.loads(args.selection_manifest.read_text())
        if previous["runs"] != identities or len(previous["selected_cases"]) != len(selected):
            raise ValueError("The preview and main figure must use the same completed evaluations.")
        for current, saved in zip(selected, previous["selected_cases"]):
            if any(saved.get(key) != value for key, value in current.items()):
                raise ValueError("Preview cases, slices, source files or display ranges differ from Figure 1.")
    png, pdf = draw_preview(cases, args.output_dir)
    print(f"Saved {png.name}: 1200 × 800 pixels, 3:2, {png.stat().st_size} bytes.")
    print(f"Saved {pdf.name}: {pdf.stat().st_size} bytes.")
    print("Verified the same two-case selection, native slices and display windows as the main figure."
          if args.selection_manifest else "Used the main figure's case, slice and display-window rules.")
    print("The middle column uses 3D residual Swin U-Net; this preview is a separate program asset.")


if __name__ == "__main__":
    main()
