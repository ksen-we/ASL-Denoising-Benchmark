"""Recreate manuscript boxplots from completed, matched validation results."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np


METRICS = ("ssim", "psnr", "mae_raw", "rmse_raw", "nmse_raw")


def import_plotter(project_root, module_path=None):
    candidates = ([module_path] if module_path else [
        project_root / "GitHub_Release" / "DL" / "plot_comparison.py",
        project_root / "DL" / "plot_comparison.py",
    ])
    path = next((Path(p) for p in candidates if Path(p).is_file()), None)
    if path is None:
        raise FileNotFoundError("Use --plot-module to locate DL/plot_comparison.py.")
    spec = importlib.util.spec_from_file_location("asl_comparison_plotter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def result_row(folder, method, mode, run_name, plotter):
    info = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    config = info["config"]
    if method not in ("BM4D", "DTCWT_NLM"):
        if (config.get("normalization"), config.get("evaluation_protocol")) != ("minmax", "legacy"):
            raise ValueError(f"Expected final minmax + legacy results: {folder}")
    records = plotter.read_csv(folder / "metrics.csv")
    return {
        "method": method, "mode": mode, "run_name": run_name,
        "n_cases": len(records), "result_dir": str(folder.resolve()),
        "evaluation_mask": config["mask_mode"],
    }, info, sorted(record["case"] for record in records)


def composite(plt, plotter, groups, output_stem, residual):
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(len(METRICS), 2, figsize=(18, 9.8), sharey="row")
    for column, (mode, distributions, count) in enumerate(groups):
        methods = list(distributions)
        positions = np.arange(1, len(methods) + 1)
        jitter = np.random.default_rng(plotter.JITTER_SEED).uniform(-0.17, 0.17, count)
        for row, metric in enumerate(plotter.METRICS):
            ax = axes[row, column]
            values = [distributions[method][metric] for method in methods]
            artists = ax.boxplot(
                values, positions=positions, widths=0.52, patch_artist=True,
                manage_ticks=False, showmeans=True, showfliers=False,
                meanprops=dict(marker="D", markerfacecolor="#222222", markeredgecolor="white", markersize=5),
                medianprops=dict(color="#222222", linewidth=1.6),
                whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"),
            )
            for patch, method in zip(artists["boxes"], methods):
                patch.set_facecolor(plotter.COLORS.get(method, "#999999"))
                patch.set_alpha(0.68)
                patch.set_edgecolor("#444444")
            for x, array in zip(positions, values):
                ax.scatter(x + jitter, array, s=11, color="#333333", alpha=0.22, zorder=2)
            ax.set_title(f"({chr(97 + row * 2 + column)})", loc="left", fontsize=11, fontweight="bold")
            ax.grid(axis="y", color="#E5E8EB", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(y=0.12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_xticks(positions)
            if row == len(METRICS) - 1:
                ax.set_xticklabels([plotter.LABELS.get(m, m) for m in methods], fontsize=11)
                ax.tick_params(axis="x", length=0, pad=9)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            if column == 0:
                ax.set_ylabel(plotter.METRIC_INFO[metric])
        axes[0, column].text(0.5, 1.14, "2.5D" if mode == "2.5d" else "3D",
                             transform=axes[0, column].transAxes, ha="center", fontsize=13)
    title = "Residual prediction" if residual else "Direct prediction"
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.09, hspace=0.45, wspace=0.10)
    for extension in plotter.SAVE_FORMATS:
        fig.savefig(output_stem.with_suffix("." + extension), dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dl-result-root", type=Path)
    parser.add_argument("--traditional-root", type=Path)
    parser.add_argument("--plot-module", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=54)
    args = parser.parse_args()
    project = args.project_root.resolve()
    data = args.data_root or project / "Data" / "Dataset001"
    dl_root = args.dl_result_root or data / "DLResult"
    traditional = args.traditional_root or data / "TraditionResult"
    plotter, module_path = import_plotter(project, args.plot_module)
    plotter.METRICS = METRICS
    plotter.METRIC_INFO.update(mae_raw="MAE (a.u.)", rmse_raw="RMSE (a.u.)", nmse_raw="NMSE")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "asl_manuscript_matplotlib"))
    plotter.matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_cases, common_protocol = None, None
    evidence = {"script": Path(__file__).name,
                "plot_module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "run_prefix": "wtm0mask_ori", "metrics": list(METRICS), "groups": []}
    for residual in (0, 1):
        groups = []
        run_name = f"wtm0mask_ori_gres{residual}_seed42"
        for mode, directory in (("2.5d", "2p5D"), ("3d", "3D")):
            selections = [(method, dl_root / method / directory / run_name / "inference" / "val", run_name)
                          for method in ("UNet", "Pix2Pix", "SwinUNet", "SwinIR")]
            if mode == "3d":
                selections += [(method, traditional / method / "evaluation" / "val" / "wtM0mask",
                                "wtm0mask_seed42") for method in ("BM4D", "DTCWT_NLM")]
            rows, checksums = [], {}
            for method, folder, name in selections:
                row, info, cases = result_row(folder, method, mode, name, plotter)
                protocol = (info["data_fingerprint"], info["metric_protocol"])
                if common_cases is None:
                    common_cases, common_protocol = cases, protocol
                if cases != common_cases or protocol != common_protocol:
                    raise ValueError("All four plot groups must share cases and evaluation protocol.")
                rows.append(row)
                checksums[method] = hashlib.sha256((folder / "metrics.csv").read_bytes()).hexdigest()
            distributions, count, mask = plotter.load_group(rows)
            if count != args.expected_cases:
                raise ValueError(f"Expected {args.expected_cases} cases; found {count}.")
            stem = args.output_dir / f"quantitative_gres{residual}_{directory}"
            plotter.draw_group(plt, distributions, count, mode, mask, stem)
            groups.append((mode, distributions, count))
            evidence["groups"].append({"global_residual": bool(residual), "mode": mode,
                                       "n_cases": count, "mask": mask,
                                       "methods": list(distributions), "metrics_sha256": checksums})
        composite(plt, plotter, groups, args.output_dir / f"quantitative_gres{residual}", residual)
    evidence["metric_protocol"] = common_protocol[1]
    evidence["data_fingerprint"] = common_protocol[0]
    (args.output_dir / "quantitative_provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Saved six PNG/PDF figure pairs; all groups contain {len(common_cases)} matched cases.")


if __name__ == "__main__":
    main()
