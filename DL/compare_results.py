"""Compare completed runs only within matching data and evaluation protocols."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

from common.config import DATA_ROOT
from common.utils import pyplot, write_csv, write_json

OUTPUT_ROOT = DATA_ROOT / "DLResult"
RUN_NAME = "wtm0mask_ori_gres0_seed42"
SPLIT = "val"
TRADITION_ROOT = DATA_ROOT / "TraditionResult"


METHOD_RUN_NAMES = {
    "BM4D": "wtm0mask_seed42",
    "DTCWT_NLM": "wtm0mask_seed42",
}


def comparison_group_names(protocols):
    """Name groups by dimension and disambiguate different evaluation protocols."""
    dimensions = {key: {"3d": "3D", "2.5d": "2p5D"}[protocol["mode"]]
                  for key, protocol in protocols.items()}
    counts, seen, names = Counter(dimensions.values()), Counter(), {}
    for key in sorted(protocols):
        dimension = dimensions[key]
        seen[dimension] += 1
        names[key] = (dimension if counts[dimension] == 1
                      else f"{dimension}_protocol{seen[dimension]}")
    return names


def output_settings(cfg):
    """Read output options, including compatibility with legacy model defaults."""
    if cfg["method"] not in {"SwinIR", "Pix2Pix", "SwinUNet", "UNet"}:
        return {"global_residual": "", "zero_init_output": ""}
    legacy = cfg["method"] == "SwinIR" or (
        cfg["method"] == "UNet" and cfg.get("model", {}).get("residual", False))
    return {key: legacy if cfg.get(key) is None else cfg[key]
            for key in ("global_residual", "zero_init_output")}


def comparison_metrics(summary):
    """Report prediction metrics and prediction-minus-input changes in the same units."""
    names = {"ssim": "ssim", "psnr": "psnr", "mae": "mae_raw",
             "rmse": "rmse_raw", "nmse": "nmse_raw"}
    values = {"n_cases": summary["n_cases"]}
    for name, source in names.items():
        for statistic in ("mean", "std"):
            values[f"{name}_{statistic}"] = summary[f"{source}_{statistic}"]
        values[f"delta_{name}"] = summary[f"{source}_mean"] - summary[f"input_{source}_mean"]
    return values


def compare(output_root=OUTPUT_ROOT, run_name=RUN_NAME, split=SPLIT, tradition_root=TRADITION_ROOT,
            method_run_names=None):
    method_run_names = dict(method_run_names or {})
    groups = {}
    paths = list(Path(output_root).glob(f"*/*/*/inference/{split}/provenance.json"))
    paths += list(Path(output_root).glob(f"*/*/*/*/inference/{split}/provenance.json"))
    if tradition_root is not None:
        paths += list(Path(tradition_root).glob(f"*/evaluation/{split}/provenance.json"))
        paths += list(Path(tradition_root).glob(f"*/evaluation/{split}/*/provenance.json"))
    for provenance in sorted(paths):
        folder = provenance.parent
        if (any((folder / name).exists() for name in (".inference.lock", ".evaluation.lock"))
                or not (folder / "completed.json").is_file()):
            continue
        info = json.loads(provenance.read_text(encoding="utf-8"))
        if info.get("family") == "traditional" and (Path(info.get("denoising_root", folder.parents[1])) / ".denoising.lock").exists():
            continue
        cfg = info["config"]
        if cfg["method"] == "Pix2Pix" and not cfg["use_gan"]:
            continue
        selected_run = method_run_names.get(cfg["method"], run_name)
        if selected_run is not None and cfg["run_name"] != selected_run:
            continue
        if not (folder / "metrics.csv").is_file():
            continue
        with (folder / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
            metrics = list(csv.DictReader(f))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if not metrics or len(metrics) != summary["n_cases"]:
            continue
        metric_keys = ("source_name", "target_name")
        if not info["metric_protocol"].startswith("reference_minmax_v1;"):
            metric_keys += ("l1_weight", "ssim_weight", "ssim_window", "ssim_sigma", "ssim_data_range")
        protocol = {"data_fingerprint": info["data_fingerprint"], "metric_protocol": info["metric_protocol"],
                    "cases": sorted(r["case"] for r in metrics), "mode": cfg["mode"],
                    **{k: cfg[k] for k in metric_keys}}
        key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:10]
        group = groups.setdefault(key, {"protocol": protocol, "runs": []})
        group["runs"].append((cfg, summary, metrics, folder, info))


    for method, selected_run in method_run_names.items():
        modes = {cfg["mode"] for group in groups.values() for cfg, *_ in group["runs"]
                 if cfg["method"] == method and (selected_run is None or cfg["run_name"] == selected_run)}
        expected = {"3d"} if method in {"BM4D", "DTCWT_NLM"} else {"3d", "2.5d"}
        if modes != expected:
            raise ValueError(f"Selected run {method}/{selected_run} requires dimensions {sorted(expected)}; found {sorted(modes)}")
    if not groups:
        print("No matching completed inference results. Train with automatic inference or set ACTION='infer'.")
        return []
    output = Path(output_root) / "_comparison" / (run_name or "all_runs") / split
    names = comparison_group_names({key: group["protocol"] for key, group in groups.items()})
    rows = []
    for key, group in groups.items():
        group_name = names[key]
        plt = pyplot()
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
        short_names = {"Pix2Pix": "Pix2Pix", "UNet": "U-Net", "SwinUNet": "Swin-Unet", "DTCWT_NLM": "DT-CWT\nNLM"}
        labels = [short_names.get(cfg["method"], cfg["method"]) for cfg, *_ in group["runs"]]
        metric_labels = {"ssim": "SSIM", "psnr": "PSNR (dB)", "mae_raw": "MAE (original units)"}
        for index, (ax, metric) in enumerate(zip(axes, metric_labels)):
            values = [[float(r[metric]) for r in metrics] for _, _, metrics, _, _ in group["runs"]]
            ax.boxplot(values, showmeans=True, manage_ticks=False)
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f"({chr(97 + index)})", loc="left", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=11)
            ax.grid(axis="y", alpha=0.2)
        axes[-1].set_xticks(range(1, len(labels) + 1))
        axes[-1].set_xticklabels(labels)
        dimension = group["protocol"]["mode"]
        fig.suptitle("3D" if dimension == "3d" else "2.5D", fontsize=14)
        for cfg, summary, _, folder, info in group["runs"]:
            rows.append({"comparison_group": group_name, "method": cfg["method"], "mode": cfg["mode"],
                         "objective": ("classical" if info.get("family") == "traditional" else
                                       "L1+SSIM+adversarial" if cfg["use_gan"] else "L1+SSIM"),
                         "adversarial_weight": cfg["adversarial_weight"] if cfg["use_gan"] else 0,
                         "adversarial_mask_mode": cfg.get("adversarial_mask_mode", "none") if cfg["use_gan"] else "none",
                         "run_name": cfg["run_name"],
                         **output_settings(cfg),
                         "evaluation_mask": cfg.get("mask_mode", "none"),
                         "seed": cfg["seed"], "checkpoint_epoch": info["checkpoint_epoch"],
                         "result_provenance": info.get("denoising_provenance", "checkpoint"),
                         "backend": ((info.get("denoising_run") or {}).get("config") or {}).get("backend", ""),
                         "inference_tiling": str(cfg["inference_patch_size"]),
                         **comparison_metrics(summary), "result_dir": str(folder)})
        group_dir = output / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(group_dir / "comparison.png", dpi=150)
        plt.close(fig)
        write_json(group_dir / "protocol.json", group["protocol"])
        write_csv(group_dir / "summary.csv", [r for r in rows if r["comparison_group"] == group_name])
    write_csv(output / "all_results.csv", rows)
    print(f"Summarized {len(rows)} runs in {len(groups)} comparable groups: {output}")
    return rows


if __name__ == "__main__":
    compare(method_run_names=METHOD_RUN_NAMES)
