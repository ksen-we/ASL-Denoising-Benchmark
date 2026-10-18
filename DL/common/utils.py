"""Reproducibility, checkpoint writing, and experiment visualization."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def setup_logger(run_dir, filename="train.log"):
    logger = logging.getLogger(f"ftasl.{run_dir}.{filename}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(run_dir) / filename, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_runtime(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    return device


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}; train first or set inference_checkpoint")

    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def fingerprint(cases):
    """Track file paths, sizes, and modification times for resume checks."""
    manifest = []
    for case in cases:
        row = {"case": case.name}
        for key in ("source", "target"):
            path = getattr(case, key)
            row[key] = None if path is None else {"path": str(path.resolve()), "bytes": path.stat().st_size,
                                                 "mtime_ns": path.stat().st_mtime_ns}
        if getattr(case, "mask", None) is not None:
            path = case.mask
            row["mask"] = {"path": str(path.resolve()), "sha256": file_hash(path)}
            sidecar = Path(str(path) + ".json")
            if sidecar.exists():
                row["mask"]["policy_sha256"] = file_hash(sidecar)
        manifest.append(row)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return digest, manifest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pyplot(show=False):

    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "ftasl_matplotlib")
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = str(Path(tempfile.gettempdir()) / "ftasl_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_preview(item, prediction, path, show=False):
    """Save a display slice in original intensity units for reproducible rendering."""
    valid = item["valid"]
    z = int(np.argmax(valid.sum(axis=(1, 2))))
    slices = {"case": np.asarray(item["case"]), "z": np.asarray(z),
              "valid": valid[z],
              "input": item["input"][z] * item["scale"] + item["offset"]}
    if prediction is not None:
        slices["prediction"] = prediction[z] * item["scale"] + item["offset"]
    if item["target"] is not None:
        slices["target"] = item["target"][z] * item["scale"] + item["offset"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **slices)
    temporary.replace(path.with_suffix(".npz"))
    render_preview_slices(slices, path, show=show)


def render_preview_slices(slices, path, show=False):
    """Render saved floating-point slices with shared display windows."""
    from common.preview_settings import SIGNAL_LIMITS, ABS_ERROR_LIMITS
    lo, hi = SIGNAL_LIMITS
    error_lo, error_hi = ABS_ERROR_LIMITS
    if not (np.isfinite([lo, hi, error_lo, error_hi]).all()
            and lo < hi and error_lo == 0 and error_hi > 0):
        raise ValueError("Preview ranges must be finite and increasing; absolute-error ranges must start at 0")
    plt = pyplot(show)
    x, p, y = slices["input"], slices.get("prediction"), slices.get("target")
    images = [(x, "Low SNR")]
    if p is not None:
        images.append((p, "Denoised"))
    if y is not None:
        images += [(y, "High SNR target"), (abs((x if p is None else p) - y),
                                            "Input absolute error" if p is None else "Absolute error")]
    fig, axes = plt.subplots(1, len(images), figsize=(3.7 * len(images), 4.1),
                             squeeze=False, constrained_layout=True)
    intensity_axes, intensity_image = [], None
    for ax, (im, title) in zip(axes[0], images):
        is_error = title.endswith("error")
        display = np.where(slices["valid"], im, np.nan)
        rendered = ax.imshow(np.rot90(display), cmap="magma" if is_error else "gray",
                             vmin=error_lo if is_error else lo,
                             vmax=error_hi if is_error else hi,
                             interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        if is_error:

            bar = fig.colorbar(rendered, ax=ax, orientation="horizontal",
                               shrink=0.85, pad=0.04, fraction=0.06, extend="max")
            bar.set_label("Absolute error (original PWI units)", fontsize=9)
            bar.ax.tick_params(labelsize=8)
        else:
            intensity_axes.append(ax)
            intensity_image = rendered

    bar = fig.colorbar(intensity_image, ax=intensity_axes, orientation="horizontal",
                       shrink=0.65, pad=0.04, fraction=0.06, extend="both")
    bar.set_label("Signal intensity (original PWI units)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.suptitle(f"{slices['case']} | z={int(slices['z'])}", fontsize=10)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, metadata={
        "ftasl_colorbar_version": "2",
        "ftasl_preview_limits": json.dumps({"signal": [lo, hi],
                                             "absolute_error": [error_lo, error_hi]}),
    })
    if show:
        plt.show(block=False)
        plt.pause(0.05)
    plt.close(fig)


def save_curves(history, path, show=False):
    if not history:
        return
    plt = pyplot(show)
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    axes[0].plot(epochs, [r["train_recon"] for r in history], label="train patches")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validation volumes")
    axes[0].set_title("Reconstruction loss (training scale)")
    axes[0].legend()

    protocols = list(dict.fromkeys(r.get("evaluation_protocol", "legacy") for r in history))
    for protocol in protocols:
        label = protocol.split(";")[0]
        for axis, key in ((axes[1], "val_ssim"), (axes[2], "val_psnr")):
            axis.plot(epochs, [r[key] if r.get("evaluation_protocol", "legacy") == protocol else np.nan
                               for r in history], label=label)
            if len(protocols) > 1:
                axis.legend(fontsize=7)
    axes[1].set_title("Validation SSIM")
    axes[2].set_title("Validation PSNR (dB)")
    axes[3].plot(epochs, [r["lr_g"] for r in history], label="G")
    if any(r.get("lr_d") is not None for r in history):
        axes[3].plot(epochs, [r.get("lr_d") for r in history], linestyle="--", label="D")
    axes[3].set_title("Learning rate (next epoch)")
    axes[3].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    if show:
        plt.show(block=False)
        plt.pause(0.05)
    plt.close(fig)
