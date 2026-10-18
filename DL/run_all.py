"""Edit the settings below, then run this file to train, infer, or check data."""
from importlib import import_module
import json

from common.config import ExperimentConfig
from common.engine import run_experiment
from common.utils import load_checkpoint, write_json


METHODS = ("SwinIR", "Pix2Pix", "SwinUNet", "UNet")
MODES = ("3d", "2.5d")
ACTION = "train"  # "train", "infer", or "check"
RUN_NAME_BASE = "wtm0mask_ori"
GLOBAL_RESIDUAL_OPTIONS = (False, True)  # Use (False,) or (True,) for one variant.
DEVICE = "auto"
RESUME = False
OVERWRITE_OUTPUTS = False  # Protect existing experiments by default.
CONTINUE_ON_ERROR = False


# Precedence: shared defaults -> method defaults -> these two override dictionaries.
CONFIG_OVERRIDES = {
    "epochs": 200,
    "lr": 2e-4,
    "num_workers": 4,
    "seed": 42,
    "normalization": "minmax",
    "evaluation_protocol": "legacy",
}

EXPERIMENT_OVERRIDES = {
    ("Pix2Pix", "2.5d", False): {"lr_policy": "pix2pix_linear",
        "early_stopping_start": "lr_decay", "adversarial_mask_mode": "foreground_fraction"},
    ("Pix2Pix", "2.5d", True): {"lr_policy": "pix2pix_linear",
        "early_stopping_start": "lr_decay", "adversarial_mask_mode": "foreground_fraction"},
    ("Pix2Pix", "3d", False): {"lr_policy": "pix2pix_linear",
        "early_stopping_start": "lr_decay", "adversarial_mask_mode": "foreground_fraction"},
    ("Pix2Pix", "3d", True): {"lr_policy": "pix2pix_linear",
        "early_stopping_start": "lr_decay", "adversarial_mask_mode": "foreground_fraction"},
}


def apply_overrides(cfg, values):
    reserved = {"method", "mode", "global_residual", "run_name", "device", "resume_checkpoint", "overwrite_outputs"}
    for key, value in values.items():
        if key not in cfg.__dataclass_fields__:
            raise ValueError(f"Unknown configuration key: {key}")
        if key in reserved:
            raise ValueError(f"Set {key} using the dedicated options at the top of run_all.py")
        if key == "model":
            cfg.model = {**cfg.model, **value}
        else:
            setattr(cfg, key, value)


def build_config(method, mode, enabled):
    """Build one experiment without starting training; resume restores the saved configuration."""
    if type(OVERWRITE_OUTPUTS) is not bool:
        raise ValueError("OVERWRITE_OUTPUTS must be True or False")
    if RESUME and OVERWRITE_OUTPUTS:
        raise ValueError("RESUME and OVERWRITE_OUTPUTS cannot both be True")
    cfg = import_module(f"{method}.config").make_config(mode, global_residual=enabled)
    overrides = EXPERIMENT_OVERRIDES.get((method, mode, enabled), {})
    apply_overrides(cfg, CONFIG_OVERRIDES)
    apply_overrides(cfg, overrides)
    cfg.set_global_residual(enabled, run_name_base=RUN_NAME_BASE)
    cfg.device = DEVICE
    if RESUME:
        if ACTION != "train":
            raise ValueError("RESUME=True requires ACTION='train'; disable RESUME for inference")
        checkpoint = cfg.run_dir / "checkpoints" / "last.pt"
        saved = load_checkpoint(checkpoint)
        restored = ExperimentConfig.from_dict(saved["config"])
        effective_residual = restored.global_residual
        if effective_residual is None:
            effective_residual = restored.method == "SwinIR" or (
                restored.method == "UNet" and restored.model.get("residual", False))
        if ((restored.method, restored.mode, restored.run_name) != (method, mode, cfg.run_name)
                or effective_residual != enabled):
            raise ValueError(f"Resume checkpoint does not match the selected experiment: {checkpoint}")
        apply_overrides(restored, CONFIG_OVERRIDES)
        apply_overrides(restored, overrides)
        restored.device = DEVICE
        restored.resume_checkpoint = checkpoint
        cfg = restored

    cfg.overwrite_outputs = OVERWRITE_OUTPUTS  # Never inherit overwrite permission from a checkpoint.
    return cfg.validate()


def main():
    for name, values in (("METHODS", METHODS), ("MODES", MODES),
                         ("GLOBAL_RESIDUAL_OPTIONS", GLOBAL_RESIDUAL_OPTIONS)):
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError(f"{name} must be a nonempty tuple/list; use a trailing comma for singletons, e.g. (True,)")
    outcomes = []
    for enabled in GLOBAL_RESIDUAL_OPTIONS:
        for method in METHODS:
            for mode in MODES:
                cfg = build_config(method, mode, enabled)
                info = {"method": method, "mode": mode, "run_name": cfg.run_name,
                        "global_residual": enabled}
                print(f"\nStarting {method} / {mode} / {cfg.run_name}", flush=True)
                try:
                    run_experiment(cfg, ACTION)
                    outcomes.append({**info, "status": "complete"})
                except Exception as exc:
                    outcomes.append({**info, "status": "failed", "error": repr(exc)})
                    write_json(cfg.output_root / f"batch_{RUN_NAME_BASE}_global_residual.json", outcomes)
                    if not CONTINUE_ON_ERROR:
                        raise
    if outcomes:
        write_json(cfg.output_root / f"batch_{RUN_NAME_BASE}_global_residual.json", outcomes)
    print(json.dumps(outcomes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
