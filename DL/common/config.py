"""Shared experiment configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

DL_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = DL_ROOT.parent / "Data" / "Dataset001"


DEFAULT_GLOBAL_RESIDUAL = False
DEFAULT_NORMALIZATION = "minmax"
DEFAULT_RUN_NAME_BASE = "wtm0mask_ori"


@dataclass
class ExperimentConfig:
    method: str = "SwinIR"
    mode: str = "3d"
    run_name: str = f"{DEFAULT_RUN_NAME_BASE}_seed42"
    global_residual: bool | None = None  # None preserves legacy checkpoint behavior.
    zero_init_output: bool | None = None
    data_root: Path = DATA_ROOT
    output_root: Path = DATA_ROOT / "DLResult"
    source_name: str = "rpwi_ave_4.nii"
    target_name: str = "rpwi_ave_max.nii"
    skip_incomplete: bool = True
    mask_mode: str = "m0"
    mask_name: str = "m0_foreground_mask.nii"
    normalization: str = DEFAULT_NORMALIZATION  # Source finite min/max; no clipping.
    evaluation_protocol: str = "legacy"  # SSIM/PSNR use the shared source scale.

    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    cpu_threads: int = 4
    cache_cases: int = 8
    epochs: int = 200
    steps_per_epoch: int = 250
    batch_size: int = 1  # Method factories set 2.5D to 16 independent three-slice samples.
    patch_size: tuple = (16, 48, 48)  # D,H,W; 2.5D factories use H,W only.
    augment: bool = True
    lr: float = 2e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    lr_policy: str = "plateau"
    lr_constant_epochs: int = 100
    lr_decay_epochs: int = 100
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 30  # 0 disables early stopping.
    early_stopping_start: str = "immediate"
    min_delta: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = True


    l1_weight: float = 1.0
    ssim_weight: float = 1.0
    ssim_window: int = 7
    ssim_sigma: float = 1.5
    ssim_data_range: float = 1.0


    use_gan: bool = False
    adversarial_weight: float = 0.01
    adversarial_mask_mode: str = "none"
    discriminator_lr: float = 2e-4
    discriminator_channels: int = 32

    model: dict = field(default_factory=dict)
    inference_batch_size: int = 4
    inference_patch_size: tuple | None = None  # None predicts the full 3D volume.
    inference_overlap: float = 0.5
    inference_split: str = "val"
    inference_case: str | None = None
    resume_checkpoint: Path | None = None
    overwrite_outputs: bool = False
    inference_checkpoint: Path | None = None
    predict_after_training: bool = True
    preview_every: int = 5
    show_plots: bool = False
    log_every: int = 50

    def set_global_residual(self, enabled, run_name_base=DEFAULT_RUN_NAME_BASE):
        """Set the image residual option and its matching run name."""
        if type(enabled) is not bool:
            raise ValueError("global_residual must be True or False")
        self.global_residual = enabled
        self.run_name = f"{run_name_base}_gres{int(enabled)}_seed{self.seed}"
        return self

    @property
    def run_dir(self) -> Path:
        dimension = "3D" if self.mode == "3d" else "2p5D"
        return Path(self.output_root) / self.method / dimension / self.run_name

    @property
    def legacy_run_dir(self) -> Path:
        """Resolve the older objective-specific directory layout."""
        objective = "GAN" if self.use_gan else "Recon"
        return self.run_dir.parent / objective / self.run_name

    def validate(self):
        if type(self.overwrite_outputs) is not bool:
            raise ValueError("overwrite_outputs must be True or False")
        if self.overwrite_outputs and self.resume_checkpoint:
            raise ValueError("Overwriting and resuming cannot be enabled together")
        for name in ("global_residual", "zero_init_output"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be bool or None for legacy compatibility")
        if "global_residual" in self.model or "zero_init_output" in self.model:
            raise ValueError("Set global_residual and zero_init_output at the configuration top level, not inside model")
        if (self.global_residual is not None and "residual" in self.model
                and self.model["residual"] != self.global_residual):
            raise ValueError("Legacy model.residual conflicts with global_residual; remove the legacy field")
        if self.mask_mode not in {"m0", "none"}:
            raise ValueError("mask_mode must be m0 or none")
        if self.normalization not in {"p1p99", "minmax"}:
            raise ValueError("normalization must be p1p99 or minmax")
        if self.evaluation_protocol not in {"reference_minmax_v1", "legacy"}:
            raise ValueError("evaluation_protocol must be reference_minmax_v1 or legacy")
        if Path(self.mask_name).name != self.mask_name or not self.mask_name:
            raise ValueError("mask_name must be a single filename within each case directory")
        if self.method not in {"SwinIR", "Pix2Pix", "SwinUNet", "UNet"}:
            raise ValueError(f"Unknown method: {self.method}")
        if self.mode not in {"3d", "2.5d"}:
            raise ValueError("mode must be '3d' or '2.5d'")
        if self.use_gan and self.method != "Pix2Pix":
            raise ValueError("Adversarial training is only available for Pix2Pix")
        if self.method == "Pix2Pix" and not self.use_gan:
            raise ValueError("Pix2Pix requires GAN training; use UNet for the convolutional reconstruction baseline")
        if self.adversarial_mask_mode not in {"none", "foreground_fraction"}:
            raise ValueError("adversarial_mask_mode must be none or foreground_fraction")
        if self.lr_policy not in {"plateau", "pix2pix_linear"}:
            raise ValueError("lr_policy must be plateau or pix2pix_linear")
        if type(self.early_stopping_patience) is not int or self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be a nonnegative integer; 0 disables it")
        if self.early_stopping_start not in {"immediate", "lr_decay"}:
            raise ValueError("early_stopping_start must be immediate or lr_decay")
        if self.early_stopping_start == "lr_decay" and self.lr_policy != "pix2pix_linear":
            raise ValueError("early_stopping_start=lr_decay requires pix2pix_linear")
        if self.lr_policy == "pix2pix_linear":
            if not self.use_gan:
                raise ValueError("pix2pix_linear is only available for coordinated Pix2Pix G/D decay")
            if (type(self.lr_constant_epochs) is not int or self.lr_constant_epochs < 1
                    or type(self.lr_decay_epochs) is not int or self.lr_decay_epochs < 1):
                raise ValueError("lr_constant_epochs and lr_decay_epochs must be positive integers")
            if self.epochs > self.lr_constant_epochs + self.lr_decay_epochs:
                raise ValueError("epochs cannot exceed the fixed linear schedule; use a new run name to change the schedule")
        ndim = 3 if self.mode == "3d" else 2
        if len(self.patch_size) != ndim or min(self.patch_size) < 8:
            raise ValueError("patch_size has the wrong dimension or a size below 8")
        if self.augment and self.patch_size[-1] != self.patch_size[-2]:
            raise ValueError("90-degree rotation augmentation requires H=W")
        if min(self.epochs, self.steps_per_epoch, self.batch_size, self.inference_batch_size) < 1:
            raise ValueError("epochs, step counts, and batch sizes must be positive")
        if self.num_workers < 0 or self.cpu_threads < 1 or min(self.preview_every, self.log_every) < 0:
            raise ValueError("Worker counts and logging intervals must be nonnegative; cpu_threads must be positive")
        if min(self.lr, self.discriminator_lr, self.grad_clip) <= 0:
            raise ValueError("Learning rates and grad_clip must be positive")
        if not 0 <= self.inference_overlap < 1:
            raise ValueError("inference_overlap must be in [0,1)")
        if self.inference_patch_size is not None and (
            len(self.inference_patch_size) != 3 or min(self.inference_patch_size) < 8
        ):
            raise ValueError("3D inference patches require three dimensions of at least 8")
        if self.ssim_window < 1 or self.ssim_window % 2 != 1:
            raise ValueError("The SSIM window must be a positive odd integer")
        if self.ssim_sigma <= 0 or self.ssim_data_range <= 0:
            raise ValueError("SSIM sigma and data_range must be positive")
        if min(self.l1_weight, self.ssim_weight) < 0 or self.l1_weight + self.ssim_weight <= 0:
            raise ValueError("Invalid reconstruction loss weights")
        if self.use_gan and self.adversarial_weight <= 0:
            raise ValueError("adversarial_weight must be positive when GAN training is enabled")
        if not self.run_name or self.run_name in {".", ".."} or Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must be a single directory name")
        return self

    def to_dict(self):
        def convert(v):
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: convert(x) for k, x in v.items()}
            if isinstance(v, (tuple, list)):
                return [convert(x) for x in v]
            return v
        return convert(asdict(self))

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values.setdefault("mask_mode", "none")
        values.setdefault("global_residual", None)
        values.setdefault("zero_init_output", None)
        values.setdefault("normalization", "minmax")
        values.setdefault("evaluation_protocol", "legacy")
        values.setdefault("lr_policy", "plateau")
        values.setdefault("early_stopping_start", "immediate")
        values.setdefault("adversarial_mask_mode", "none")
        return cls(**values)
