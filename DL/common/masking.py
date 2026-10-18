"""Validate binary foreground masks on the source image grid."""
from pathlib import Path
import nibabel as nib
import numpy as np


def load_foreground_mask(path, reference):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing foreground mask: {path}; run Data/prepare_m0_masks.py first")
    image = nib.load(str(path))
    values = image.get_fdata(dtype=np.float32)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != reference.shape[:3] or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
        raise ValueError(f"Foreground mask does not match the input grid: {path}")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all() or not values.any():
        raise ValueError(f"Expected a nonempty, finite, binary foreground mask: {path}")
    return values.astype(bool)


def metric_protocol(cfg):
    region = "m0_foreground_v1" if cfg.mask_mode == "m0" else "finite_FOV"
    if cfg.evaluation_protocol == "reference_minmax_v1":
        return (f"reference_minmax_v1; reference_all_finite; prediction_clip_0_1_for_ssim_psnr; {region}; "
                "slice2d_masked_gaussian_ssim_window7_sigma1.5_range1; valid_voxel_average; "
                "psnr_volume_mse_floor1e-12; raw_errors_unclipped; degenerate_range_scale1; case_macro_average")
    return f"source_{cfg.normalization}_v1; {region}; slice2d_masked_gaussian_ssim; case_macro_average"
