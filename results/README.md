# Recorded validation results

`validation_summary.csv` contains validation results for four networks in 2.5D and 3D, with direct and residual prediction, plus BM4D, DT-CWT-NLM, and the low-SNR input baseline. All rows use the same 54 validation cases, input-derived scaling, and foreground mask. Blank residual/epoch fields indicate that the field does not apply.

The table includes all 16 deep learning configurations. Scores come from evaluation of the selected checkpoint for each configuration.

SSIM and PSNR are higher-is-better. MAE, RMSE, and NMSE are lower-is-better. The `_raw` metrics are computed from original PWI intensities: MAE/RMSE retain those units, while NMSE is dimensionless. None is a CBF measure. Standard deviations are between cases, not confidence intervals or repeated-training variability. Classical methods process 3D volumes; they are not relabeled as pseudo-3D models.

The cohort also served checkpoint selection, and the all-repeat reference includes the input repetitions. Interpret these as descriptive validation results. No subject identifiers, per-case reports, images, or model weights are included.
