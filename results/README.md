# Recorded validation results

`validation_summary.csv` contains the recorded `wtm0mask_ori` validation results: four networks, two input dimensions, and two global output-residual settings, plus BM4D, DT-CWT-NLM, and the common low-SNR input baseline. All rows use the same 54 validation cases and `legacy` source-minmax evaluation. Blank residual/epoch fields indicate that the field does not apply.

Values are taken from the existing completed evaluation reports, not from a new training run. All 16 deep learning variants are retained, including the poor PSNR/RMSE of the direct-output U-Net runs. These values agree with the corresponding best-epoch validation logs; no variant was removed based on its score.

SSIM and PSNR are higher-is-better. MAE, RMSE, and NMSE are lower-is-better. The `_raw` metrics are computed from original PWI intensities: MAE/RMSE retain those units, while NMSE is dimensionless. None is a CBF measure. Standard deviations are between cases, not confidence intervals or repeated-training variability. Classical methods process 3D volumes; they are not relabeled as pseudo-3D models.

The cohort also served checkpoint selection, and the all-repeat reference includes the input repetitions. Interpret these as descriptive validation results. No subject identifiers, per-case reports, images, or model weights are included.
