# ASL PWI Denoising Benchmark

A practical comparison of established image-restoration models for low-SNR arterial spin labeling (ASL) perfusion-weighted images (PWI). The aim is to help researchers choose a denoising baseline for ASL, using a shared data and evaluation pipeline.

The deep learning baselines are **U-Net, Pix2Pix, SwinIR, and SwinUNet**, adapted to three-slice pseudo-3D (2.5D) and volumetric 3D inputs. **BM4D** and **DT-CWT-NLM** provide classical 3D baselines. The models map a four-repeat PWI average to a higher-SNR, 18-repeat PWI reference. Inputs and targets are PWI; M0 is used only to derive a foreground mask.

## Data

The study contains **271 cases: 217 training and 54 validation cases**. The cohort labels and total counts are:

| Cohort | Cases |
| --- | ---: |
| Healthy controls | 67 |
| Stroke | 103 |
| Obstructive sleep apnea (OSA) | 59 |
| COPD | 14 |
| OS | 28 |

All acquisitions use a **GRASE readout**. Post-labeling delays (PLDs) are **1500 ms in 79 cases, 1800 ms in 124 cases, and 2000 ms in 68 cases**. Preprocessing includes motion correction, and the included data were screened to exclude substantial head motion. PWI volumes have a 64 x 64 in-plane matrix and approximately 3.5 x 3.5 x 6 mm voxels.

Imaging data and trained weights are not included. Place prepared cases under:

```text
Data/Dataset001/
  train/<case>/
  val/<case>/
    rpwi_ave_4.nii
    rpwi_ave_max.nii
    m0_foreground_mask.nii
    m0.nii                     # only needed to generate the mask
```

## Quick start

Use Python 3.9-3.12 and a PyTorch installation suitable for your hardware. The recorded experiments used Python 3.9 and PyTorch 2.5.1.

```bash
pip install -r requirements.txt
python Data/prepare_m0_masks.py   # if masks have not been prepared
python DL/run_all.py
```

The main configuration files are:

| File | Settings |
| --- | --- |
| `DL/run_all.py` | Models, dimensions, direct and residual prediction, training options, and device |
| `DL/common/config.py` | Default dataset location and image filenames |
| `Data/prepare_m0_masks.py` | Dataset location and mask preparation |
| `DL/compare_results.py`, `DL/plot_comparison.py` | Result locations and runs to compare |

Adjust paths if your data are stored elsewhere. Existing training outputs are protected by default.

For BM4D and the MATLAB DT-CWT-NLM implementation, see [Tradition/README.md](Tradition/README.md). DT-CWT-NLM requires an external ONLM MEX dependency.

After prediction, evaluate and compare:

```bash
python Tradition/evaluate_results.py
python DL/compare_results.py
python DL/plot_comparison.py
```

Select the same run name in the two comparison scripts. Classical results retain their actual 3D processing dimension.

## Evaluation and results

Methods share the same input-derived scaling and foreground mask for evaluation. We report **SSIM, PSNR, MAE, RMSE, and NMSE**, averaged across validation cases. MAE and RMSE retain original PWI intensity units; NMSE is dimensionless.

[Aggregate validation results](results/validation_summary.csv) include the direct and residual model variants and classical baselines. [Publication scripts](publication/README.md) generate comparison plots, PWI and error montages, and the results table. These are validation-set results, and the 18-repeat reference includes the four input repetitions.

See [Methods and references](docs/METHODS.md) for architecture adaptations and the evaluation definition. Run `python DL/verify.py` for synthetic checks. Third-party source attribution and dependency terms are listed in [Tradition/THIRD_PARTY_NOTICES.md](Tradition/THIRD_PARTY_NOTICES.md).
