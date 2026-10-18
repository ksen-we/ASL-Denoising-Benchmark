# ASL PWI Denoising Benchmark

A practical comparison of established image-restoration models for low-SNR arterial spin labeling (ASL) perfusion-weighted images (PWI). The aim is to help researchers choose a denoising baseline for ASL, using a shared data and evaluation pipeline.

The deep learning baselines are **U-Net, Pix2Pix, SwinIR, and a HUST-inspired Swin U-Net**, each adapted to three-slice pseudo-3D (2.5D) and volumetric 3D inputs. **BM4D** and **DT-CWT-NLM** provide classical 3D baselines. The networks map a four-repeat PWI average to a higher-SNR, all-repeat PWI reference; CBF maps are not network inputs or targets. M0 is used to derive a foreground mask, not as an additional network channel.

## Data

The study contains **271 cases: 217 training and 54 validation cases**. The cohort labels and total counts are:

| Cohort | Cases |
| --- | ---: |
| Control groups (CG and Control) | 67 |
| Stroke | 103 |
| Obstructive sleep apnea (OSA) | 59 |
| Chronic obstructive pulmonary disease (COPD) | 14 |
| OS cohort (original dataset label) | 28 |

All acquisitions use a **GRASE readout**. The default post-labeling delay (PLD) is **1800 ms**; explicit `_1500` or `_2000` suffixes indicate a different PLD. Preprocessing includes motion correction, and the included data were screened to exclude substantial head motion. PWI volumes have a 64 x 64 in-plane matrix and approximately 3.5 x 3.5 x 6 mm voxels.

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

Edit the options at the top of `DL/run_all.py` to select models, dimensions, output residuals, and device. The release selects both direct prediction (`gres0`) and input-residual prediction (`gres1`). Existing training outputs are protected by default.

For BM4D and the MATLAB DT-CWT-NLM implementation, see [Tradition/README.md](Tradition/README.md). DT-CWT-NLM requires an external ONLM MEX dependency.

After prediction, evaluate and compare:

```bash
python Tradition/evaluate_results.py
python DL/compare_results.py
python DL/plot_comparison.py
```

Select the same run name in the two comparison scripts. Classical results retain their actual 3D processing dimension.

## Evaluation and results

The release follows the **`wtm0mask_ori`** experiments: source-derived **min-max normalization**, **`legacy` evaluation**, and a shared M0 foreground mask. SSIM and PSNR use the common source scale. MAE and RMSE use original PWI intensity units; NMSE is a dimensionless error relative to reference signal energy. Reported scores are averaged across validation cases. Alternative settings remain available but are not used for the reported comparison.

[Aggregate validation results](results/validation_summary.csv) include every direct/residual model variant and the classical baselines. [Publication scripts](publication/README.md) recreate the comparison boxplots, PWI/error montages, and monochrome results table from saved outputs. These are validation results, not independent test results. The all-repeat reference includes the four input repetitions and is not noise-free.

See [Methods and references](docs/METHODS.md) for architecture adaptations and the evaluation definition. Run `python DL/verify.py` for synthetic checks. Third-party source attribution and dependency terms are listed in [Tradition/THIRD_PARTY_NOTICES.md](Tradition/THIRD_PARTY_NOTICES.md).
