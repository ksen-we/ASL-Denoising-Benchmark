# Methods and references

These are task-adapted baselines, not exact reproductions of the cited networks. The comparison keeps input/reference pairs, data split, foreground region, reconstruction objective, and evaluation definitions consistent. Model capacities and the Pix2Pix adversarial objective differ.

## Models

| Model | Adaptation used here |
| --- | --- |
| U-Net | Four encoder scales (32/64/128/256 channels), three decoder scales, GroupNorm, and a linear single-channel regression head. |
| Pix2Pix | A compact U-Net generator with a conditional PatchGAN discriminator; GroupNorm, decoder dropout, signed PWI regression, and foreground-weighted adversarial BCE. |
| SwinIR | 60 features, six residual groups with six Swin blocks each, alternating window/shifted-window attention, and a single-channel output head. |
| HUST-inspired Swin U-Net | Three encoder scales (48/96/192 channels), two decoder scales, patch merging, Swin blocks, and fused transposed-convolution/interpolation upsampling. |

Pseudo-3D models take three adjacent PWI slices and predict the center slice. The three-slice choice follows the denoising/CBF-bias compromise discussed by Shou et al.; it is not a claim that three slices maximize every similarity metric. Their M0 experiment improved image similarity but increased white-matter CBF bias, motivating omission of M0 as an additional image channel here. This choice was not separately optimized on this cohort.

Volumetric versions use 3D operations, including anisotropic downsampling where applicable. Training crops are 48 x 48 for pseudo-3D and 16 x 48 x 48 (depth, height, width) for 3D. Batch sizes are 16 and 1, respectively. Models retain the native voxel grid; internal padding is cropped from the output.

All models support direct prediction (`gres0`) and a final addition of the input volume or center slice (`gres1`). Output heads are zero-initialized. The residual addition introduces no trainable parameters. Parameter counts are not matched across architectures.

## Training

Training uses Adam, an initial learning rate of 2e-4, seed 42, 250 updates per epoch, and at most 200 epochs. All models share foreground L1 + (1 - SSIM), with equal weights. Paired in-plane flips and quarter-turn rotations augment the training crops. Input and reference share source-derived min-max scaling; values are not clipped.

Pix2Pix adds an adversarial term weighted by 0.01. Generator and discriminator learning rates are held for 100 epochs and then decay together over 100 epochs. Its early-stop counter starts in the decay phase. Other models use a validation-loss plateau scheduler. Early-stop patience is 30; the best checkpoint is selected using mean validation reconstruction loss throughout training.

## Evaluation

The final `wtm0mask_ori` results use `normalization="minmax"` and `evaluation_protocol="legacy"`. A fixed M0-derived foreground mask is intersected with finite source and reference voxels. The mask may include non-brain tissue and should not be described as a brain segmentation.

The source image's full finite field of view determines a common offset and scale. SSIM is computed on axial slices with a 7 x 7 Gaussian window (sigma 1.5, data range 1), using masked local statistics and valid-voxel aggregation. PSNR uses the volume-wide masked mean squared error on the same scale. Predictions are not independently normalized or clipped. MAE and RMSE are reported in original PWI intensity units; NMSE is the raw mean squared error divided by the reference signal energy and is dimensionless. Cases receive equal weight in the reported mean and sample standard deviation.

Classical predictions use the same evaluation functions as the networks. Their internal denoising scales remain method-specific. BM4D results correspond to the local CUDA implementation; the optional official CPU backend is not numerically interchangeable with it.

The validation cohort is also used for checkpoint selection. The higher-SNR reference contains the four repetitions used in the input, so input and reference noise are correlated. Scores describe this validation setting; they do not establish independent generalization, CBF accuracy, or clinical effectiveness. No statistical significance or universally optimal architecture is claimed.

## References

- Lindner T, et al. Current state and guidance on arterial spin labeling perfusion MRI in clinical neuroimaging. *Magn Reson Med*. 2023;89:2024-2047. [doi:10.1002/mrm.29572](https://doi.org/10.1002/mrm.29572).
- Shou Q, et al. Transformer-based deep learning denoising of single and multi-delay 3D arterial spin labeling. *Magn Reson Med*. 2024;91:803-818. [doi:10.1002/mrm.29887](https://doi.org/10.1002/mrm.29887).
- Liang J, et al. SwinIR: Image Restoration Using Swin Transformer. *ICCV Workshops*. 2021. [Paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html).
- Cheema MN, et al. Transformer-based arterial spin labeling perfusion MRI denoising. *The Visual Computer*. 2025;41:10663-10673. [doi:10.1007/s00371-025-04061-x](https://doi.org/10.1007/s00371-025-04061-x).
- Ronneberger O, et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. 2015. [doi:10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).
- Isola P, et al. Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*. 2017. [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html).
- Maggioni M, et al. Nonlocal transform-domain filter for volumetric data denoising and reconstruction. *IEEE Trans Image Process*. 2013;22:119-133. [doi:10.1109/TIP.2012.2210725](https://doi.org/10.1109/TIP.2012.2210725).
- Liang X, Connelly A, Calamante F. Voxel-Wise Functional Connectomics Using Arterial Spin Labeling Functional Magnetic Resonance Imaging: The Role of Denoising. *Brain Connect*. 2015;5:543-553. [doi:10.1089/brain.2014.0290](https://doi.org/10.1089/brain.2014.0290).
