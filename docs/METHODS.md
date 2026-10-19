# Methods and references

These are task-adapted baselines, not exact reproductions of the cited networks. All methods use the same pairs of four-repeat inputs and 18-repeat references, foreground region, and evaluation definitions. All networks share the same PWI denoising task, data split, and reconstruction loss. Model sizes differ, and Pix2Pix additionally includes an adversarial loss.

## Models

| Model | Adaptation used here |
| --- | --- |
| U-Net | Four encoder scales with 32, 64, 128, and 256 channels, three decoder scales, GroupNorm, and a linear single-channel regression head. |
| Pix2Pix | A compact U-Net generator with a conditional PatchGAN discriminator; GroupNorm, decoder dropout, signed PWI regression, and foreground-weighted adversarial BCE. |
| SwinIR | 60 features, six residual groups with six Swin blocks each, alternating window and shifted-window attention, and a single-channel output head. |
| SwinUNet | Adapted from the HUST architecture described by Cheema et al.: three encoder scales with 48, 96, and 192 channels, two decoder scales, patch merging, Swin blocks, and fused transposed-convolution and interpolation upsampling. |

Pseudo-3D models take three adjacent PWI slices and predict the center slice. This choice follows Shou et al.'s reported balance between denoising performance and CBF bias. Their M0 experiment improved image similarity but increased white-matter CBF bias, motivating omission of M0 as an additional image channel here. Neither input choice was separately optimized on this cohort.

Volumetric versions use 3D operations, including anisotropic downsampling where applicable. Training crops are 48 x 48 for pseudo-3D and 16 x 48 x 48 (depth, height, width) for 3D. Batch sizes are 16 and 1, respectively. Models retain the native voxel grid; internal padding is cropped from the output.

All models support direct prediction and residual prediction, in which the input volume or center slice is added to the network output. The final output-convolution weights and biases are initialized to zero. On the normalized training scale, the initial direct prediction is therefore zero, while the initial residual prediction equals the input. The residual addition introduces no trainable parameters. Parameter counts are not matched across architectures.

## Training

Training uses Adam, an initial learning rate of 2e-4, seed 42, 250 updates per epoch, and at most 200 epochs. All models share foreground L1 + (1 - SSIM), with equal weights. Paired in-plane flips and quarter-turn rotations augment the training crops. Input and reference share source-derived min-max scaling; values are not clipped.

Pix2Pix adds an adversarial term weighted by 0.01. Generator and discriminator learning rates are held for 100 epochs and then decay together over 100 epochs. Its early-stop counter starts in the decay phase. Other models use a validation-loss plateau scheduler. Early-stop patience is 30; the best checkpoint is selected using mean validation reconstruction loss throughout training.

## Evaluation

The reported results use input-derived min–max scaling and a shared evaluation protocol. A fixed M0-derived foreground mask is intersected with finite source and reference voxels. The mask may include non-brain tissue and should not be described as a brain segmentation.

The source image's full finite field of view determines a common offset and scale. SSIM is computed on axial slices with a 7 x 7 Gaussian window (sigma 1.5, data range 1), using masked local statistics and valid-voxel aggregation. PSNR uses the volume-wide masked mean squared error on the same scale. Predictions are not independently normalized or clipped. MAE and RMSE are reported in original PWI intensity units; NMSE is the sum of squared errors divided by the sum of squared reference intensities within the evaluation mask. Cases receive equal weight in the reported mean and sample standard deviation.

Classical predictions use the same evaluation functions as the networks. Their internal denoising scales remain method-specific. BM4D results correspond to the local CUDA implementation; the optional official CPU backend is not numerically interchangeable with it.

The validation cohort is also used for checkpoint selection. The 18-repeat reference contains the four repetitions used in the input, so input and reference noise are correlated. Scores describe this validation setting; they do not establish independent generalization, CBF accuracy, or clinical effectiveness. No statistical significance or universally optimal architecture is claimed.

## References

1. Lindner T, Bolar D S, Achten E, et al. Current state and guidance on arterial spin labeling perfusion MRI in clinical neuroimaging[J]. Magnetic resonance in medicine, 2023, 89(5): 2024-2047.
2. Shou Q, Zhao C, Shao X, et al. Transformer‐based deep learning denoising of single and multi‐delay 3D arterial spin labeling[J]. Magnetic resonance in medicine, 2024, 91(2): 803-818.
3. Cheema M N, Zhang L, Nazir A, et al. Transformer-based arterial spin labeling perfusion MRI denoising[J]. The Visual Computer, 2025, 41(13): 10663-10673.
4. Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical image computing and computer-assisted intervention. Cham: Springer international publishing, 2015: 234-241.
5. Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[C]//2017 IEEE conference on computer vision and pattern recognition (CVPR). Ieee, 2017: 5967-5976.
6. Liang J, Cao J, Sun G, et al. Swinir: Image restoration using swin transformer[C]//2021 IEEE/CVF international conference on computer vision workshops (ICCVW). IEEE, 2021: 1833-1844.
7. Maggioni M, Katkovnik V, Egiazarian K, et al. Nonlocal transform-domain filter for volumetric data denoising and reconstruction[J]. IEEE transactions on image processing, 2012, 22(1): 119-133.
8. Liang X, Connelly A, Calamante F. Voxel-wise functional connectomics using arterial spin labeling functional magnetic resonance imaging: the role of denoising[J]. Brain connectivity, 2015, 5(9): 543-553.
