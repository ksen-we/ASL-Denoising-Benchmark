# Manuscript and figure generation

These scripts generate manuscript figures and tables from saved predictions and evaluation reports. They do not train models or recalculate the reported metrics.

Run from the `LowSNR2High` project directory:

```bash
python Pepar/figure_scripts/qualitative_comparison.py \
  --dataset-root Data/Dataset001 \
  --output-dir Pepar/ASL_Denoising_Figures

python Pepar/figure_scripts/validation_table.py \
  --summary GitHub_Release/results/validation_summary.csv \
  --output-dir Pepar/ASL_Denoising_Figures
```

The plotting scripts require NumPy, Matplotlib and NiBabel.

| Manuscript item | Output stem | Generator |
| --- | --- | --- |
| Figure 1: lower and higher input SSIM, 3D residual networks | `qualitative_low_high` | `qualitative_comparison.py` |
| Table 1: input, conventional methods and direct prediction (11 rows) | `validation_table_direct` | `validation_table.py` |
| Table 2: residual prediction (8 rows) | `validation_table_residual` | `validation_table.py` |

Each item is saved as PNG and PDF. Both tables report SSIM, PSNR, RMSE and NMSE, as mean ± SD across the 54 validation cases. The editable Word tables use the same summary CSV. Within each numeric cell, the mean appears above its standard deviation to fit the portrait page. Dimension labels are merged vertically, and black horizontal rules separate groups and methods. The best mean for each metric is bold within each table; standard deviations remain in normal type. Metric definitions are given in [Methods](../../GitHub_Release/docs/METHODS.md).

## Qualitative selection and display

Two cases are selected using input SSIM nearest the 25th and 75th validation-set percentiles. The slice is selected by cumulative foreground voxel count. Model performance is not used to select cases or slices. Figure 1 shows the 3D residual networks alongside the input, conventional methods and reference; all 16 network configurations remain in the two tables.

PSNR and SSIM annotations are whole-volume values read from the final saved evaluation CSVs, not scores computed from the displayed slice. Each annotation contains two numbers only: PSNR followed by SSIM on one line, as defined once in the caption. The values appear in a thin separate band above each PWI panel, leaving the anatomy unobscured. The two case blocks are stacked in a four-column portrait layout with compact method labels. Each case shares one pair of intensity and error color bars; a redundant overall title is omitted. The reference score band contains `PSNR | SSIM` as a shared legend. Reference and error panels do not receive self-comparison scores. The headings state the repetition count explicitly; the caption identifies the inputs and references as averages of four and 18 repetitions.

Within each case, all methods share the reference-foreground P1–P99 display window and the P95 input-to-reference absolute-error upper limit. These settings do not alter predictions or evaluation metrics. Error maps use the common evaluation mask. Native voxels are displayed with nearest-neighbor rendering and no spatial resampling.

`qualitative_selection.json` records the cases, slices, display ranges, exact annotation values, source report fields and paths, and source checksums. It contains private identifiers and stays beside the local figures; it is not included in the upload package. The figure uses anonymous Lower SSIM and Higher SSIM labels.

## Word document

Install `python-docx>=1.2,<2` in addition to the plotting dependencies. Edit `manuscript_content.json` for the text and `figure_manifest.json` for the selected image and caption, then run:

```bash
python Pepar/figure_scripts/build_manuscript.py \
  --template Pepar/Jinhan_SCORD_ISMRM.docx \
  --summary GitHub_Release/results/validation_summary.csv \
  --figures-manifest Pepar/figure_scripts/figure_manifest.json \
  --figures-dir Pepar/ASL_Denoising_Figures \
  --output Pepar/ASL_Denoising_Abstract.docx
```

The manuscript follows the Jinhan reference's typography and page geometry: Times New Roman, 12 pt body text, 14 pt section headings and a 16 pt title. The main text contains Introduction, Methods, Results, Discussion and Conclusion, and References; Synopsis and Impact are not included in this document. All pages, including the two editable tables and the qualitative figure, use A4 portrait geometry with the reference margins. Paragraph alignment, spacing and typography are copied from the corresponding reference paragraphs; affiliations use 10.5 pt justified text. Template study text, images and metadata are not copied. The author affiliations follow the confirmed mapping to the reference document.

The DOCX can be exported to PDF in Word or LibreOffice. The builder also writes `ASL_Denoising_Abstract_ISMRM2027_Fields.txt`, containing the separate Synopsis, Impact, abstract body, references, acknowledgements and captions for ECHO. It checks the 2027 text and caption limits. Code availability is confined to Acknowledgements. Author names, affiliation numbers, affiliation text and the code repository URL are specified in `manuscript_content.json`. Unresolved study details remain documented in `../ASL_Denoising_Review_Notes.md`. The reference document, generated images and private selection manifest are excluded from the public release.

## Optional distribution plots

The quantitative boxplot generator is retained for exploration but is not used in the current manuscript:

```bash
python Pepar/figure_scripts/quantitative_comparison.py \
  --project-root . --data-root Data/Dataset001 \
  --output-dir Pepar/ASL_Denoising_Figures
```

It loads the release copy of `DL/plot_comparison.py` to preserve its case and evaluation checks and plotting style, and saves composite and separate dimensional comparisons for all five metrics.

## ISMRM 2027 submission

The [standard guidelines](https://www.ismrm.org/27m/call/standard/) limit the abstract body to 750 words, Synopsis to 100 words, Impact to 40 words, title to 125 characters and each caption to 500 characters. The review DOCX retains the reference layout; Synopsis and Impact are separate submission fields. References use Vancouver style and bracketed numerical superscripts.

Upload tables as high-resolution images through Add Figure, as described in the [ECHO guide](https://www.ismrm.org/27m/call/submission-guide/). The two tables and the qualitative figure occupy three of the five available positions; verify ECHO's automatic numbering when entering the table and figure citations.

The [current call](https://www.ismrm.org/27m/call/) also requires a separate program preview. Generate it with:

```bash
python Pepar/figure_scripts/preview_figure.py \
  --dataset-root Data/Dataset001 \
  --output-dir Pepar/ASL_Denoising_Figures \
  --selection-manifest Pepar/ASL_Denoising_Figures/qualitative_selection.json
```

`ismrm2027_preview.png` is 1200 × 800 pixels (3:2), below 1 MB, with large labels and no caption or metric annotations. It uses the same two cases, slices and display windows as Figure 1. The middle column shows the 3D residual Swin U-Net. The PDF is a viewing companion. This program preview is separate from the three review figures and tables.
