# Manuscript and figure generation

These scripts read the completed `wtm0mask_ori` results. They do not train models, change predictions, or recalculate the reported evaluation metrics.

Run from the repository root:

```bash
python publication/quantitative_comparison.py \
  --project-root . --data-root Data/Dataset001 \
  --output-dir publication/generated

python publication/qualitative_comparison.py \
  --dataset-root Data/Dataset001 \
  --output-dir publication/generated

python publication/validation_table.py \
  --summary results/validation_summary.csv \
  --output-dir publication/generated
```

The plotting scripts require NumPy, Matplotlib and NiBabel. The quantitative script also loads the existing `DL/plot_comparison.py` to preserve its case/protocol checks and plotting style. Separate 2.5D and 3D plots are saved in addition to the composites.

| Manuscript item | Output stem | Generator |
| --- | --- | --- |
| Figure 1: direct-output distributions | `quantitative_gres0` | `quantitative_comparison.py` |
| Figure 2: residual-output distributions | `quantitative_gres1` | `quantitative_comparison.py` |
| Figure 3: lower/higher input SSIM, 3D residual models | `qualitative_low_high` | `qualitative_comparison.py` |
| Table 1: all recorded configurations | `validation_table` | `validation_table.py` |

Each figure/table is saved as PNG and PDF. The editable Word table is built independently from the same summary CSV, with the same best-value rule. Best overall means are bold; no significance tests or significance markers are used.

## Qualitative selection and display

Two cases are selected using input SSIM nearest the 25th and 75th validation-set percentiles. The slice is selected by cumulative foreground voxel mass. Model performance is not used to select cases or slices. Figure 3 shows the explicitly labeled 3D residual subset; all 16 DL variants remain in the quantitative plots and table.

Within each case, all methods share the reference-foreground P1–P99 display window and the P95 input-to-reference absolute-error upper limit. These are display settings only: training normalization and evaluation remain `minmax + legacy`, and the stored predictions are unchanged. Error maps use the common evaluation mask. Native voxels are displayed with nearest-neighbor rendering and no spatial resampling.

`qualitative_selection.json` records the exact cases, slices, display ranges and source checksums. It contains private identifiers and stays beside the local figures. It is not included in the upload package. The figure uses anonymous Lower SSIM / Higher SSIM labels. Internal run identifiers are not displayed in the manuscript or figures.

## Word document

Install `python-docx>=1.2,<2` in addition to the plotting dependencies. Edit `manuscript_content.json` for the text and `figure_manifest.json` for the selected images/captions, then run:

```bash
python publication/build_manuscript.py \
  --template /path/to/mmasl_ISMRM2025_reference.docx \
  --summary results/validation_summary.csv \
  --figures-manifest publication/figure_manifest.json \
  --figures-dir publication/generated \
  --output publication/generated/ASL_Denoising_Abstract.docx
```

The builder copies the supplied reference's paragraph formatting and portrait page geometry: Arial, 6.5 pt body text and 12 pt section labels. It preserves the section framework, including the singular `Reference` heading. Comparison figures use separate landscape pages for readability. The table is editable and uses three black horizontal rules. Template study text, authors, images and metadata are not copied.

The DOCX can be exported to PDF in Word or LibreOffice. The original reference DOCX is not distributed. The content remains a draft with author/affiliation and OS cohort details to be confirmed. Generated images and the private selection manifest are not included in this repository.
