"""Build the abstract using the supplied DOCX's paragraph and page formatting."""

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


METRICS = (("ssim", 4, max), ("psnr", 2, max), ("rmse_raw", 2, min))
METHOD_NAMES = {"UNet": "U-Net", "SwinUNet": "Swin U-Net", "DTCWT_NLM": "DT-CWT-NLM"}


def text_value(value):
    return value.replace("+/-", "±").replace(" x ", " × ")


def copy_properties(target, source, tag):
    existing = target.find(qn(tag))
    if existing is not None:
        target.remove(existing)
    prop = source.find(qn(tag))
    if prop is not None:
        target.insert(0, deepcopy(prop))


def paragraph(doc, template, index, text="", label=None):
    sample = template.paragraphs[index]
    p = doc.add_paragraph()
    copy_properties(p._p, sample._p, "w:pPr")
    sample_run = next((r for r in sample.runs if r.text), None)
    for value, bold in ((label + ": ", True), (text_value(text), False)) if label else ((text_value(text), False),):
        r = p.add_run(value)
        if sample_run is not None:
            copy_properties(r._r, sample_run._r, "w:rPr")
        else:
            r.font.name = "Arial"
            r.font.size = Pt(6.5)
        if label:
            r.bold = bold
    return p


def border(parent, side, value, size=8):
    element = OxmlElement("w:" + side)
    element.set(qn("w:val"), value)
    element.set(qn("w:sz"), str(size))
    element.set(qn("w:color"), "000000")
    parent.append(element)


def add_table(doc, template, rows):
    paragraph(doc, template, 36, "Table 1. Validation results for all input and output configurations.").paragraph_format.page_break_before = True
    paragraph(doc, template, 36,
              "Mean ± sample standard deviation across 54 cases. Bold values mark the best overall mean "
              "for each metric (higher SSIM/PSNR and lower RMSE). RMSE is in original PWI intensity units.")
    table = doc.add_table(rows=1, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    usable = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    fractions = (0.195, 0.08, 0.115, 0.225, 0.205, 0.18)
    for column, fraction in zip(table.columns, fractions):
        column.width = int(usable * fraction)
    props = table._tbl.tblPr
    table_borders = OxmlElement("w:tblBorders")
    for side in ("top", "bottom", "left", "right", "insideH", "insideV"):
        border(table_borders, side, "single" if side in ("top", "bottom") else "nil", 10)
    props.append(table_borders)
    for cell, text in zip(table.rows[0].cells, ("Method", "Input", "Output", "SSIM ↑", "PSNR (dB) ↑", "RMSE ↓")):
        cell.text = text
        borders = OxmlElement("w:tcBorders")
        border(borders, "bottom", "single", 6)
        cell._tc.get_or_add_tcPr().append(borders)
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    best = {name: choose(float(r[name + "_mean"]) for r in rows) for name, _, choose in METRICS}
    method_order = {name: i for i, name in enumerate(("Input", "BM4D", "DTCWT_NLM", "UNet", "Pix2Pix", "SwinUNet", "SwinIR"))}
    rows = sorted(rows, key=lambda r: (
        0 if r["method"] in ("Input", "BM4D", "DTCWT_NLM") else 1 if r["mode"] == "2.5d" else 2,
        method_order[r["method"]], r["global_residual"]))
    for row in rows:
        mode = "—" if row["method"] == "Input" else "2.5D" if row["mode"] == "2.5d" else "3D"
        output = {"False": "Direct", "True": "Residual", "": "—"}[row["global_residual"]]
        values = [METHOD_NAMES.get(row["method"], row["method"]), mode, output]
        values += [f"{float(row[name + '_mean']):.{precision}f} ± {float(row[name + '_std']):.{precision}f}"
                   for name, precision, _ in METRICS]
        cells = table.add_row().cells
        for i, (cell, value) in enumerate(zip(cells, values)):
            cell.text = value
            if i >= 3:
                metric = METRICS[i - 3][0]
                cell.paragraphs[0].runs[0].bold = float(row[metric + "_mean"]) == best[metric]
    for i, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, cell in enumerate(row.cells):
            cell.width = int(usable * fractions[j])
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_with_next = True
                for r in p.runs:
                    r.font.name = "Arial"
                    r.font.size = Pt(6.5)
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    if i == 0:
                        r.bold = True
    paragraph(doc, template, 36,
              "All methods use the same validation cases, foreground mask and source-minmax evaluation. "
              "Standard deviations describe between-case variation. The reference contains the four input repetitions.")


def build(args):
    content = json.loads(args.content.read_text(encoding="utf-8"))
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 19 or any(int(row["n_cases"]) != 54 or row["normalization"] != "minmax"
                              or row["evaluation_protocol"] != "legacy" for row in rows):
        raise ValueError("Expected the final 19-row, 54-case minmax + legacy summary.")
    figures = json.loads(args.figures_manifest.read_text(encoding="utf-8"))
    template = Document(args.template)
    for index, expected in ((0, "Title:"), (4, "Synopsis"), (11, "Introduction"),
                            (14, "Methods"), (21, "Results"), (25, "Discussion and Conclusion"),
                            (28, "Reference"), (34, "Figures")):
        if len(template.paragraphs) <= index or template.paragraphs[index].text.strip() != expected:
            raise ValueError("Use the supplied reference DOCX; its paragraph formatting is required.")
    doc = Document()
    # Copy only formatting; no template authors, images or study text are retained.
    for tag in ("w:docDefaults",):
        copy_properties(doc.styles.element, template.styles.element, tag)
    for tag in ("w:pPr", "w:rPr"):
        copy_properties(doc.styles["Normal"].element, template.styles["Normal"].element, tag)
    target_section = doc.sections[0]._sectPr
    for child in list(target_section):
        target_section.remove(child)
    for child in template.sections[0]._sectPr:
        if child.tag not in (qn("w:headerReference"), qn("w:footerReference")):
            target_section.append(deepcopy(child))
    core = doc.core_properties
    core.title = content["title"]
    core.subject = "ASL PWI denoising benchmark: validation-set abstract draft"
    core.author = core.last_modified_by = ""
    core.comments = "Formatting follows the supplied abstract template."
    core.created = core.modified = datetime(2026, 10, 18, tzinfo=timezone.utc)
    core.revision = 2

    paragraph(doc, template, 0, "Title:")
    paragraph(doc, template, 1, content["title"])
    paragraph(doc, template, 2, "Authors:")
    paragraph(doc, template, 3, content["authors"])
    paragraph(doc, template, 4, "Synopsis")
    for label, text in content["synopsis"]:
        paragraph(doc, template, 5, f"{label}: {text}")
    paragraph(doc, template, 9, "Impact: " + content["impact"])
    paragraph(doc, template, 10)
    paragraph(doc, template, 11, "Introduction")
    for text in content["introduction"]:
        paragraph(doc, template, 12, text)
    paragraph(doc, template, 13)
    paragraph(doc, template, 14, "Methods")
    for label, text in content["methods"]:
        paragraph(doc, template, 16, text, label)
    paragraph(doc, template, 21, "Results").paragraph_format.page_break_before = True
    for text in content["results"]:
        paragraph(doc, template, 22, text)
    paragraph(doc, template, 25, "Discussion and Conclusion")
    for text in content["discussion"]:
        paragraph(doc, template, 26, text)
    paragraph(doc, template, 27)
    paragraph(doc, template, 28, "Reference")
    for i, text in enumerate(content["references"], 1):
        paragraph(doc, template, 29, f"{i}. {text}")

    add_table(doc, template, rows)
    # Large comparison plates use landscape pages; text retains the template geometry.
    figure_section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    figure_section.orientation = WD_ORIENT.LANDSCAPE
    figure_section.page_width, figure_section.page_height = (
        doc.sections[0].page_height, doc.sections[0].page_width)
    for i, figure in enumerate(figures):
        if i == 0:
            paragraph(doc, template, 34, "Figures")
        p = paragraph(doc, template, 35)
        if i:
            p.paragraph_format.page_break_before = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        usable = figure_section.page_width - figure_section.left_margin - figure_section.right_margin
        p.add_run().add_picture(str(args.figures_dir / figure["file"]), width=usable)
        paragraph(doc, template, 36, figure["caption"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)
    paragraphs = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        paragraphs += ["\t".join(c.text for c in row.cells) for row in table.rows]
    args.output.with_suffix(".txt").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {len(doc.inline_shapes)} figures, {len(doc.tables)} table")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--content", type=Path, default=Path(__file__).with_name("manuscript_content.json"))
    parser.add_argument("--figures-manifest", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
