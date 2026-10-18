"""Inspect paired data, geometry, foreground masks, and intensity ranges."""
from pathlib import Path
import numpy as np

from common.config import ExperimentConfig
from common.data import discover_splits, load_case
from common.utils import save_preview, write_csv, write_json

CONFIG = ExperimentConfig()
CHECK_ALL_INTENSITIES = True


def main():
    train, val, report = discover_splits(CONFIG)
    folder = Path(CONFIG.output_root) / "_data_check"
    write_json(folder / "data_audit.json", report)
    rows = []
    for split, cases in (("train", train), ("val", val)):
        print(f"{split}: {len(cases)} pairs; skipped {len(report[split]['skipped'])} incomplete directories", flush=True)
        for index, case in enumerate(cases if CHECK_ALL_INTENSITIES else cases[:1]):
            item = load_case(case, CONFIG.normalization)
            finite = item["input"][item["source_valid"]]
            rows.append({"split": split, "case": case.name, "shape_dhw": str(item["input"].shape),
                         "invalid_source_voxels": int((~item["source_valid"]).sum()),
                         "valid_pair_fraction": float(item["valid"].mean()),
                         "normalization": CONFIG.normalization,
                         "offset": item["offset"], "scale": item["scale"],
                         "normalization_upper": item["normalization"]["upper"],
                         "normalized_source_std": float(np.std(finite))})
            if index == 0:
                save_preview(item, None, folder / f"{split}_example.png")
            if (index + 1) % 50 == 0:
                print(f"  Intensity audit {index + 1}/{len(cases)}", flush=True)
    write_csv(folder / "intensity_audit.csv", rows)
    print(f"Audit complete: {folder}")


if __name__ == "__main__":
    main()
