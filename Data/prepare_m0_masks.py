"""Generate fixed foreground masks from M0 images."""
from pathlib import Path
import sys
import hashlib
import json

import nibabel as nib
import numpy as np
from scipy.ndimage import label, binary_fill_holes

DATASET_ROOT = Path(__file__).resolve().parent / "Dataset001"
SPLITS = ("train", "val")
MASK_NAME = "m0_foreground_mask.nii"
ALPHA = 0.10                  
PERCENTILE = 99.5             
OVERWRITE = False             

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DL"))
from common.utils import file_hash, write_json, write_csv, pyplot


def make_mask(volume, alpha=ALPHA, percentile=PERCENTILE):
    if volume.ndim != 3 or not 0 < alpha < 1 or not 0 < percentile <= 100:
        raise ValueError("A 3D M0 volume and valid threshold/percentile are required")
    finite = np.isfinite(volume)
    positive = volume[finite & (volume > 0)]
    if not len(positive):
        raise ValueError("M0 has no positive finite voxels")
    scale = float(np.percentile(positive, percentile))

    mask = (volume > alpha * scale) & finite
    components, count = label(mask, np.ones((3, 3, 3), dtype=bool))
    if count == 0:
        raise ValueError("The thresholded mask is empty")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    mask = components == sizes.argmax()  
    for z in range(mask.shape[2]):
        mask[:, :, z] = binary_fill_holes(mask[:, :, z])  
    mask &= finite
    return mask, scale


def prepare_case(folder):
    m0_path = folder / "m0.nii"
    image = nib.load(str(m0_path))
    m0 = image.get_fdata(dtype=np.float32)
    if m0.ndim == 4 and m0.shape[-1] == 1:
        m0 = m0[..., 0]
    mask, scale = make_mask(m0)

    source = folder / "rpwi_ave_4.nii"
    reference = nib.load(str(source)) if source.exists() else image
    if reference.shape[:3] != m0.shape or not np.allclose(reference.affine, image.affine, rtol=0, atol=1e-4):
        raise ValueError("M0 and PWI grids differ; align them before generating masks")
    policy = dict(version="m0_foreground_v2_no_smoothing", alpha=ALPHA, positive_percentile=PERCENTILE,
                  smoothing=False, largest_component=26, fill_holes="axial_2d",
                  m0_sha256=file_hash(m0_path), reference_affine=reference.affine.tolist(),
                  generator_sha256=file_hash(__file__))
    signature = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    output = folder / MASK_NAME
    sidecar = Path(str(output) + ".json")
    if output.exists() and not OVERWRITE:
        record = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if record.get("signature") != signature or record.get("mask_sha256") != file_hash(output):
            raise ValueError("Existing mask provenance differs; verify it before enabling OVERWRITE")
        status = "verified_existing"
    else:
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        result = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header)
        result.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
        result.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
        result.header.set_slope_inter(1, 0)
        temporary = output.with_name(".tmp_" + output.name)
        nib.save(result, str(temporary))
        temporary.replace(output)
        write_json(sidecar, dict(signature=signature, policy=policy, mask_sha256=file_hash(output),
                                threshold=ALPHA * scale, fraction=float(mask.mean())))
        status = "created"
    return dict(split=folder.parent.name, case=folder.name, status=status,
                fraction=float(mask.mean()), threshold=ALPHA * scale, message=""), m0, mask


def preview(m0, mask, name, path):
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    scale = np.percentile(m0[np.isfinite(m0) & (m0 > 0)], 99.5)
    for ax, fraction in zip(axes, (0.1, 0.5, 0.9)):
        z = int(round((m0.shape[2] - 1) * fraction))
        ax.imshow(np.rot90(m0[:, :, z]), cmap="gray", vmin=0, vmax=scale * 0.7)
        if mask[:, :, z].any() and not mask[:, :, z].all():
            ax.contour(np.rot90(mask[:, :, z]), levels=[0.5], colors=["lime"], linewidths=0.8)
        ax.set_title(f"{name} | z={z}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    output = DATASET_ROOT / "DLResult" / "_audit" / "m0_masks"
    rows, shown = [], set()
    for split in SPLITS:
        for folder in sorted(p for p in (DATASET_ROOT / split).iterdir() if p.is_dir()):
            try:
                row, m0, mask = prepare_case(folder)
                group = "Control" if folder.name.startswith("Control") else folder.name.split("_")[0]
                if (split, group) not in shown:
                    preview(m0, mask, folder.name, output / "previews" / f"{split}_{folder.name}.png")
                    shown.add((split, group))
            except Exception as exc:
                row = dict(split=split, case=folder.name, status="error", fraction=None,
                           threshold=None, message=str(exc))
            rows.append(row)
            if len(rows) % 50 == 0 or row["status"] == "error":
                print(f"[{len(rows)}] {split}/{folder.name}: {row['status']} {row['message']}", flush=True)
    write_csv(output / "cases.csv", rows)
    good = [r for r in rows if r["status"] != "error"]
    write_json(output / "summary.json", dict(n_cases=len(rows), successful=len(good),
               alpha=ALPHA, percentile=PERCENTILE, smoothing=False,
               fraction_median=float(np.median([r["fraction"] for r in good])) if good else None,
               errors=[r for r in rows if r["status"] == "error"]))
    print(f"M0 masks completed: {len(good)}/{len(rows)}; reports and previews: {output}", flush=True)
    if len(good) != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
