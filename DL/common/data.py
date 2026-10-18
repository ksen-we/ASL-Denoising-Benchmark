"""Paired NIfTI loading, source normalization, and foreground patch sampling."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from common.config import DEFAULT_NORMALIZATION
from common.masking import load_foreground_mask


@dataclass(frozen=True)
class Case:
    name: str
    source: Path
    target: Path | None
    mask: Path | None = None


def subject_id(name):

    return re.sub(r"_(1500|2000)$", "", name)


def spatial_shape(shape):
    shape = tuple(shape)
    if len(shape) == 4 and shape[-1] == 1:
        shape = shape[:3]
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI or singleton final axis, got {shape}")
    return shape


def discover_cases(root, cfg, require_target=True):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    cases, skipped = [], []
    shapes = Counter()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        source, target = folder / cfg.source_name, folder / cfg.target_name
        missing = [p.name for p in (source, target) if not p.is_file() and (p == source or require_target)]
        if missing:
            skipped.append({"case": folder.name, "missing": missing})
            continue
        image = nib.load(str(source))
        shape = spatial_shape(image.shape)
        if target.is_file():
            ref = nib.load(str(target))
            if spatial_shape(ref.shape) != shape or not np.allclose(image.affine, ref.affine, rtol=0, atol=1e-4):
                raise ValueError(f"{folder.name}: source/target shape or affine mismatch; registration is required")
        mask = folder / cfg.mask_name if cfg.mask_mode == "m0" else None
        if mask is not None:
            load_foreground_mask(mask, image)
        cases.append(Case(folder.name, source, target if target.is_file() else None, mask))
        shapes[str(shape)] += 1
    if skipped and not cfg.skip_incomplete:
        raise ValueError(f"Directories with incomplete source/target pairs: {skipped}")
    if not cases:
        raise ValueError(f"No usable paired data in {root}")
    return cases, {"root": str(root), "included": len(cases), "shapes_xyz": dict(shapes), "skipped": skipped}


def discover_splits(cfg):
    train, tr = discover_cases(Path(cfg.data_root) / "train", cfg)
    val, va = discover_cases(Path(cfg.data_root) / "val", cfg)
    overlap = {subject_id(c.name) for c in train} & {subject_id(c.name) for c in val}
    if overlap:
        raise ValueError(f"Subjects overlap between train and val: {sorted(overlap)}")
    return train, val, {"train": tr, "val": va, "subject_overlap": [],
                       "subject_rule": "remove terminal _1500 or _2000; no further identity metadata available"}


def read_volume(path):
    image = nib.load(str(path))
    shape = spatial_shape(image.shape)
    data = image.get_fdata(dtype=np.float32).reshape(shape)
    return data.transpose(2, 0, 1), image


def load_case(case, normalization=DEFAULT_NORMALIZATION):
    if normalization not in {"p1p99", "minmax"}:
        raise ValueError("normalization must be p1p99 or minmax")
    source, image = read_volume(case.source)
    source_valid = np.isfinite(source)
    if not source_valid.any():
        raise ValueError(f"{case.name}: source contains no finite values")
    values = source[source_valid]
    if normalization == "p1p99":
        low, high = map(float, np.percentile(values, [1, 99]))
    else:
        low, high = float(values.min()), float(values.max())
    span = high - low
    scale = span if span > 1e-6 else 1.0

    x = np.where(source_valid, (source - low) / scale, 0).astype(np.float32)
    foreground = source_valid.copy() if case.mask is None else load_foreground_mask(case.mask, image).transpose(2, 0, 1)
    target, valid = None, source_valid & foreground
    reference_range = None
    if case.target is not None:
        raw, _ = read_volume(case.target)
        valid &= np.isfinite(raw)
        if not valid.any():
            raise ValueError(f"{case.name}: source and target have no jointly finite voxels")

        finite_reference = raw[np.isfinite(raw)]
        ref_low, ref_high = float(finite_reference.min()), float(finite_reference.max())
        reference_range = {"lower": ref_low, "upper": ref_high,
                           "scale": ref_high - ref_low if ref_high - ref_low > 1e-6 else 1.0,
                           "degenerate": ref_high - ref_low <= 1e-6}
        target = np.where(valid, (raw - low) / scale, 0).astype(np.float32)
    if not valid.any():
        raise ValueError(f"{case.name}: no jointly finite foreground voxels")
    return {"input": x, "target": target, "valid": valid, "source_valid": source_valid,
            "foreground": foreground, "mask_path": case.mask,
            "normalization": {"method": normalization, "lower": low, "upper": high,
                              "percentiles": [1, 99] if normalization == "p1p99" else None,
                              "clipped": False, "statistics_scope": "source_all_finite"},
            "offset": low, "scale": scale, "reference_range": reference_range,
            "image": image, "case": case.name}


def adjacent_slices(volume, z):
    """Return [z-1, z, z+1], repeating the nearest slice at volume boundaries."""
    ids = np.clip([z - 1, z, z + 1], 0, volume.shape[0] - 1)
    return volume[ids]


class TrainingDataset(Dataset):
    def __init__(self, cases, cfg):
        self.cases, self.cfg = cases, cfg
        self.epoch = 0
        self.cache = OrderedDict()

    def __len__(self):
        return self.cfg.steps_per_epoch * self.cfg.batch_size

    def _get_case(self, index):
        if index not in self.cache:
            value = load_case(self.cases[index], self.cfg.normalization)
            if self.cfg.cache_cases <= 0:
                return value
            self.cache[index] = value
            if len(self.cache) > self.cfg.cache_cases:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):

        rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch, index]))
        item = self._get_case(int(rng.integers(len(self.cases))))
        x, y, m = item["input"], item["target"], item["valid"]
        if self.cfg.mode == "2.5d":
            valid_slices = np.flatnonzero(m.any(axis=(1, 2)))
            z = int(rng.choice(valid_slices))
            x, y, m = adjacent_slices(x, z), y[z:z + 1], m[z:z + 1]

        else:
            x, y, m = x[None], y[None], m[None]
        patch = tuple(self.cfg.patch_size)
        padding = ((0, 0),) + tuple((0, max(p - n, 0)) for p, n in zip(patch, y.shape[1:]))
        x, y, m = np.pad(x, padding, mode="edge"), np.pad(y, padding), np.pad(m, padding)

        coords = np.argwhere(m[0])
        center = coords[int(rng.integers(len(coords)))]
        starts = [int(np.clip(c - rng.integers(p), 0, n - p))
                  for c, p, n in zip(center, patch, y.shape[1:])]
        region = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(starts, patch))
        x, y, m = x[region], y[region], m[region]
        if self.cfg.augment:
            for axis in (-1, -2):
                if rng.random() < 0.5:
                    x, y, m = (np.flip(v, axis=axis) for v in (x, y, m))
            k = int(rng.integers(4))
            x, y, m = (np.rot90(v, k, axes=(-2, -1)) for v in (x, y, m))
        if self.cfg.mode == "2.5d" and (x.shape != (3, *patch) or y.shape != (1, *patch)):
            raise ValueError(f"2.5D requires three input slices and one center target slice, got {x.shape} -> {y.shape}")
        return {"input": torch.from_numpy(np.ascontiguousarray(x)),
                "target": torch.from_numpy(np.ascontiguousarray(y)),
                "mask": torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))}


def save_prediction(prediction, item, path):
    """Restore original units and preserve source NIfTI geometry."""
    restored = prediction * item["scale"] + item["offset"]
    restored = np.where(item["source_valid"], restored, 0).transpose(1, 2, 0).astype(np.float32)
    original = item["image"]
    if len(original.shape) == 4:
        restored = restored[..., None]
    header = original.header.copy()
    header.set_data_dtype(np.float32)
    output = original.__class__(restored, original.affine, header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    output.header.set_slope_inter(1.0, 0.0)
    output.header["cal_min"], output.header["cal_max"] = float(restored.min()), float(restored.max())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(path))
