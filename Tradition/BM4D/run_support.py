"""Record configurations and hashes; write results atomically."""
import csv
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def ensure_config(root, config, overwrite=False):
    path = Path(root) / "run_config.json"
    signature = digest_json(config)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["signature"] != signature and not overwrite:
            raise ValueError("Output configuration differs. Choose a new OUTPUT_ROOT or set OVERWRITE=True.")
    write_json(path, {"signature": signature, "config": config})
    return signature


def verified_existing(output, source, signature):
    sidecar = Path(str(output) + ".json")
    if not sidecar.is_file():
        raise ValueError(f"Missing provenance for {output}; choose a new output directory or set OVERWRITE=True.")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    if (saved.get("signature") != signature or saved.get("input_sha256") != file_hash(source)
            or saved.get("output_sha256") != file_hash(output)):
        raise ValueError(f"Configuration, input or output fingerprint mismatch: {output}.")
    return saved


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
