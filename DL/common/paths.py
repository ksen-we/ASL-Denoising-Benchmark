"""Output locking and compatibility with older directory layouts."""
from pathlib import Path
import shutil


def claim_output_dir(folder, cfg, *, training=False):
    """Claim an exclusive lock before preparing or replacing output files."""
    folder = Path(folder)
    root = Path(cfg.output_root).resolve()
    if folder.is_symlink() or root not in folder.resolve().parents:
        raise ValueError(f"Output directory cannot be a symlink or outside output_root: {folder}")
    reuse = training and bool(cfg.resume_checkpoint)
    if reuse and cfg.overwrite_outputs:
        raise ValueError("Overwriting and resuming cannot be enabled together")
    if folder.exists() and not (reuse or cfg.overwrite_outputs):
        raise FileExistsError(f"Output exists: {folder}; set OVERWRITE_OUTPUTS=True to restart or RESUME=True to continue")
    created_dirs = []
    for path in (folder, *folder.parents):
        if path.exists():
            break
        created_dirs.append(path)
    folder.mkdir(parents=True, exist_ok=reuse or cfg.overwrite_outputs)
    lock = folder / (".training.lock" if training else ".inference.lock")
    try:
        with lock.open("x"):
            pass
    except FileExistsError as exc:
        raise RuntimeError(f"Task lock exists: {lock}; remove it only after confirming the previous task has stopped") from exc
    try:

        conflicts = [p for p in folder.rglob(".inference.lock") if p != lock]
        if not training:
            conflicts.append(cfg.run_dir / ".training.lock")
            for parent in folder.parents:
                if parent == cfg.run_dir:
                    break
                conflicts.append(parent / ".inference.lock")
        active = [p for p in conflicts if p.exists() or p.is_symlink()]
        if active:
            raise RuntimeError(f"An active training/inference lock prevents overwriting or concurrent writes: {active[0]}")
        if cfg.overwrite_outputs:
            for child in folder.iterdir():
                if child == lock:
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        return lock
    except BaseException:
        lock.unlink(missing_ok=True)

        for path in created_dirs:
            try:
                path.rmdir()
            except OSError:
                break
        raise


def prepare_run_dir(cfg):
    """Migrate idle legacy folders without merging or overwriting experiments."""
    old, new = cfg.legacy_run_dir, cfg.run_dir
    if not old.is_dir():
        return new
    if new.exists() or new.is_symlink():
        raise FileExistsError(f"Both current and legacy run directories exist; refusing to merge: {new}; {old}")
    if (old / ".training.lock").exists() or any(old.glob("inference/**/.inference.lock")):
        raise RuntimeError(f"Legacy run is locked: {old}; retry migration after the task finishes")
    old.rename(new)

    try:
        old.parent.rmdir()
    except OSError:
        pass
    print(f"Migrated experiment directory: {old} -> {new}", flush=True)
    return new


def resolve_checkpoint(path, cfg):
    """Resolve current and legacy checkpoint paths."""
    path = Path(path)
    if path.is_file():
        return path
    for source, destination in ((cfg.legacy_run_dir, cfg.run_dir),
                                (cfg.run_dir, cfg.legacy_run_dir)):
        try:
            relative = path.absolute().relative_to(source.absolute())
        except ValueError:
            continue
        candidate = destination / relative
        if candidate.is_file():
            return candidate
    return path
