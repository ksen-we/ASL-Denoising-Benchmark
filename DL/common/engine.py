"""Shared training and inference execution."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.config import DL_ROOT, ExperimentConfig
from common.data import TrainingDataset, discover_cases, discover_splits
from common.evaluation import evaluate
from common.factory import build_discriminator, build_model
from common.losses import ReconstructionLoss, weighted_bce_with_logits
from common.masking import metric_protocol
from common.paths import claim_output_dir, prepare_run_dir, resolve_checkpoint
from common.scheduling import build_schedulers, step_schedulers, update_early_stopping
from common.utils import (atomic_checkpoint, file_hash, fingerprint, load_checkpoint,
                          restore_rng, rng_state, save_curves, setup_logger,
                          setup_runtime, write_csv, write_json)


def _check_resume(saved, cfg, data_fingerprint):

    mutable = {"device", "num_workers", "cpu_threads", "cache_cases", "epochs", "resume_checkpoint",
               "inference_checkpoint", "inference_split", "inference_case", "predict_after_training",
               "show_plots", "preview_every", "log_every", "overwrite_outputs", "evaluation_protocol"}
    old, new = ExperimentConfig.from_dict(saved["config"]).to_dict(), cfg.to_dict()

    if (old["lr_policy"] == new["lr_policy"] == "pix2pix_linear"
            and old["early_stopping_patience"] > 0 and new["early_stopping_patience"] == 0):
        mutable.add("early_stopping_patience")
    if (old["lr_policy"] == new["lr_policy"] == "pix2pix_linear"
            and old["early_stopping_start"] == "immediate" and new["early_stopping_start"] == "lr_decay"):
        mutable.add("early_stopping_start")

        if old["early_stopping_patience"] == 0:
            mutable.add("early_stopping_patience")
    changed = [k for k in new if k not in mutable and old.get(k) != new[k]]
    if changed:
        raise ValueError(f"Incompatible resume configuration: {changed}; use a new run_name for changed conditions")
    if saved["data_fingerprint"] != data_fingerprint:
        raise ValueError("Data files or modification times changed; refusing to resume this experiment")


def _snapshot_code(run_dir):
    dest = run_dir / "code_snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DL_ROOT / "run_all.py", dest / "run_all.py")
    for folder in ("common", "SwinIR", "Pix2Pix", "SwinUNet", "UNet"):
        for source in (DL_ROOT / folder).glob("*.py"):
            target = dest / folder / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _train(cfg):
    cfg.validate()
    device = setup_runtime(cfg)
    train_cases, val_cases, audit = discover_splits(cfg)
    train_fp, train_manifest = fingerprint(train_cases)
    val_fp, val_manifest = fingerprint(val_cases)
    data_fp = train_fp + ":" + val_fp
    resume = load_checkpoint(resolve_checkpoint(cfg.resume_checkpoint, cfg)) if cfg.resume_checkpoint else None
    if resume:
        _check_resume(resume, cfg, data_fp)
    run_dir = prepare_run_dir(cfg)
    lock = claim_output_dir(run_dir, cfg, training=True)
    logger = None
    try:
        logger = setup_logger(run_dir)
        logger.info("%s | %s | %s | device=%s", cfg.method, cfg.mode, "GAN" if cfg.use_gan else "L1+SSIM", device)
        logger.info("Train %d / val %d; incomplete pairs skipped %d / %d", len(train_cases), len(val_cases), len(audit["train"]["skipped"]), len(audit["val"]["skipped"]))
        logger.info("Output: %s", run_dir)
        logger.info("Sampling/loss/evaluation mask: %s; source context retained, %s normalization over all finite source voxels without clipping", cfg.mask_mode, cfg.normalization)
        logger.info("Reported metric protocol: %s; best checkpoint, early stopping, and plateau monitor validation reconstruction loss on the training scale", metric_protocol(cfg))
        if device.type == "cpu":
            logger.info("Running on CPU; CUDA is recommended for full training")
        if cfg.mode == "2.5d":
            logger.info("Each sample [3,H,W] -> [1,H,W]; batch=%d contains %d independent three-slice samples", cfg.batch_size, cfg.batch_size)
        write_json(run_dir / "config.json", cfg.to_dict())
        write_json(run_dir / "metric_protocol.json", {"metric_protocol": metric_protocol(cfg),
                   "selection_metric": "training_scale_reconstruction_loss",
                   "training_normalization": cfg.normalization})
        write_json(run_dir / "data_audit.json", audit)
        write_json(run_dir / "data_manifest.json", {"train": train_manifest, "val": val_manifest,
                                                   "train_fingerprint": train_fp, "val_fingerprint": val_fp})
        write_csv(run_dir / "split.csv", [{"split": split, "case": c.name, "source": str(c.source), "target": str(c.target)}
                                         for split, cases in (("train", train_cases), ("val", val_cases)) for c in cases])
        if resume is None:
            _snapshot_code(run_dir)
        model = build_model(cfg).to(device)
        logger.info("Global image residual=%s; zero output-head initialization=%s; final-image supervision against the high-SNR reference",
                    model.global_residual, model.zero_init_output)
        discriminator = build_discriminator(cfg)
        if discriminator is not None:
            discriminator.to(device)
            logger.info("Adversarial BCE weighting=%s; convolution support=%s; G and D share the weight map",
                        cfg.adversarial_mask_mode, discriminator.receptive_field)
        params = sum(p.numel() for p in model.parameters())
        logger.info("Generator parameters %.3f M; %d steps/epoch; batch=%d; patch=%s", params / 1e6, cfg.steps_per_epoch, cfg.batch_size, cfg.patch_size)
        write_json(run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "generator_parameters": params, "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()) if discriminator else 0})
        criterion = ReconstructionLoss(cfg).to(device)
        opt_g = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.discriminator_lr, betas=(0.5, 0.999)) if discriminator else None
        scheduler, scheduler_d = build_schedulers(cfg, opt_g, opt_d)
        logger.info("LR policy=%s; initial G=%g, D=%s; early-stopping patience=%d (0 disables)",
                    cfg.lr_policy, cfg.lr, cfg.discriminator_lr if opt_d else "none", cfg.early_stopping_patience)
        if scheduler_d is not None:
            logger.info("Coordinated G/D schedule: %d constant epochs, then %d decay epochs; validation loss does not change learning rates",
                        cfg.lr_constant_epochs, cfg.lr_decay_epochs)
            if cfg.early_stopping_patience == 0:
                logger.info("Early stopping disabled: run through epoch %d and select best.pt by full-validation reconstruction loss", cfg.epochs)
                if resume and ExperimentConfig.from_dict(resume["config"]).early_stopping_patience > 0:
                    logger.info("Early stopping disabled for resume; preserving G/D optimizers, scheduler state, and the best checkpoint")
            elif cfg.early_stopping_start == "lr_decay":
                logger.info("Delayed early stopping: skip %d epochs, set the baseline at epoch %d, then stop after %d epochs without sufficient improvement; best.pt spans all epochs",
                            cfg.lr_constant_epochs, cfg.lr_constant_epochs + 1, cfg.early_stopping_patience)
            else:
                logger.warning("Early stopping may shorten the fixed G/D schedule; set early_stopping_patience=0 to complete the schedule")
        amp = cfg.amp and device.type == "cuda"
        scaler_g = torch.amp.GradScaler("cuda", enabled=amp)
        scaler_d = torch.amp.GradScaler("cuda", enabled=amp)
        start, best, early_best, bad_epochs, history = 1, float("inf"), float("inf"), 0, []
        if resume:
            model.load_state_dict(resume["model"])
            opt_g.load_state_dict(resume["optimizer_g"])
            scheduler.load_state_dict(resume["scheduler"])
            if scheduler_d is not None:
                if resume.get("scheduler_d") is None:
                    raise ValueError("Linear G/D checkpoint is missing scheduler_d; refusing to reset discriminator decay")
                scheduler_d.load_state_dict(resume["scheduler_d"])
            scaler_g.load_state_dict(resume["scaler_g"])
            if discriminator:
                discriminator.load_state_dict(resume["discriminator"])
                opt_d.load_state_dict(resume["optimizer_d"])
                scaler_d.load_state_dict(resume["scaler_d"])
            start, best, bad_epochs = resume["epoch"] + 1, resume["best_loss"], resume["bad_epochs"]
            early_best = resume.get("early_best", best)
            history = resume["history"]
            old_cfg = ExperimentConfig.from_dict(resume["config"])
            old_protocol = metric_protocol(old_cfg)
            for row in history:
                row.setdefault("evaluation_protocol", old_protocol)
                row.setdefault("val_ssim_training_scale", row["val_ssim"] if old_cfg.evaluation_protocol == "legacy" else None)
                for key in ("val_mae_raw", "val_rmse_raw", "val_nmse_raw", "val_evaluation_clipped_fraction"):
                    row.setdefault(key, None)
            if old_cfg.evaluation_protocol != cfg.evaluation_protocol:
                logger.info("Metric protocol changed %s -> %s; retaining labeled history with separate curves; validation reconstruction loss and checkpoint/stopping/scheduler state are unchanged",
                            old_cfg.evaluation_protocol, cfg.evaluation_protocol)
            if ExperimentConfig.from_dict(resume["config"]).early_stopping_start != cfg.early_stopping_start:
                early_best, bad_epochs = float("inf"), 0
                for row in history:
                    early_best, bad_epochs = update_early_stopping(
                        cfg, row["epoch"], row["val_loss"], early_best, bad_epochs)
                logger.info("Resume switched to delayed early stopping; replayed history gives counter=%d; G/D schedule and the best checkpoint are preserved", bad_epochs)

            for row in history:
                row.setdefault("lr_d", cfg.discriminator_lr if opt_d else None)
                row.setdefault("lr_g_used", None)
                row.setdefault("lr_d_used", None)
            restore_rng(resume["rng"])
            logger.info("Resuming at epoch %d; best validation loss=%.6f", start, best)
        dataset = TrainingDataset(train_cases, cfg)
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=device.type == "cuda", persistent_workers=False, generator=generator)
        for epoch in range(start, cfg.epochs + 1):
            lr_g_used = opt_g.param_groups[0]["lr"]
            lr_d_used = opt_d.param_groups[0]["lr"] if opt_d else None
            dataset.epoch = epoch
            generator.manual_seed(cfg.seed + epoch)
            model.train()
            if discriminator:
                discriminator.train()
            totals = {"recon": 0.0, "l1": 0.0, "ssim": 0.0, "g_adv": 0.0, "d_loss": 0.0}
            tick = time.perf_counter()
            for step, batch in enumerate(loader, 1):
                x, y, mask = (batch[k].to(device, non_blocking=True) for k in ("input", "target", "mask"))
                opt_g.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    prediction = model(x)
                d_loss = prediction.new_zeros(())
                if discriminator:

                    adv_weights = (discriminator.foreground_weights(mask)
                                   if cfg.adversarial_mask_mode == "foreground_fraction" else None)
                    discriminator.requires_grad_(True)
                    opt_d.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):

                        real_logits = discriminator(x, y * mask)
                        fake_logits = discriminator(x, prediction.detach() * mask)
                        d_loss = 0.5 * (
                            weighted_bce_with_logits(real_logits, torch.ones_like(real_logits), adv_weights)
                            + weighted_bce_with_logits(fake_logits, torch.zeros_like(fake_logits), adv_weights))
                    if not torch.isfinite(d_loss):
                        raise FloatingPointError("Discriminator loss is NaN/Inf")
                    scaler_d.scale(d_loss).backward()
                    scaler_d.unscale_(opt_d)
                    nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.grad_clip, error_if_nonfinite=not amp)
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    discriminator.requires_grad_(False)
                recon, parts = criterion(prediction, y, mask)
                adv = recon.new_zeros(())
                if discriminator:
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits = discriminator(x, prediction * mask)
                        adv = weighted_bce_with_logits(logits, torch.ones_like(logits), adv_weights)
                loss = recon + cfg.adversarial_weight * adv
                if not torch.isfinite(loss):
                    raise FloatingPointError("Generator loss is NaN/Inf")
                scaler_g.scale(loss).backward()
                scaler_g.unscale_(opt_g)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=not amp)
                scaler_g.step(opt_g)
                scaler_g.update()
                for key, value in (("recon", recon), ("l1", parts["l1"]), ("ssim", parts["ssim"]), ("g_adv", adv), ("d_loss", d_loss)):
                    totals[key] += float(value.detach())
                if cfg.log_every and step % cfg.log_every == 0:
                    logger.info("epoch %d/%d | step %d/%d | recon %.5f", epoch, cfg.epochs, step, len(loader), totals["recon"] / step)
            preview = run_dir / "previews" / f"epoch_{epoch:04d}.png" if cfg.preview_every and (epoch == 1 or epoch % cfg.preview_every == 0) else None
            summary, _ = evaluate(model, val_cases, cfg, device, logger, preview_path=preview)
            val_loss = summary["loss_mean"]
            step_schedulers(cfg, scheduler, scheduler_d, val_loss)
            improved = val_loss < best
            early_best, bad_epochs = update_early_stopping(cfg, epoch, val_loss, early_best, bad_epochs)
            if improved:
                best = val_loss
            row = {"epoch": epoch, **{"train_" + k: v / len(loader) for k, v in totals.items()},
                   "val_loss": val_loss, "val_l1": summary["mae_normalized_mean"], "val_ssim": summary["ssim_mean"],
                   "val_psnr": summary["psnr_mean"], "lr_g": opt_g.param_groups[0]["lr"],
                   "evaluation_protocol": metric_protocol(cfg),
                   "val_ssim_training_scale": summary.get("ssim_training_scale_mean", summary["ssim_mean"]),
                   "val_mae_raw": summary.get("mae_raw_mean"), "val_rmse_raw": summary.get("rmse_raw_mean"),
                   "val_nmse_raw": summary.get("nmse_raw_mean"),
                   "val_evaluation_clipped_fraction": summary.get("evaluation_clipped_fraction_mean"),
                   "lr_d": opt_d.param_groups[0]["lr"] if opt_d else None,
                   "lr_g_used": lr_g_used, "lr_d_used": lr_d_used,
                   "seconds": time.perf_counter() - tick}
            history.append(row)
            state = {"format_version": 2, "epoch": epoch, "config": cfg.to_dict(), "model": model.state_dict(),
                     "optimizer_g": opt_g.state_dict(), "scheduler": scheduler.state_dict(), "scaler_g": scaler_g.state_dict(),
                     "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
                     "discriminator": discriminator.state_dict() if discriminator else None,
                     "optimizer_d": opt_d.state_dict() if opt_d else None, "scaler_d": scaler_d.state_dict(),
                     "best_loss": best, "early_best": early_best, "bad_epochs": bad_epochs, "history": history, "rng": rng_state(),
                     "data_fingerprint": data_fp}
            if improved:
                atomic_checkpoint(run_dir / "checkpoints" / "best.pt", state)
            atomic_checkpoint(run_dir / "checkpoints" / "last.pt", state)
            write_csv(run_dir / "history.csv", history)
            save_curves(history, run_dir / "learning_curves.png", cfg.show_plots)
            logger.info("epoch %d | train %.5f | val %.5f | SSIM %.4f | PSNR %.2f dB | %.1fs%s", epoch,
                        row["train_recon"], val_loss, row["val_ssim"], row["val_psnr"], row["seconds"], " | best" if improved else "")
            if opt_d:
                logger.info("Current LR G=%g / D=%g; next epoch G=%g / D=%g",
                            lr_g_used, lr_d_used, row["lr_g"], row["lr_d"])
            if cfg.early_stopping_patience and bad_epochs >= cfg.early_stopping_patience:
                logger.info("Early stopping after %d epochs without sufficient improvement", bad_epochs)
                break
        logger.info("Training complete. Best checkpoint: %s", run_dir / "checkpoints" / "best.pt")
    finally:
        lock.unlink(missing_ok=True)
        for handler in logger.handlers[:] if logger else []:
            handler.close()
            logger.removeHandler(handler)
    return run_dir


def train(cfg):

    run_dir = _train(cfg)
    if cfg.predict_after_training:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        infer(replace(cfg, inference_checkpoint=run_dir / "checkpoints" / "best.pt",
                      resume_checkpoint=None,
                      overwrite_outputs=cfg.overwrite_outputs or bool(cfg.resume_checkpoint)))
    return run_dir


def infer(cfg):
    cfg.validate()
    checkpoint_path = Path(cfg.inference_checkpoint) if cfg.inference_checkpoint else cfg.run_dir / "checkpoints" / "best.pt"
    checkpoint_path = resolve_checkpoint(checkpoint_path, cfg)
    saved = load_checkpoint(checkpoint_path)
    trained = ExperimentConfig.from_dict(saved["config"]).validate()
    if (cfg.method, cfg.mode) != (trained.method, trained.mode):
        raise ValueError("Inference method or dimension does not match the checkpoint")

    effective = replace(trained, data_root=cfg.data_root, output_root=cfg.output_root, device=cfg.device,
                        cpu_threads=cfg.cpu_threads, amp=cfg.amp, inference_batch_size=cfg.inference_batch_size,
                        inference_patch_size=cfg.inference_patch_size, inference_overlap=cfg.inference_overlap,
                        inference_split=cfg.inference_split, inference_case=cfg.inference_case,
                        evaluation_protocol=cfg.evaluation_protocol,
                        show_plots=cfg.show_plots, overwrite_outputs=cfg.overwrite_outputs,
                        resume_checkpoint=None)
    if effective.inference_split not in {"train", "val", "test"}:
        raise ValueError("inference_split must be train, val, or a separately supplied test directory")
    prepare_run_dir(effective)
    checkpoint_path = resolve_checkpoint(checkpoint_path, effective)
    device = setup_runtime(effective)
    cases, audit = discover_cases(Path(effective.data_root) / effective.inference_split, effective, require_target=False)
    if effective.inference_case:
        cases = [c for c in cases if c.name == effective.inference_case]
        if not cases:
            raise ValueError(f"Requested case not found: {effective.inference_case}")
    folder = effective.run_dir / "inference" / effective.inference_split
    if effective.inference_case:
        folder = folder / "single_cases" / effective.inference_case
    lock = claim_output_dir(folder, effective)
    logger = None
    try:
        logger = setup_logger(folder, "inference.log")
        logger.info("Loaded %s (epoch %d); predicting %d cases", checkpoint_path, saved["epoch"], len(cases))
        logger.info("Evaluation protocol: %s", metric_protocol(effective))
        model = build_model(trained).to(device)
        logger.info("Checkpoint settings: global residual=%s; zero output-head initialization=%s; normalization=%s",
                    model.global_residual, model.zero_init_output, effective.normalization)
        model.load_state_dict(saved["model"])
        summary, _ = evaluate(model, cases, effective, device, logger, export_dir=folder)
        digest, manifest = fingerprint(cases)
        write_json(folder / "provenance.json", {"config": effective.to_dict(), "checkpoint": str(checkpoint_path),
                   "checkpoint_sha256": file_hash(checkpoint_path), "checkpoint_epoch": saved["epoch"],
                   "data_fingerprint": digest, "manifest": manifest, "audit": audit,
                   "metric_protocol": metric_protocol(effective)})
        write_json(folder / "completed.json", {"checkpoint_epoch": saved["epoch"], "n_cases": len(cases)})
        logger.info("Complete: %s", folder)
        return summary
    finally:
        lock.unlink(missing_ok=True)
        for handler in logger.handlers[:] if logger else []:
            handler.close()
            logger.removeHandler(handler)


def run_experiment(cfg, action="train"):
    if action == "train":
        return train(cfg)
    if action == "infer":
        return infer(cfg)
    if action == "check":
        cfg.validate()
        _, _, report = discover_splits(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    raise ValueError("ACTION must be 'train', 'infer', or 'check'")
