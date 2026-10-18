"""Coordinated Pix2Pix linear decay and validation-based early stopping."""
import torch


def build_schedulers(cfg, optimizer_g, optimizer_d=None):
    if cfg.lr_policy == "plateau":
        return (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr), None)
    if cfg.lr_policy != "pix2pix_linear" or optimizer_d is None:
        raise ValueError("pix2pix_linear requires both generator and discriminator optimizers")
    constant, decay = cfg.lr_constant_epochs, cfg.lr_decay_epochs

    def factor(completed_epochs):
        return max(0.0, 1.0 - max(0, completed_epochs + 1 - constant) / float(decay + 1))

    return tuple(torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
                 for optimizer in (optimizer_g, optimizer_d))


def step_schedulers(cfg, scheduler_g, scheduler_d, val_loss):
    if cfg.lr_policy == "plateau":
        scheduler_g.step(val_loss)
    else:
        scheduler_g.step()
        scheduler_d.step()


def update_early_stopping(cfg, epoch, val_loss, early_best, bad_epochs):
    """Track patience separately from the best checkpoint over all epochs."""
    if cfg.early_stopping_start == "lr_decay":
        start = cfg.lr_constant_epochs + 1
        if epoch < start:
            return float("inf"), 0
        if epoch == start:
            return val_loss, 0
    if val_loss < early_best - cfg.min_delta:
        return val_loss, 0
    return early_best, bad_epochs + 1
