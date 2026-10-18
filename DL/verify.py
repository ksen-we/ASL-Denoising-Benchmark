"""Run synthetic CPU checks: python DL/verify.py. No study data are required."""
from itertools import product
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

import run_all
from common.data import Case, adjacent_slices, load_case, save_prediction
from common.factory import build_discriminator, build_model
from common.losses import ReconstructionLoss, weighted_bce_with_logits
from common.scheduling import build_schedulers, step_schedulers, update_early_stopping


def small_config(method, mode, residual=False):
    cfg = run_all.build_config(method, mode, residual)
    cfg.device, cfg.amp = "cpu", False
    if method == "SwinIR":
        cfg.model = dict(embed_dim=8, depths=(2,), heads=2,
                         window=(2, 4, 4) if mode == "3d" else (4, 4),
                         mlp_ratio=2, use_checkpoint=True)
    elif method == "SwinUNet":
        cfg.model = dict(embed_dim=8, depths=(2, 2, 2), heads=(2, 2, 4),
                         window=(2, 4, 4) if mode == "3d" else (4, 4),
                         mlp_ratio=2, use_checkpoint=True)
    elif method == "UNet":
        cfg.model = dict(base_channels=4, levels=4)
    else:
        cfg.model = dict(base_channels=4, dropout=0.2)
        cfg.discriminator_channels = 4
    return cfg


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_study_configuration(self):
        for method, mode, residual in product(run_all.METHODS, run_all.MODES, (False, True)):
            cfg = run_all.build_config(method, mode, residual)
            self.assertEqual((cfg.normalization, cfg.evaluation_protocol), ("minmax", "legacy"))
            self.assertEqual(cfg.run_name, f"wtm0mask_ori_gres{int(residual)}_seed42")
            self.assertEqual(cfg.batch_size, 1 if mode == "3d" else 16)
            self.assertFalse(cfg.overwrite_outputs)
            if method == "Pix2Pix":
                self.assertEqual((cfg.lr_policy, cfg.early_stopping_start, cfg.adversarial_mask_mode),
                                 ("pix2pix_linear", "lr_decay", "foreground_fraction"))

    def test_all_models_shape_initialization_and_gradients(self):
        for method, mode, residual in product(run_all.METHODS, run_all.MODES, (False, True)):
            with self.subTest(method=method, mode=mode, residual=residual):
                cfg = small_config(method, mode, residual)
                model = build_model(cfg)
                x = torch.randn((1, 1, 9, 17, 19) if mode == "3d" else (2, 3, 17, 19))
                center = x if mode == "3d" else x[:, 1:2]
                expected = center if residual else torch.zeros_like(center)
                torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                for _ in range(2):
                    optimizer.zero_grad()
                    prediction = model(x)
                    self.assertEqual(prediction.shape, center.shape)
                    loss, _ = ReconstructionLoss(cfg)(prediction, center * 0.8 + 0.1)
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    optimizer.step()
                if method in {"SwinIR", "SwinUNet"}:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for name, p in model.named_parameters() if "qkv" in name))

    def test_foreground_bce(self):
        for mode in run_all.MODES:
            discriminator = build_discriminator(small_config("Pix2Pix", mode))
            shape = (1, 1, 8, 16, 16) if mode == "3d" else (1, 1, 16, 16)
            mask = torch.ones(shape)
            mask[..., :8] = 0
            weights = discriminator.foreground_weights(mask)
            condition = torch.randn(shape if mode == "3d" else (1, 3, 16, 16))
            logits = discriminator(condition, torch.randn(shape) * mask)
            self.assertEqual(weights.shape, logits.shape)
            self.assertTrue(((weights >= 0) & (weights <= 1)).all())
            targets = torch.ones_like(logits)
            expected = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none") * weights).sum() / weights.sum()
            torch.testing.assert_close(weighted_bce_with_logits(logits, targets, weights), expected)
            with self.assertRaises(ValueError):
                weighted_bce_with_logits(logits, targets, torch.zeros_like(weights))

    def test_normalization_and_nifti_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.arange(120, dtype=np.float32).reshape(5, 6, 4) - 20
            target = 2 * source + 10
            affine = np.diag([3.5, 3.5, 6, 1])
            for name, values in (("source.nii", source), ("target.nii", target)):
                nib.save(nib.Nifti1Image(values, affine), root / name)
            loaded = load_case(Case("synthetic", root / "source.nii", root / "target.nii"))
            self.assertEqual((loaded["offset"], loaded["scale"]), (-20, 119))
            self.assertGreater(loaded["target"].max(), 1)
            save_prediction(loaded["input"], loaded, root / "prediction.nii")
            actual = nib.load(root / "prediction.nii")
            np.testing.assert_allclose(actual.get_fdata(), source, atol=1e-5)
            np.testing.assert_array_equal(actual.affine, affine)
            np.testing.assert_array_equal(adjacent_slices(loaded["input"], 0)[0], loaded["input"][0])

    def test_coordinated_decay_and_delayed_stopping(self):
        cfg = run_all.build_config("Pix2Pix", "3d", False)
        g, d = (torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr) for _ in range(2))
        sg, sd = build_schedulers(cfg, g, d)
        best, bad = float("inf"), 0
        for epoch in range(1, 201):
            self.assertEqual(g.param_groups[0]["lr"], d.param_groups[0]["lr"])
            self.assertGreater(g.param_groups[0]["lr"], 0)
            if epoch <= 100:
                self.assertEqual(g.param_groups[0]["lr"], cfg.lr)
            best, bad = update_early_stopping(cfg, epoch, 1.0, best, bad)
            if epoch <= 101:
                self.assertEqual(bad, 0)
            if epoch == 131:
                self.assertEqual(bad, cfg.early_stopping_patience)
            g.step()
            d.step()
            step_schedulers(cfg, sg, sd, 1.0)
        self.assertEqual(g.param_groups[0]["lr"], 0)
        self.assertEqual(d.param_groups[0]["lr"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
