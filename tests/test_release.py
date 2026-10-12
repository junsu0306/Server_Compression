"""Packaging keeps inference tensors and excludes training-only state."""
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.package_release import allowlist_text, safe_relative, slim_checkpoint


class ReleaseTests(unittest.TestCase):
    def test_training_state_removed_without_changing_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training.pt"
            target = Path(directory) / "release.pt"
            state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            torch.save({"model": state, "model_ema": state, "epoch": 29, "optimizer": {"step": 999}}, source)
            meta = slim_checkpoint(source, target, ("model", "model_ema", "epoch"))
            result = torch.load(target, weights_only=False)
            self.assertNotIn("optimizer", result)
            self.assertTrue(result["release_inference_only"])
            self.assertEqual(meta["completed_epochs"], 30)
            self.assertTrue(torch.equal(result["model"]["weight"], state["weight"]))
            self.assertIn("optimizer", torch.load(source, weights_only=False))

    def test_missing_checkpoint_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.pt"
            torch.save({"model": {}, "epoch": 29}, source)
            with self.assertRaisesRegex(ValueError, "Missing required"):
                slim_checkpoint(source, Path(directory) / "output.pt", ("model", "model_ema"))

    def test_allowlist_is_exact(self):
        result = allowlist_text(["tiny30/weights.pt", "README.md"])
        self.assertIn("\n*\n!*/\n", result)
        self.assertIn("!/tiny30/weights.pt\n", result)
        self.assertNotIn("!*.pt", result)

    def test_parent_and_absolute_paths_rejected(self):
        for path in ("../secret", "/etc/passwd", "tiny30/../../secret", ""):
            with self.assertRaises(ValueError):
                safe_relative(path)
        self.assertEqual(safe_relative("tiny30/model.pt"), Path("tiny30/model.pt"))
