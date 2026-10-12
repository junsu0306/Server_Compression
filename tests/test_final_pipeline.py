"""CPU regression checks for the memory-enabled final experiment wiring."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import timm
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("final_runner", "scripts/run_pipeline.py")
exporter = load_script("final_exporter", "scripts/export_final_onnx.py")


class FinalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.matrix = yaml.safe_load((ROOT / "configs/final.yaml").read_text())

    def test_gpu_assignments_memory_and_unique_run_ids(self):
        self.assertEqual([entry["gpu"] for entry in self.matrix["models"].values()], [0, 1, 2, 3])
        ids = []
        for key in self.matrix["models"]:
            structural, _ = runner.structural_config(self.matrix, key)
            patch_cfg, _ = runner.patch_config(self.matrix, key)
            self.assertEqual(structural["memory_penalty_weight"], 1)
            self.assertEqual(structural["epochs"], 30)
            self.assertEqual(structural["lr_schedule_epochs"], 50)
            self.assertIn("structural30", structural["wandb_run_name"])
            self.assertIn(key, structural["memory_budget_json"])
            self.assertTrue(structural["memory_budget_json"].endswith("00_budget/budget.json"))
            self.assertNotIn("memory_reference", structural)
            self.assertEqual(patch_cfg["block_finetune"]["epochs"], 3)
            self.assertEqual(patch_cfg["final_finetune"]["epochs"], 30)
            self.assertEqual(patch_cfg["search"]["epsilon"], 0.02)
            ids.extend([structural["wandb_run_id"], patch_cfg["wandb"]["run_id"],
                        patch_cfg["final_finetune"]["wandb_run_id"]])
        self.assertEqual(len(set(ids)), 12)
        self.assertTrue(all("20261006" in value for value in ids))

    def test_only_final_method_is_accepted(self):
        self.matrix["method"] = "other"
        with self.assertRaises(ValueError):
            runner.materialize(self.matrix)

    def test_materialization_is_repeatable_but_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as folder:
            self.matrix["output_root"] = folder
            runner.materialize(self.matrix)
            runner.materialize(self.matrix)
            modified = copy.deepcopy(self.matrix)
            modified["structural"]["memory_penalty_weight"] = 2
            with self.assertRaises(ValueError):
                runner.materialize(modified)

    def test_cached_batch_does_not_repeat_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "batch_probe/structural.json"
            target.parent.mkdir()
            target.write_text('{"selected_batch": 384}')
            job = {"key": "tiny30", "gpu": 1, "paths": {"root": folder}}
            with patch.object(runner, "run", side_effect=AssertionError("must not rerun")):
                self.assertEqual(runner.training_batch(self.matrix, job, "structural"), 384)

    def test_terminal_checkpoint_is_exact_and_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            torch.save({"epoch": 29, "model": {"weight": torch.tensor([30.0])}}, directory / "checkpoint_last.pt")
            # A different best checkpoint must not affect selection.
            torch.save({"epoch": 26}, directory / "checkpoint_best.pt")
            final = runner.pin_terminal_checkpoint(directory, 30)
            self.assertEqual(final.name, "checkpoint_final.pt")
            self.assertEqual(torch.load(final, weights_only=False)["epoch"], 29)
            self.assertEqual(runner.pin_terminal_checkpoint(directory, 30), final)

    def test_terminal_checkpoint_rejects_under_and_overtrained(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for completed in (28, 31, 50):
                torch.save({"epoch": completed - 1}, directory / "checkpoint_last.pt")
                with self.assertRaisesRegex(ValueError, "Need exact epoch 30"):
                    runner.pin_terminal_checkpoint(directory, 30)

    def test_stored_lr_schedule_is_preserved(self):
        from pruning.train import build_scheduler, validate_resume_schedule
        args = SimpleNamespace(epochs=30, lr_schedule_epochs=50, warmup_epochs=5, min_lr=1e-6)
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
        scheduler = build_scheduler(optimizer, args)
        self.assertEqual(scheduler.state_dict()["_schedulers"][1]["T_max"], 45)
        checkpoint = {"args": {"epochs": 50}, "epoch": 27}
        validate_resume_schedule(checkpoint, args)
        with self.assertRaisesRegex(ValueError, "LR horizon"):
            validate_resume_schedule(checkpoint, SimpleNamespace(epochs=30, lr_schedule_epochs=30))
        checkpoint["epoch"] = 30
        with self.assertRaisesRegex(ValueError, "beyond"):
            validate_resume_schedule(checkpoint, args)

    def test_npu_baseline_wrapper_matches_original(self):
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()
            wrapper = exporter.baseline_wrapper(model, SimpleNamespace(npu_safe=True))
            image = torch.randn(1, 224, 224, 3)
            with torch.inference_mode():
                torch.testing.assert_close(wrapper(image), model(image.permute(0, 3, 1, 2)), rtol=0, atol=1e-6)
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
