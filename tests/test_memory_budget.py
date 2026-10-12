"""CPU checks for scalar penalty integration and isolated snapshot handling."""

import copy
import hashlib
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from pruning.engine import train_one_epoch
from pruning.memory_budget import MemoryBudget, memory_penalty
from pruning.vit_pruning import ViTPruner
_spec = importlib.util.spec_from_file_location(
    "profile_reduced_memory", Path(__file__).resolve().parents[1] / "scripts/profiling/profile_reduced_memory.py")
profile_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile_module)
load_candidate = profile_module.load_candidate


class MiniViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].mlp = nn.Module()
        self.blocks[0].mlp.fc1 = nn.Linear(3, 6)
        self.blocks[0].mlp.fc2 = nn.Linear(6, 3)
        self.head = nn.Linear(3, 5)

    def forward(self, x):
        mlp = self.blocks[0].mlp
        return self.head(mlp.fc2(torch.relu(mlp.fc1(x))))


def options(output):
    output = Path(output)
    source = output / "baseline.pt"
    if not source.exists():
        source.write_bytes(b"test baseline provenance")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    budget = output / "budget.json"
    if not budget.exists():
        budget.write_text(json.dumps({"model": "mini", "trained_steps": 0,
            "target_compression": 0.1, "metric": "cuda_peak_allocated_bytes", "max_mib": 10,
            "baseline_checkpoint": str(source), "target_checkpoint": str(source),
            "baseline_sha256": digest, "target_sha256": digest,
            "target_profile": {"dtype": "float32", "warmups": 10, "runs": 50,
                               "input_shape": [1, 3, 224, 224], "peak_allocated_bytes": 10 * 2**20}}))
    return SimpleNamespace(model="mini", memory_penalty_weight=0.5,
                           memory_budget_json=str(budget), target_compression=0.1,
                           memory_profile_interval=2, memory_batch_size=1,
                           input_size=224, memory_warmups=10, memory_runs=50,
                           output_dir=output)


class MemoryBudgetTests(unittest.TestCase):
    def test_hinge_and_invalid_values(self):
        self.assertEqual(memory_penalty(12, 10, 0.5), 1)
        self.assertEqual(memory_penalty(10, 10, 0.5), 0)
        self.assertEqual(memory_penalty(8, 10, 0.5), 0)
        for values in [(float("nan"), 10, 1), (10, 0, 1), (10, 5, -1)]:
            with self.assertRaises(ValueError):
                memory_penalty(*values)

    def test_scalar_loss_preserves_optimizer_update(self):
        torch.manual_seed(10)
        model = MiniViT()
        with_penalty = copy.deepcopy(model)
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        data = [(torch.randn(2, 3), torch.tensor([0, 1])) for _ in range(3)]
        results = []
        for candidate, budget in [(model, None), (with_penalty, SimpleNamespace(penalty=7.0))]:
            optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
            result = train_one_epoch(candidate, nn.CrossEntropyLoss(), data, optimizer,
                                     torch.amp.GradScaler("cuda", enabled=False),
                                     torch.device("cpu"), 0, amp=False, teacher=teacher,
                                     memory_budget=budget)
            results.append(result)
        for a, b in zip(model.parameters(), with_penalty.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertAlmostEqual(results[1]["loss"] - results[0]["loss"], 7, places=5)
        self.assertEqual(results[1]["memory_loss"], 7)
        self.assertAlmostEqual(results[0]["task_kd_loss"], results[1]["task_kd_loss"])

    def test_ema_reducing_preserves_source_and_outputs(self):
        raw = MiniViT()
        ema = copy.deepcopy(raw)
        with torch.no_grad():
            raw.blocks[0].mlp.fc1.weight[1] = 0
            raw.blocks[0].mlp.fc1.bias[1] = 0
            raw.blocks[0].mlp.fc2.weight[:, 1] = 0
        checkpoint = {"args": {"model": "mini"}, "model": raw.state_dict(), "model_ema": ema.state_dict()}
        before = copy.deepcopy(checkpoint)
        with patch.object(profile_module.timm, "create_model", side_effect=lambda *a, **kw: MiniViT()):
            reduced, name = load_candidate(checkpoint)
        self.assertEqual(name, "mini")
        self.assertEqual(reduced.blocks[0].mlp.fc1.out_features, 5)
        x = torch.randn(2, 3)
        torch.testing.assert_close(reduced(x), raw(x))
        for kind in ("model", "model_ema"):
            for key in checkpoint[kind]:
                torch.testing.assert_close(checkpoint[kind][key], before[kind][key])

    def test_measurement_follows_masking_and_applies_on_next_batch(self):
        model = MiniViT()
        pruner = ViTPruner(model, target_compression=0.1, sparsity=0.2, index_refresh_steps=100)
        calls = []
        budget = SimpleNamespace(penalty=0.0)
        def refresh(candidate, ema, epoch):
            self.assertTrue((candidate.blocks[0].mlp.fc1.weight.norm(dim=1) == 0).any())
            calls.append(pruner._step)
            budget.penalty = 3.0
        budget.maybe_refresh = refresh
        data = [(torch.randn(2, 3), torch.tensor([0, 1])) for _ in range(3)]
        result = train_one_epoch(model, nn.CrossEntropyLoss(), data,
                                 torch.optim.SGD(model.parameters(), lr=0.01),
                                 torch.amp.GradScaler("cuda", enabled=False),
                                 torch.device("cpu"), 0, amp=False, pruner=pruner,
                                 memory_budget=budget)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(result["memory_loss"], 2.0)

    def test_reference_interval_resume_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            args = options(directory)
            budget = MemoryBudget(args, "cuda:0")
            model = MiniViT().train()
            initial = copy.deepcopy(model.state_dict())
            rng = torch.get_rng_state().clone()
            def profile(checkpoint, output):
                state = torch.load(checkpoint, weights_only=False)
                self.assertTrue(all(v.device.type == "cpu" for v in state["model"].values()))
                return 12.0
            with patch.object(budget, "_profile", side_effect=profile) as measure:
                budget.maybe_refresh(model, None, 0)
                budget.maybe_refresh(model, None, 1)
                self.assertEqual(measure.call_count, 1)
                resumed = MemoryBudget(args, "cuda:0")
                resumed.load_state_dict(budget.state_dict())
                self.assertEqual(resumed.penalty, 1.0)
                budget.maybe_refresh(model, None, 2)
                self.assertEqual(measure.call_count, 2)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, initial[k])
            self.assertFalse(list(Path(directory).rglob("candidate.pt")))
            args.memory_penalty_weight = 2
            with self.assertRaises(ValueError):
                MemoryBudget(args, "cuda:0").load_state_dict(budget.state_dict())

    def test_profile_failure_is_not_silently_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = MemoryBudget(options(directory), "cuda:0")
            with patch.object(budget, "_profile", side_effect=RuntimeError("CUDA failure")):
                with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
                    budget.maybe_refresh(MiniViT(), None, 0)

    def test_modified_budget_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = options(directory)
            (Path(directory) / "baseline.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint has changed"):
                MemoryBudget(args, "cuda:0")


if __name__ == "__main__":
    unittest.main()
