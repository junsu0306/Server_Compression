"""The budget target must be built without training or modifying the baseline."""
import copy
import unittest
import torch
import timm
from scripts.profiling.prepare_budget import build_target
from pruning.vit_pruning import ViTPruner
from pruning.vit_reducing import reduce_vit_model


class TargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2022)
        self.base = timm.create_model("vit_tiny_patch16_224", pretrained=False).eval()

    def test_target_structure_and_source_preservation(self):
        before = copy.deepcopy(self.base.state_dict())
        candidate = build_target(self.base, 0.3, 0.95)
        ratio = 1 - sum(p.numel() for p in candidate.parameters()) / sum(p.numel() for p in self.base.parameters())
        self.assertLess(abs(ratio - 0.3), 0.001)
        for key in before:
            torch.testing.assert_close(self.base.state_dict()[key], before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in candidate.parameters()))

    def test_pruned_reduced_equivalence_and_mask_resume(self):
        pruner = ViTPruner(self.base, 0.3)
        pruner.apply(self.base)
        restored = ViTPruner(self.base, 0.3)
        restored.load_state_dict(pruner.state_dict())
        image = torch.randn(1, 3, 224, 224)
        reduced = copy.deepcopy(self.base)
        reduce_vit_model(reduced)
        with torch.inference_mode():
            torch.testing.assert_close(self.base(image), reduced(image), atol=2e-6, rtol=1e-5)
        restored.apply(self.base)
        self.assertEqual(restored._step, 2)


if __name__ == "__main__":
    unittest.main()
