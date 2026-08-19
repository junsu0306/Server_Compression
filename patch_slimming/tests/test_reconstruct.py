"""§16.7 — Reconstruction gradient routing test.

The searched block must receive gradients through the frozen next block, while
the next block and teacher remain frozen.
"""
import copy

import timm
import torch
import torch.nn.functional as F

from _util import run_tests
from psvit.model_utils import get_shape
from psvit.reconstruct import (
    _set_trainable_only_block, _student_prediction, teacher_features,
)


def test_reconstruction_gradient_routing():
    torch.manual_seed(0)
    student = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    shp = get_shape(student)
    t = shp.num_blocks - 2
    all_ids = torch.arange(shp.num_global_tokens)
    current_ids = torch.tensor([0, 3, 7, 11, 19])
    next_ids = torch.tensor([0])
    z_prev = torch.randn(1, shp.num_global_tokens, shp.embed_dim)

    _set_trainable_only_block(student, t)
    pred = _student_prediction(student, t, z_prev, current_ids, next_ids, all_ids)
    loss = F.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()

    current_grads = [p.grad for p in student.blocks[t].parameters()]
    assert any(g is not None and torch.isfinite(g).all() for g in current_grads)
    assert all(p.grad is None for p in student.blocks[t + 1].parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_teacher_target_is_next_block_output():
    """t의 pruning target은 block t가 아니라 정확히 block t+1 출력이어야 한다."""
    torch.manual_seed(1)
    teacher = timm.create_model("deit_tiny_patch16_224", pretrained=False).eval()
    shp = get_shape(teacher)
    t = shp.num_blocks - 2
    images = torch.randn(1, 3, 224, 224)
    next_ids = torch.tensor([0, 7])

    z_prev, target = teacher_features(teacher, images, t, next_ids)
    after_t = teacher.blocks[t](z_prev)
    expected = teacher.blocks[t + 1](after_t).index_select(1, next_ids)
    wrong_layer = after_t.index_select(1, next_ids)

    torch.testing.assert_close(target, expected)
    assert not torch.allclose(target, wrong_layer)


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
