"""§16.8 — Candidate ordering, architecture round-trip, and resume state tests."""
import tempfile
from types import SimpleNamespace

import torch

from _util import run_tests
from psvit.architecture import ArchitectureSpec, build_spec
from psvit.ids import sorted_union
from psvit.search import (
    _load_checkpoint, _save_checkpoint, initial_growth_count,
    rank_candidate_patch_ids,
)


def test_candidate_masks_grow_by_search_step():
    scores = torch.tensor([0.0, 0.1, 0.9, 0.2, 0.8, 0.3])
    base = torch.tensor([0])
    ranked = rank_candidate_patch_ids(scores, base, num_global=6)
    masks = [sorted_union(base, ranked[:r]) for r in (0, 2, 4, 5)]
    assert [m.numel() for m in masks] == [1, 3, 5, 6]
    for parent, child in zip(masks, masks[1:]):
        assert set(parent.tolist()).issubset(set(child.tolist()))


def test_candidate_ids_follow_score_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = torch.arange(6, dtype=torch.float32, device=device)
    ranked = rank_candidate_patch_ids(scores, torch.tensor([0], device=device), num_global=6)
    assert ranked.device == scores.device
    assert ranked.tolist() == [5, 4, 3, 2, 1]


def test_progressive_growth_skips_deepest_searchable_layer():
    cfg = {"min_growth_per_shallower_layer": 5}
    assert initial_growth_count(10, 10, 196, cfg) == 0
    assert initial_growth_count(9, 10, 76, cfg) == 5
    assert initial_growth_count(0, 10, 3, cfg) == 3
    assert initial_growth_count(9, 10, 76, {}) == 0


def test_architecture_round_trip():
    shp = SimpleNamespace(num_blocks=3, num_global_tokens=6, num_prefix_tokens=1)
    keep = [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1]), torch.tensor([0])]
    spec = build_spec("dummy_vit", shp, keep, [0.01, 0.02, 0.0],
                      epsilon=0.02, error_metric="mse", search_step=2,
                      score_mode="paper_path_energy_dp")
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/architecture.json"
        spec.to_json(path)
        loaded = ArchitectureSpec.from_json(path)
    loaded.validate()
    assert loaded == spec
    assert loaded.token_schedule() == [6, 4, 2, 1]


def test_search_checkpoint_resume_round_trip():
    model = torch.nn.Linear(3, 2)
    keep = [None, torch.tensor([0, 2]), torch.tensor([0])]
    errors = [float("nan"), 0.01, 0.0]
    records = [{"layer": 1, "r": 1, "accepted": True}]
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/search_last.pt"
        _save_checkpoint(path, model, keep, errors, records, next_layer=0)
        restored = torch.nn.Linear(3, 2)
        got_keep, got_errors, got_records, next_layer = _load_checkpoint(
            path, restored, torch.device("cpu")
        )
    assert next_layer == 0 and got_records == records
    assert got_errors[1:] == errors[1:]
    assert got_keep[0] is None
    assert torch.equal(got_keep[1], keep[1]) and torch.equal(got_keep[2], keep[2])
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
