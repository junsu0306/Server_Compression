import torch

from _util import run_tests
from stage_policy_experiment import _rank_ids, transformer_macs


def test_rank_ids_preserves_required_and_count():
    scores = torch.tensor([0.0, 0.1, 0.9, 0.5, 0.3])
    got = _rank_ids(scores, torch.tensor([0, 3]), 4)
    assert got.tolist() == [0, 2, 3, 4]


def test_rank_ids_rejects_too_small_count():
    try:
        _rank_ids(torch.ones(5), torch.tensor([0, 2, 4]), 2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    run_tests(globals())
