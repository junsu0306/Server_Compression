"""§16.2 — Global/Local ID 매핑 + nested mask invariant 테스트."""
import torch
from _util import run_tests

from psvit.ids import (global_to_local, validate_nested_masks, assert_sorted_unique,
                       sorted_union, CLS_ID)


def test_global_to_local_subset():
    active = torch.tensor([0, 3, 7, 10])
    req = torch.tensor([0, 7])
    local = global_to_local(active, req)
    assert local.tolist() == [0, 2], local.tolist()


def test_global_to_local_missing_raises():
    active = torch.tensor([0, 3, 7])
    try:
        global_to_local(active, torch.tensor([0, 5]))     # 5는 active에 없음
    except ValueError:
        return
    raise AssertionError("없는 ID인데 예외 안 남")


def test_duplicate_reject():
    try:
        assert_sorted_unique(torch.tensor([0, 3, 3, 7]))
    except ValueError:
        return
    raise AssertionError("중복인데 예외 안 남")


def test_unsorted_reject():
    try:
        assert_sorted_unique(torch.tensor([0, 7, 3]))
    except ValueError:
        return
    raise AssertionError("비정렬인데 예외 안 남")


def test_nested_ok():
    keep = [torch.tensor([0, 1, 2, 3, 5]), torch.tensor([0, 2, 5]), torch.tensor([0])]
    validate_nested_masks(keep)                            # 예외 없어야 함


def test_nested_violation():
    keep = [torch.tensor([0, 1, 3]), torch.tensor([0, 2]), torch.tensor([0])]  # 2 ⊄ {0,1,3}
    try:
        validate_nested_masks(keep)
    except ValueError:
        return
    raise AssertionError("nested 위반인데 예외 안 남")


def test_cls_missing():
    keep = [torch.tensor([1, 2, 3]), torch.tensor([2])]    # CLS(0) 없음
    try:
        validate_nested_masks(keep)
    except ValueError:
        return
    raise AssertionError("CLS 누락인데 예외 안 남")


def test_sorted_union():
    base = torch.tensor([0, 5, 9])
    extra = torch.tensor([3, 5, 1])
    u = sorted_union(base, extra)
    assert u.tolist() == [0, 1, 3, 5, 9], u.tolist()


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
