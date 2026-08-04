"""
§3 — Global / Local token ID 규칙과 mask invariant.

Global token ID: 원본 unpruned sequence에서의 영구 위치.
    표준 단일 CLS ViT (timm num_prefix_tokens=1): global id 0 = CLS,
    1..N_patch = image patch. 총 N = N_patch + 1.

Local token index: compact tensor 안에서의 현재 위치.

모든 mask/score/serialization은 global ID 기준으로 관리한다 (SPEC §3.2, §20).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor

CLS_ID = 0  # timm ViT는 CLS를 sequence 맨 앞(index 0)에 둔다.


def as_long_ids(ids: Sequence[int] | Tensor) -> Tensor:
    """정수 시퀀스/텐서를 1D int64 텐서로 정규화."""
    if isinstance(ids, Tensor):
        return ids.detach().to(torch.long).reshape(-1)
    return torch.tensor(list(ids), dtype=torch.long)


def assert_sorted_unique(ids: Tensor, name: str = "keep_ids") -> None:
    """오름차순 + 중복 없음 검증 (SPEC §3.4)."""
    ids = as_long_ids(ids)
    if ids.numel() == 0:
        raise ValueError(f"{name}: 비어 있음")
    uniq = torch.unique(ids)
    if uniq.numel() != ids.numel():
        raise ValueError(f"{name}: 중복 ID 존재")
    if not torch.equal(ids, torch.sort(ids).values):
        raise ValueError(f"{name}: 정렬되어 있지 않음")


def global_to_local(active_global_ids: Tensor, requested_global_ids: Tensor) -> Tensor:
    """requested_global_ids가 active_global_ids의 부분집합인지 검증하고 local index 반환.

    active_global_ids: [N_in]  현재 tensor에 실제로 존재하는 global ID (정렬·유일 가정)
    requested_global_ids: [N_out]  뽑아낼 global ID (active의 부분집합이어야 함)
    반환: [N_out]  active_global_ids 내에서의 위치(local index)

    예: active=[0,3,7,10], requested=[0,7] → local=[0,2]  (SPEC §3.3)
    """
    active = as_long_ids(active_global_ids)
    requested = as_long_ids(requested_global_ids)

    # active 내 위치를 조회할 lookup: global_id → local_index
    # active는 임의의 int이므로 dict 대신 searchsorted (active는 정렬 가정)
    assert_sorted_unique(active, "active_global_ids")
    pos = torch.searchsorted(active, requested)
    # searchsorted 결과가 실제로 그 값을 가리키는지(=부분집합인지) 확인
    if (pos >= active.numel()).any() or not torch.equal(active[pos.clamp(max=active.numel() - 1)], requested):
        missing = requested[~torch.isin(requested, active)]
        raise ValueError(
            f"requested_global_ids가 active_global_ids의 부분집합이 아님. "
            f"없는 ID: {missing.tolist()[:10]}"
        )
    return pos.to(torch.long)


def validate_nested_masks(keep_ids: List[Optional[Tensor]], upto: Optional[int] = None) -> None:
    """모든 layer에서 mask invariant 검증 (SPEC §3.4, §16.4).

        CLS_ID ∈ keep_ids[l]
        set(keep_ids[l+1]) ⊆ set(keep_ids[l])
        keep_ids[l] 은 정렬·유일

    upto: keep_ids[l]가 아직 None인 layer는 건너뜀 (search 진행 중 부분 검증용).
          upto가 주어지면 l >= upto 인 layer만 (이미 확정된 것) 검사.
    """
    L = len(keep_ids)
    for l in range(L):
        if keep_ids[l] is None:
            continue
        if upto is not None and l < upto:
            continue
        ids_l = as_long_ids(keep_ids[l])
        assert_sorted_unique(ids_l, f"keep_ids[{l}]")
        if CLS_ID not in ids_l.tolist():
            raise ValueError(f"keep_ids[{l}]: CLS_ID({CLS_ID}) 누락")
        if l + 1 < L and keep_ids[l + 1] is not None:
            child = set(as_long_ids(keep_ids[l + 1]).tolist())
            parent = set(ids_l.tolist())
            if not child.issubset(parent):
                extra = sorted(child - parent)
                raise ValueError(
                    f"nested mask 위반: keep_ids[{l+1}] ⊄ keep_ids[{l}]. "
                    f"자식에만 있는 ID: {extra[:10]}"
                )


def sorted_union(base_ids: Tensor, extra_ids: Tensor) -> Tensor:
    """base ∪ extra 를 정렬·유일 1D 텐서로 반환 (candidate mask 확장용, SPEC §8.5)."""
    both = torch.cat([as_long_ids(base_ids), as_long_ids(extra_ids)])
    return torch.unique(both)  # torch.unique는 정렬된 결과 반환
