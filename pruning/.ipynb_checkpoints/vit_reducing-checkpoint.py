"""
Soft Pruning 완료 모델의 zero 채널을 물리적으로 제거해 Dense 모델로 변환.

흐름:
  1. fc1.weight 행(row) 중 L2 norm == 0 인 채널 = Soft Pruning으로 수렴된 dead 채널
  2. 해당 행/열을 제거한 새 Linear 레이어로 in-place 교체
  3. 결과: fc1 (mlp_dim, embed_dim) → (n_survived, embed_dim)
           fc2 (embed_dim, mlp_dim) → (embed_dim, n_survived)

사용법:
    from pruning.vit_reducing import reduce_vit_model, get_reduced_config

    reduce_vit_model(model)               # in-place
    mlp_dims = get_reduced_config(model)  # 저장용 블록별 new mlp_dim
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _survived_idx(weight: torch.Tensor) -> torch.Tensor:
    """fc1.weight 행(row) 기준 L2 norm이 0이 아닌 인덱스(오름차순) 반환.

    Soft Pruning은 정확히 0으로 마스킹하므로 임계값 없이 == 0 판정이 안전하다.
    """
    n = weight.shape[0]
    norms = torch.norm(weight.reshape(n, -1), dim=1)
    return torch.nonzero(norms != 0, as_tuple=False).flatten()


@torch.no_grad()
def _reduce_ffn(mlp: nn.Module) -> int:
    """FFN 블록 하나를 survived 채널만으로 in-place 교체.

    반환: 제거된 파라미터 수 (0이면 변화 없음)
    """
    survived = _survived_idx(mlp.fc1.weight)
    n_new = survived.numel()
    n_old = mlp.fc1.weight.shape[0]  # 원본 mlp_dim

    if n_new == n_old:
        return 0

    embed_dim = mlp.fc2.weight.shape[0]  # 출력 = embed_dim (고정)
    dev   = mlp.fc1.weight.device
    dtype = mlp.fc1.weight.dtype

    # fc1: (n_old, embed_dim) → (n_new, embed_dim)
    new_fc1 = nn.Linear(
        mlp.fc1.in_features, n_new,
        bias=(mlp.fc1.bias is not None),
    ).to(dev, dtype)
    new_fc1.weight.data.copy_(mlp.fc1.weight.data[survived])
    if mlp.fc1.bias is not None:
        new_fc1.bias.data.copy_(mlp.fc1.bias.data[survived])
    mlp.fc1 = new_fc1

    # fc2: (embed_dim, n_old) → (embed_dim, n_new)
    new_fc2 = nn.Linear(
        n_new, embed_dim,
        bias=(mlp.fc2.bias is not None),
    ).to(dev, dtype)
    new_fc2.weight.data.copy_(mlp.fc2.weight.data[:, survived])
    if mlp.fc2.bias is not None:
        # fc2.bias shape = (embed_dim,) — 출력 소속이므로 그대로 복사
        new_fc2.bias.data.copy_(mlp.fc2.bias.data)
    mlp.fc2 = new_fc2

    # 제거된 파라미터 수 (secondary effect 포함)
    n_pruned  = n_old - n_new
    removed   = n_pruned * mlp.fc1.in_features   # fc1 weight 행
    removed  += n_pruned                           # fc1 bias (있을 경우)
    removed  += embed_dim * n_pruned               # fc2 weight 열
    return removed


# ── 공개 API ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def reduce_vit_model(model: nn.Module) -> nn.Module:
    """모델 전체 블록을 in-place로 Dense reduce.

    Soft Pruning 학습 완료 후 호출. 동일 객체를 반환한다.

    주의: EMA weights를 로드한 상태에서 호출해야 실제 성능이 보존된다.
    raw network weight는 매 step 0으로 리셋되어 있어 Reducing만 하면 dead 모델이 된다.
    """
    total_removed = 0
    for i, block in enumerate(model.blocks):
        removed = _reduce_ffn(block.mlp)
        total_removed += removed

    n_survived = sum(block.mlp.fc1.out_features for block in model.blocks)
    n_total_mlp = sum(
        block.mlp.fc1.out_features + (block.mlp.fc1.out_features - block.mlp.fc1.out_features)
        for block in model.blocks
    )
    print(f"[Reducer] removed {total_removed:,} params across {len(model.blocks)} blocks")
    return model


def get_reduced_config(model: nn.Module) -> list[int]:
    """Reducing 후 각 블록의 new mlp_dim 리스트 반환.

    체크포인트에 함께 저장해두면 나중에 구조를 재현할 수 있다.
    """
    return [block.mlp.fc1.out_features for block in model.blocks]


@torch.no_grad()
def transfer_pruning_mask(raw_model: nn.Module, ema_model: nn.Module) -> None:
    """raw model의 zero 채널 패턴을 ema_model에 이식.

    Soft Pruning 후:
    - raw model: dead 채널이 정확히 0  (매 step pruner.apply()로 리셋됨)
    - ema model: dead 채널이 0에 수렴하지만 부동소수점 상 정확히 0이 아님
                 (EMA decay^N 이 초기값에 곱해진 상태)

    reduce_vit_model() 호출 전에 이 함수를 먼저 실행하면
    EMA weights를 그대로 사용하면서 dead 채널을 정확히 0으로 만들 수 있다.
    """
    for raw_blk, ema_blk in zip(raw_model.blocks, ema_model.blocks):
        raw_norms = torch.norm(
            raw_blk.mlp.fc1.weight.detach().reshape(raw_blk.mlp.fc1.weight.shape[0], -1),
            dim=1,
        )
        dead = raw_norms == 0
        if not dead.any():
            continue
        ema_blk.mlp.fc1.weight.data[dead] = 0.0
        if ema_blk.mlp.fc1.bias is not None:
            ema_blk.mlp.fc1.bias.data[dead] = 0.0
        ema_blk.mlp.fc2.weight.data[:, dead] = 0.0


def apply_reduced_config(model: nn.Module, mlp_dims: list[int]) -> nn.Module:
    """저장된 mlp_dims로 timm 원본 모델의 구조를 축소.

    reduce.py 에서 저장한 reduced.pt 를 다시 로드할 때 사용:
        model = timm.create_model(name, pretrained=False)
        apply_reduced_config(model, ckpt['mlp_dims'])
        model.load_state_dict(ckpt['state_dict'], strict=True)
    """
    for block, new_dim in zip(model.blocks, mlp_dims):
        mlp     = block.mlp
        old_dim = mlp.fc1.out_features
        if new_dim == old_dim:
            continue

        embed_dim = mlp.fc2.weight.shape[0]
        dev   = mlp.fc1.weight.device
        dtype = mlp.fc1.weight.dtype

        mlp.fc1 = nn.Linear(
            mlp.fc1.in_features, new_dim,
            bias=(mlp.fc1.bias is not None),
        ).to(dev, dtype)
        mlp.fc2 = nn.Linear(
            new_dim, embed_dim,
            bias=(mlp.fc2.bias is not None),
        ).to(dev, dtype)

    return model
