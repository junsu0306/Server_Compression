"""
timm ViT / DeiT Soft Pruning

사용법:
    pruner = ViTPruner(model, target_compression=0.30)

    for epoch in range(epochs):
        for samples, targets in loader:
            ...
            optimizer.step()
            pruner.apply(model)   # ← optimizer.step() 직후, model_ema.update() 이전
            if model_ema: model_ema.update(model)

    metrics = pruner.log_sparsity(model)  # WandB 로깅용
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple


# ── 상수 ───────────────────────────────────────────────────────────────────────

MIN_SURVIVE = 4  # 어느 그룹이든 최소 4채널은 보존


# ── 수치 헬퍼 ──────────────────────────────────────────────────────────────────

def _calc_n_prune(n_total: int, sparsity: float) -> int:
    """sparsity 비율로 제거할 채널 수. MIN_SURVIVE 보장.

    int() 대신 round() 사용 — 소규모 레이어에서 int()는 하향 편향 발생.
    예: round(16 * 0.30) = 5,  int(16 * 0.30) = 4
    """
    n_prune = round(n_total * sparsity)
    n_prune = min(n_prune, n_total - MIN_SURVIVE)
    return max(n_prune, 0)


def _topk_smallest_l2_idx(weight: torch.Tensor, k: int) -> torch.Tensor:
    """첫 차원(out_features) 기준 L2 norm 하위 k개 인덱스 반환."""
    n = weight.shape[0]
    norms = torch.norm(weight.detach().reshape(n, -1), dim=1)
    _, idx = torch.topk(norms, k, largest=False)
    return idx


# ── Sparsity 이진탐색 ──────────────────────────────────────────────────────────

def _estimate_removed_ffn(mlp: nn.Module, sparsity: float) -> int:
    """FFN 블록 하나에서 sparsity로 제거되는 파라미터 수.

    fc2 입력 열도 같이 제거되는 secondary effect 포함.
    """
    fc1_w   = mlp.fc1.weight          # (mlp_dim, embed_dim)
    mlp_dim  = fc1_w.shape[0]
    embed_dim = fc1_w.shape[1]
    fc2_out  = mlp.fc2.weight.shape[0]  # = embed_dim

    n_prune = _calc_n_prune(mlp_dim, sparsity)
    if n_prune <= 0:
        return 0

    removed  = n_prune * embed_dim   # fc1 weight 행
    if mlp.fc1.bias is not None:
        removed += n_prune           # fc1 bias
    removed += fc2_out * n_prune     # fc2 weight 열 (secondary effect)
    # fc2.bias 는 embed_dim 소속이므로 제거 없음
    return removed


def _estimate_total_removed(model: nn.Module, sparsity: float) -> int:
    return sum(_estimate_removed_ffn(block.mlp, sparsity) for block in model.blocks)


def _find_sparsity_by_bisection(
    model: nn.Module,
    target_compression: float,
    max_sparsity: float = 0.95,
    iters: int = 64,
) -> float:
    """target_compression을 달성하는 per-group sparsity를 이진탐색으로 계산."""
    if target_compression <= 0:
        return 0.0
    total_params  = sum(p.numel() for p in model.parameters())
    target_remove = target_compression * total_params
    lo, hi = 0.0, max_sparsity
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── PruneGroup: 캐시 기반 벡터화 마스킹 ────────────────────────────────────────

@dataclass
class _PruneGroup:
    """하나의 FFN 블록에 대한 마스킹 상태를 보관.

    매 step model.modules()를 순회하는 CPU 병목을 피하기 위해
    첫 apply() 호출 시 텐서 레퍼런스를 캐시하고 이후에는 재사용한다.

    targets: (tensor, dim, fill_value) 리스트
        - dim=0: 행(row) 마스킹  (fc1.weight, fc1.bias)
        - dim=1: 열(col) 마스킹  (fc2.weight)
        - fill_value=0.0: 마스킹 시 채울 값
    """
    criterion: torch.Tensor                           # 랭킹 기준 (fc1.weight)
    sparsity:  float
    targets:   List[Tuple[torch.Tensor, int, float]]  # (tensor, dim, fill)
    _mask:     torch.Tensor | None = field(default=None, repr=False)

    def refresh(self) -> None:
        """L2 norm 기반 마스크 재계산. index_refresh_steps 마다 호출."""
        n = self.criterion.shape[0]
        n_prune = _calc_n_prune(n, self.sparsity)
        mask = torch.ones(n, dtype=torch.float32, device=self.criterion.device)
        if n_prune > 0:
            norms = torch.norm(self.criterion.detach().reshape(n, -1), dim=1)
            _, idx = torch.topk(norms, n_prune, largest=False)
            mask[idx] = 0.0
        self._mask = mask

    @torch.no_grad()
    def apply(self) -> None:
        """캐시된 마스크로 모든 target 텐서를 벡터화 마스킹."""
        if self._mask is None:
            return
        mask = self._mask
        for tensor, dim, fill in self.targets:
            if tensor is None:
                continue
            # dim 위치에 mask를 브로드캐스팅
            shape = [1] * tensor.dim()
            shape[dim] = -1
            m = mask.view(shape)
            if fill == 0.0:
                tensor.data.mul_(m)
            else:
                # fill != 0 인 경우: pruned 위치 → fill, alive 위치 → 현재 값 유지
                tensor.data.mul_(m).add_(fill * (1.0 - m))


def _make_ffn_group(mlp: nn.Module, sparsity: float) -> _PruneGroup:
    """FFN 블록 하나에 대한 _PruneGroup 생성.

    fc1.weight 행(dim=0)과 fc2.weight 열(dim=1)을 동일 인덱스로 묶는다.
    """
    targets: List[Tuple[torch.Tensor, int, float]] = []

    # fc1 출력 행 (dim=0)
    targets.append((mlp.fc1.weight, 0, 0.0))
    if mlp.fc1.bias is not None:
        targets.append((mlp.fc1.bias, 0, 0.0))

    # fc2 입력 열 (dim=1) ← fc1과 반드시 동일 인덱스
    targets.append((mlp.fc2.weight, 1, 0.0))
    # fc2.bias 는 embed_dim 소속이므로 포함하지 않음

    return _PruneGroup(criterion=mlp.fc1.weight, sparsity=sparsity, targets=targets)


# ── ViTPruner ──────────────────────────────────────────────────────────────────

class ViTPruner:
    """timm ViT / DeiT FFN Soft Pruning 컨트롤러.

    Args:
        model:              timm ViT 또는 DeiT 모델 (model.blocks 접근 가능해야 함)
        target_compression: 목표 파라미터 압축률 (0.0 ~ 1.0). 0이면 비활성.
        max_sparsity:       per-group sparsity 상한 (기본 0.95)
        sparsity:           직접 지정 시 이진탐색 생략. None이면 이진탐색으로 결정.
        index_refresh_steps: 마스크 재계산 간격 (step 단위).
                             0 또는 음수이면 매 step 재계산.
    """

    def __init__(
        self,
        model: nn.Module,
        target_compression: float,
        max_sparsity: float = 0.95,
        sparsity: float | None = None,
        index_refresh_steps: int = 100,
    ) -> None:
        self.target_compression  = float(target_compression)
        self.max_sparsity        = float(max_sparsity)
        self.index_refresh_steps = int(index_refresh_steps)
        self._step  = 0
        self._groups: list[_PruneGroup] = []  # lazy init (첫 apply() 시 수집)

        if sparsity is not None:
            self.sparsity = float(sparsity)
        else:
            self.sparsity = _find_sparsity_by_bisection(
                model, self.target_compression, self.max_sparsity
            )

        total = sum(p.numel() for p in model.parameters())
        est   = _estimate_total_removed(model, self.sparsity)
        rate  = 100.0 * est / max(total, 1)
        print(
            f"[ViTPruner] target={self.target_compression*100:.1f}%  "
            f"sparsity={self.sparsity:.4f}  "
            f"estimated_compression={rate:.2f}%  "
            f"blocks={len(model.blocks)}"
        )

    # ── 내부 ────────────────────────────────────────────────────────────────────

    def _collect_groups(self, model: nn.Module) -> list[_PruneGroup]:
        return [_make_ffn_group(block.mlp, self.sparsity) for block in model.blocks]

    def _need_refresh(self) -> bool:
        if self.index_refresh_steps <= 0:
            return True
        return self._step % self.index_refresh_steps == 0

    # ── 공개 API ────────────────────────────────────────────────────────────────

    def apply(self, model: nn.Module) -> None:
        """optimizer.step() 직후, model_ema.update() 이전에 호출.

        DDP 환경에서는 model.module 을 넘겨야 한다:
            actual = model.module if hasattr(model, 'module') else model
            pruner.apply(actual)
        """
        if self.sparsity <= 0:
            return

        # Lazy init: 첫 호출 시 모델이 이미 CUDA에 있을 때 그룹을 수집한다.
        # __init__ 시점에 수집하면 CPU 텐서 레퍼런스가 cached 되어 device mismatch 발생.
        if not self._groups:
            self._groups = self._collect_groups(model)
            for g in self._groups:
                g.refresh()
            self._step += 1
            return

        if self._need_refresh():
            for g in self._groups:
                g.refresh()

        for g in self._groups:
            g.apply()

        self._step += 1

    @torch.no_grad()
    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """실제 zero 비율을 계산해 반환. WandB 로깅에 사용.

        반환 키:
            pruning/actual_sparsity   — 전체 prunable 채널 중 zero 비율
            pruning/zero_filters      — zero 채널 수 (절대값)
            pruning/prunable_filters  — 전체 prunable 채널 수
            pruning/target_sparsity   — 설정된 목표 sparsity
            pruning/layer/<block>     — 블록별 zero 비율
        """
        n_total, n_zero = 0, 0
        result: dict[str, float] = {}

        for name, module in model.named_modules():
            # fc1, fc2 를 직접 갖는 MLP 모듈만 대상
            if not (hasattr(module, "fc1") and hasattr(module, "fc2")):
                continue
            if hasattr(module, "mlp"):
                # 블록 컨테이너 자체는 건너뜀, mlp 서브모듈만 처리
                continue

            w = module.fc1.weight       # (mlp_dim, embed_dim)
            n = w.shape[0]
            norms = torch.norm(w.detach().reshape(n, -1), dim=1)
            n_z   = int((norms == 0).sum().item())

            n_total += n
            n_zero  += n_z

            safe_name = name.replace(".", "/")
            result[f"pruning/layer/{safe_name}"] = n_z / max(n, 1)

        result["pruning/zero_filters"]     = float(n_zero)
        result["pruning/prunable_filters"] = float(n_total)
        result["pruning/actual_sparsity"]  = n_zero / max(n_total, 1)
        result["pruning/target_sparsity"]  = self.sparsity
        return result

    def state_dict(self) -> dict:
        """체크포인트 저장용."""
        return {
            "sparsity":            self.sparsity,
            "target_compression":  self.target_compression,
            "step":                self._step,
            "index_refresh_steps": self.index_refresh_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """체크포인트 복원용."""
        self.sparsity            = state["sparsity"]
        self.target_compression  = state["target_compression"]
        self._step               = state["step"]
        self.index_refresh_steps = state["index_refresh_steps"]
        self._groups             = []  # 다음 apply()에서 재수집
