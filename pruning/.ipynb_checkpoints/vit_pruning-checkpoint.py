"""
timm ViT / DeiT FFN Soft Pruning

pruning_mode:
    "uniform" — 모든 블록에 동일한 sparsity 적용 (기존 방식)
    "global"  — 전체 블록 채널을 global 랭킹, 자동 non-uniform 분배 (기본값)

importance:
    "l2"     — fc1.weight 행벡터의 L2 norm (weight magnitude)
    "taylor" — |fc1.weight × ∇fc1.weight|의 채널합 (gradient × weight)
               backprop 후 param.grad를 재활용하므로 추가 연산 없음.
               grad가 없는 첫 step은 L2로 fallback.

사용법:
    pruner = ViTPruner(model, target_compression=0.50, mode="global", importance="taylor")

    for epoch in range(epochs):
        for samples, targets in loader:
            ...
            optimizer.step()
            pruner.apply(model)   # optimizer.step() 직후, model_ema.update() 이전
            if model_ema: model_ema.update(model)

    metrics = pruner.log_sparsity(model)  # WandB 로깅용
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple


# ── 상수 ───────────────────────────────────────────────────────────────────────

MIN_SURVIVE = 4  # 블록당 최소 보존 채널 수


# ── 수치 헬퍼 ──────────────────────────────────────────────────────────────────

def _calc_n_prune(n_total: int, sparsity: float) -> int:
    """sparsity 비율로 제거할 채널 수. MIN_SURVIVE 보장.

    int() 대신 round() — 소규모 레이어에서 int()는 하향 편향 발생.
    """
    n_prune = round(n_total * sparsity)
    n_prune = min(n_prune, n_total - MIN_SURVIVE)
    return max(n_prune, 0)


# ── Sparsity 이진탐색 ──────────────────────────────────────────────────────────

def _estimate_removed_ffn(mlp: nn.Module, sparsity: float) -> int:
    """FFN 블록 하나에서 sparsity로 제거되는 파라미터 수 (secondary effect 포함)."""
    fc1_w    = mlp.fc1.weight
    mlp_dim  = fc1_w.shape[0]
    embed_dim = fc1_w.shape[1]
    fc2_out  = mlp.fc2.weight.shape[0]

    n_prune = _calc_n_prune(mlp_dim, sparsity)
    if n_prune <= 0:
        return 0

    removed = n_prune * embed_dim
    if mlp.fc1.bias is not None:
        removed += n_prune
    removed += fc2_out * n_prune
    return removed


def _estimate_total_removed(model: nn.Module, sparsity: float) -> int:
    return sum(_estimate_removed_ffn(block.mlp, sparsity) for block in model.blocks)


def _find_sparsity_by_bisection(
    model: nn.Module,
    target_compression: float,
    max_sparsity: float = 0.95,
    iters: int = 64,
) -> float:
    """target_compression을 달성하는 equivalent sparsity를 이진탐색으로 계산.

    global mode에서도 전체 제거 채널 수의 기준값으로 사용된다.
    """
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


# ── PruneGroup ─────────────────────────────────────────────────────────────────

@dataclass
class _PruneGroup:
    """하나의 FFN 블록에 대한 마스킹 상태.

    targets: (tensor, dim, fill_value) 리스트
        dim=0 → 행 마스킹 (fc1.weight, fc1.bias)
        dim=1 → 열 마스킹 (fc2.weight)
    """
    criterion: torch.Tensor                           # 랭킹 기준 텐서 (fc1.weight)
    targets:   List[Tuple[torch.Tensor, int, float]]  # (tensor, dim, fill)
    _mask:     torch.Tensor | None = field(default=None, repr=False)

    def refresh(self, sparsity: float, scores: torch.Tensor | None = None) -> None:
        """Per-block 중요도 기반 마스크 재계산 (uniform mode 전용).

        scores: 외부에서 미리 계산된 채널 중요도 (낮을수록 제거). None이면 L2 fallback.
        """
        n = self.criterion.shape[0]
        n_prune = _calc_n_prune(n, sparsity)
        mask = torch.ones(n, dtype=torch.float32, device=self.criterion.device)
        if n_prune > 0:
            if scores is None:
                scores = torch.norm(self.criterion.detach().reshape(n, -1), dim=1)
            _, idx = torch.topk(scores, n_prune, largest=False)
            mask[idx] = 0.0
        self._mask = mask

    def set_mask(self, mask: torch.Tensor) -> None:
        """Global ranking에서 외부 계산된 mask를 주입 (global mode 전용)."""
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
            shape = [1] * tensor.dim()
            shape[dim] = -1
            m = mask.view(shape)
            if fill == 0.0:
                tensor.data.mul_(m)
            else:
                tensor.data.mul_(m).add_(fill * (1.0 - m))


def _make_ffn_group(mlp: nn.Module) -> _PruneGroup:
    """FFN 블록 하나에 대한 _PruneGroup 생성."""
    targets: List[Tuple[torch.Tensor, int, float]] = []
    targets.append((mlp.fc1.weight, 0, 0.0))
    if mlp.fc1.bias is not None:
        targets.append((mlp.fc1.bias, 0, 0.0))
    targets.append((mlp.fc2.weight, 1, 0.0))
    return _PruneGroup(criterion=mlp.fc1.weight, targets=targets)


# ── ViTPruner ──────────────────────────────────────────────────────────────────

class ViTPruner:
    """timm ViT / DeiT FFN Soft Pruning 컨트롤러.

    Args:
        model:               timm ViT 모델 (model.blocks 접근 가능)
        target_compression:  목표 파라미터 압축률 (0.0 ~ 1.0)
        max_sparsity:        sparsity 상한 (기본 0.95)
        sparsity:            직접 지정 시 이진탐색 생략
        index_refresh_steps: 마스크 재계산 간격 (step 단위). 0=매 step.
        mode:                "global" (non-uniform, 기본값) | "uniform"
        importance:          "l2" (weight magnitude) | "taylor" (gradient × weight)
        grad_ema_beta:       taylor EMA 감쇠율 (기본 0.9 ≈ 최근 10 step 평균)

    mode 비교:
        uniform — 각 블록 독립적으로 하위 sparsity% 채널 제거
                  모든 블록이 동일한 비율로 잘림
        global  — 전체 블록의 채널을 한번에 global 랭킹
                  중요도가 높은 블록은 채널을 더 많이 보존
                  중복이 많은 블록은 더 많이 잘림 (자동 non-uniform)

    importance 비교:
        l2     — ||fc1.weight||₂ 채널 크기만 봄. 빠르고 안정적.
        taylor — |fc1.weight × ∇fc1.weight|의 채널합을 grad_ema_beta EMA로 누적.
                 backprop gradient를 재활용하므로 추가 forward/backward 불필요.
                 EMA 덕분에 single-batch gradient noise를 평균화.
                 grad가 한 번도 없으면 l2로 fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        target_compression: float,
        max_sparsity: float = 0.95,
        sparsity: float | None = None,
        index_refresh_steps: int = 100,
        mode: str = "global",
        warmup_epochs: int = 0,
        ramp_epochs: int = 0,
        importance: str = "l2",
        grad_ema_beta: float = 0.9,
    ) -> None:
        self.target_compression  = float(target_compression)
        self.max_sparsity        = float(max_sparsity)
        self.index_refresh_steps = int(index_refresh_steps)
        self.mode                = mode
        self.importance          = importance
        self.grad_ema_beta       = float(grad_ema_beta)
        self._grad_ema:  dict[int, torch.Tensor] = {}  # id(weight) → EMA Taylor score
        self._step               = 0
        self._groups: list[_PruneGroup] = []  # lazy init
        self._warmup_epochs      = int(warmup_epochs)
        self._ramp_epochs        = int(ramp_epochs)
        self._current_epoch      = 0

        if sparsity is not None:
            self._target_sparsity = float(sparsity)
        else:
            self._target_sparsity = _find_sparsity_by_bisection(
                model, self.target_compression, self.max_sparsity
            )

        # Progressive mode: epoch 0에서는 sparsity=0으로 시작
        progressive = (warmup_epochs > 0 or ramp_epochs > 0)
        self.sparsity = 0.0 if progressive else self._target_sparsity

        total = sum(p.numel() for p in model.parameters())
        est   = _estimate_total_removed(model, self._target_sparsity)
        rate  = 100.0 * est / max(total, 1)
        n_prune_total = sum(
            _calc_n_prune(block.mlp.fc1.weight.shape[0], self._target_sparsity)
            for block in model.blocks
        )
        imp_str = self.importance
        if self.importance == "taylor":
            imp_str += f"(ema_beta={self.grad_ema_beta})"
        print(
            f"[ViTPruner] target={self.target_compression*100:.1f}%  "
            f"equiv_sparsity={self._target_sparsity:.4f}  "
            f"estimated_compression={rate:.2f}%  "
            f"total_prune_channels={n_prune_total}  "
            f"mode={self.mode}  importance={imp_str}"
        )
        if progressive:
            print(
                f"[ViTPruner] progressive=ON  "
                f"warmup={self._warmup_epochs}  ramp={self._ramp_epochs}  "
                f"(epoch {self._warmup_epochs}~"
                f"{self._warmup_epochs + self._ramp_epochs}에서 점진적 증가)"
            )

    # ── 내부 ────────────────────────────────────────────────────────────────────

    def _channel_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """fc1.weight 행(=출력 채널)별 중요도 계산.

        taylor: |w × ∇w|의 채널합을 EMA로 누적. Single-batch noise 평균화.
                grad가 한 번도 없으면 l2로 fallback.
                grad가 있으면 EMA를 업데이트하고 EMA 값을 반환.
        l2:     ||w||₂ 채널 크기.

        AMP scale은 전체 채널에 공통 적용되므로 상대 랭킹에 영향 없음.
        """
        n = weight.shape[0]
        w = weight.detach().reshape(n, -1)

        if self.importance == "taylor":
            pid = id(weight)
            if weight.grad is not None:
                g = weight.grad.detach().reshape(n, -1)
                taylor_now = (w * g).abs().sum(dim=1)
                if pid not in self._grad_ema:
                    self._grad_ema[pid] = taylor_now
                else:
                    b = self.grad_ema_beta
                    self._grad_ema[pid] = b * self._grad_ema[pid] + (1.0 - b) * taylor_now
            if pid in self._grad_ema:
                return self._grad_ema[pid]
            # grad 한 번도 없음 → L2 fallback

        return torch.norm(w, dim=1)

    def _collect_groups(self, model: nn.Module) -> list[_PruneGroup]:
        return [_make_ffn_group(block.mlp) for block in model.blocks]

    def _need_refresh(self) -> bool:
        if self.index_refresh_steps <= 0:
            return True
        return self._step % self.index_refresh_steps == 0

    def _do_refresh(self) -> None:
        if self.mode == "global":
            self._refresh_global()
        else:
            for g in self._groups:
                scores = self._channel_importance(g.criterion)
                g.refresh(self.sparsity, scores=scores)

    def _refresh_global(self) -> None:
        """Global channel ranking — non-uniform pruning.

        전체 블록의 fc1.weight row L2 norm을 한번에 비교하여
        norm이 작은 채널(중요도 낮음)부터 전체에서 N개 제거.

        per-block 상한(max_sparsity):
            각 블록에서 제거 가능한 최대 채널 수를 cap으로 제한.
            cap에 걸린 블록의 초과분은 여유 있는 다른 블록에서 추가 제거.
            → 특정 블록이 과도하게 파괴되는 것을 방지하면서 자동 non-uniform.
        """
        device = self._groups[0].criterion.device

        norms_list = [
            self._channel_importance(g.criterion)
            for g in self._groups
        ]
        group_sizes = [norms.shape[0] for norms in norms_list]

        # 블록당 제거 가능한 최대 채널 수
        max_prune_per_block = [
            min(_calc_n_prune(n, self.max_sparsity), n - MIN_SURVIVE)
            for n in group_sizes
        ]

        n_total_prune = sum(_calc_n_prune(n, self.sparsity) for n in group_sizes)

        masks = [
            torch.ones(n, dtype=torch.float32, device=device)
            for n in group_sizes
        ]

        if n_total_prune > 0:
            # ── 핵심 아이디어: 블록별 cap 초과 채널은 norm을 inf로 올려 선택 불가 처리 ──
            # 각 블록에서 local 순위로 max_prune_per_block[b] 번째 이후 채널은
            # 전역 선택에서 제외 (→ norm을 inf로 마킹)
            eligible_norms = torch.cat(norms_list).clone()
            cumsum = [0] + [sum(group_sizes[:i + 1]) for i in range(len(group_sizes))]

            for b, (start, max_p) in enumerate(zip(cumsum, max_prune_per_block)):
                n_b = group_sizes[b]
                if max_p < n_b:
                    # local 순위에서 max_p 번째 이후(= cap 초과) 채널은 선택 불가
                    local_sorted_asc = torch.argsort(norms_list[b])
                    ineligible_local = local_sorted_asc[max_p:]
                    eligible_norms[start + ineligible_local] = float("inf")

            # eligible 채널 중 global 하위 n_total_prune개 선택
            _, global_prune_idx = torch.topk(eligible_norms, n_total_prune, largest=False)

            # global index → 블록별 mask
            cumsum_t = torch.tensor(cumsum, device=device)
            for b in range(len(self._groups)):
                in_block = (global_prune_idx >= cumsum_t[b]) & (
                    global_prune_idx < cumsum_t[b + 1]
                )
                local_idx = global_prune_idx[in_block] - cumsum_t[b]
                masks[b][local_idx] = 0.0

        # MIN_SURVIVE 최종 보장 후 mask 설정
        # (중요도 높은 채널 상위 MIN_SURVIVE개 보존)
        for g, mask, scores in zip(self._groups, masks, norms_list):
            if int(mask.sum().item()) < MIN_SURVIVE:
                n = scores.shape[0]
                mask = torch.zeros(n, dtype=torch.float32, device=device)
                mask[torch.argsort(scores, descending=True)[:MIN_SURVIVE]] = 1.0
            g.set_mask(mask)

    # ── Progressive sparsity 스케줄 ────────────────────────────────────────────

    def _scheduled_sparsity(self, epoch: int) -> float:
        """Zhu & Gupta (2018) 스타일 cubic schedule.

        epoch < warmup_epochs:          sparsity = 0
        warmup <= epoch < warmup+ramp:  0 → target  (cubic ease-out)
        epoch >= warmup+ramp:           sparsity = target
        """
        if self._ramp_epochs == 0 and self._warmup_epochs == 0:
            return self._target_sparsity
        if epoch < self._warmup_epochs:
            return 0.0
        ramp_end = self._warmup_epochs + self._ramp_epochs
        if epoch >= ramp_end:
            return self._target_sparsity
        progress = (epoch - self._warmup_epochs) / max(self._ramp_epochs, 1)
        # cubic ease-out: 1-(1-t)^3 → 초반에 빠르게 증가, 후반에 완만
        return self._target_sparsity * (1.0 - (1.0 - progress) ** 3)

    def set_epoch(self, epoch: int) -> None:
        """에포크 시작 전에 train.py에서 호출. sparsity 스케줄 업데이트."""
        self._current_epoch = epoch
        new_sp = self._scheduled_sparsity(epoch)
        if abs(new_sp - self.sparsity) > 1e-7:
            old_sp = self.sparsity
            self.sparsity = new_sp
            # 이미 init된 상태라면 즉시 마스크 재계산
            if self._groups:
                self._do_refresh()
            print(
                f"[ViTPruner] epoch={epoch}  "
                f"sparsity: {old_sp:.4f} → {new_sp:.4f}"
                f"  ({new_sp / self._target_sparsity * 100:.1f}% of target)"
            )

    # ── 공개 API ────────────────────────────────────────────────────────────────

    def apply(self, model: nn.Module) -> None:
        """optimizer.step() 직후, model_ema.update() 이전에 호출.

        DDP 환경: pruner.apply(model.module)
        """
        if self.sparsity <= 0:
            return

        # Lazy init: 첫 apply() 시 모델이 이미 CUDA에 있을 때 그룹 수집
        if not self._groups:
            self._groups = self._collect_groups(model)
            self._do_refresh()
            self._step += 1
            return

        if self._need_refresh():
            self._do_refresh()

        for g in self._groups:
            g.apply()

        self._step += 1

    @torch.no_grad()
    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """실제 zero 비율 계산 및 블록별 survived 채널 수 반환."""
        n_total, n_zero = 0, 0
        result: dict[str, float] = {}

        for name, module in model.named_modules():
            if not (hasattr(module, "fc1") and hasattr(module, "fc2")):
                continue
            if hasattr(module, "mlp"):
                continue

            w     = module.fc1.weight
            n     = w.shape[0]
            norms = torch.norm(w.detach().reshape(n, -1), dim=1)
            n_z   = int((norms == 0).sum().item())

            n_total += n
            n_zero  += n_z

            safe_name = name.replace(".", "/")
            result[f"pruning/layer/{safe_name}"]          = n_z / max(n, 1)
            result[f"pruning/survived/{safe_name}"]       = float(n - n_z)

        result["pruning/zero_filters"]     = float(n_zero)
        result["pruning/prunable_filters"] = float(n_total)
        result["pruning/actual_sparsity"]  = n_zero / max(n_total, 1)
        result["pruning/target_sparsity"]  = self._target_sparsity
        result["pruning/current_sparsity"] = self.sparsity
        return result

    def state_dict(self) -> dict:
        return {
            "sparsity":            self.sparsity,
            "target_sparsity":     self._target_sparsity,
            "target_compression":  self.target_compression,
            "step":                self._step,
            "index_refresh_steps": self.index_refresh_steps,
            "mode":                self.mode,
            "importance":          self.importance,
            "grad_ema_beta":       self.grad_ema_beta,
            "warmup_epochs":       self._warmup_epochs,
            "ramp_epochs":         self._ramp_epochs,
            "current_epoch":       self._current_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        self._target_sparsity    = state.get("target_sparsity", state["sparsity"])
        self.sparsity            = state["sparsity"]
        self.target_compression  = state["target_compression"]
        self._step               = state["step"]
        self.index_refresh_steps = state["index_refresh_steps"]
        self.mode                = state.get("mode", "global")
        self.importance          = state.get("importance", "l2")
        self.grad_ema_beta       = state.get("grad_ema_beta", 0.9)
        self._warmup_epochs      = state.get("warmup_epochs", 0)
        self._ramp_epochs        = state.get("ramp_epochs", 0)
        self._current_epoch      = state.get("current_epoch", 0)
        self._groups             = []
        self._grad_ema           = {}  # resume 시 EMA는 초기화 (첫 step에서 재누적)
