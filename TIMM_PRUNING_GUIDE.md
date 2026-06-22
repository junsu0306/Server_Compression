# Soft Pruning → Reducing 적용 가이드 (timm ViT / DeiT)

> 본 문서는 EfficientViT 분류 모델에 적용한 Soft Pruning 방법론을  
> **timm 라이브러리의 ViT / DeiT 모델에 재적용**하기 위한 전달 문서다.  
> 새 레포에서 작업하는 담당자가 처음부터 끝까지 이 문서만으로 구현할 수 있도록  
> 의사결정 근거, 코드 패턴, 주의사항을 모두 포함한다.

---

## 목차

1. [방법론 개요](#1-방법론-개요)
2. [전체 파이프라인](#2-전체-파이프라인)
3. [Step 1 — 아키텍처 분석 & 메모리 프로파일링](#3-step-1--아키텍처-분석--메모리-프로파일링)
4. [Step 2 — Soft Pruning (Fine-tuning 중)](#4-step-2--soft-pruning-fine-tuning-중)
5. [Step 3 — Reducing (학습 완료 후)](#5-step-3--reducing-학습-완료-후)
6. [Sparsity 이진탐색](#6-sparsity-이진탐색)
7. [WandB 연동](#7-wandb-연동)
8. [훈련 루프 연동](#8-훈련-루프-연동)
9. [검증 체크리스트](#9-검증-체크리스트)
10. [흔한 실수 & 주의사항](#10-흔한-실수--주의사항)
11. [레퍼런스 구현 (EfficientViT) 요약](#11-레퍼런스-구현-efficientvit-요약)

---

## 1. 방법론 개요

### Soft Pruning이란

weight를 완전히 삭제하지 않고 **0으로 마스킹한 채로 학습을 계속**하는 방식이다.

```
[일반 학습]     forward → loss → backward → optimizer.step()

[Soft Pruning]  forward → loss → backward → optimizer.step()
                                                      ↓
                                    L2 norm 하위 X% weight를 0으로 리셋
                                                      ↓
                                    (다음 step에서 gradient로 살아날 수도 있음)
                                                      ↓
                                    다시 0으로 리셋 → 반복 수렴
```

**왜 이 방식인가:**

| 특성 | 설명 |
|------|------|
| **Soft** | 0으로 리셋되어도 gradient를 받으면 살아날 수 있다 → 진짜 중요한 채널만 살아남음 |
| **Dynamic** | 매 step 중요도를 재평가 → 초기에 잘못 제거된 채널이 복구 가능 |
| **Dense 유지** | 학습 중 아키텍처 구조가 변하지 않음 → 기존 학습 코드 재사용, DDP 호환 |
| **최소 침투** | `optimizer.step()` 직후 `pruner.apply(model)` 한 줄만 삽입 |

### Hard Pruning과의 차이

Hard Pruning은 학습 중 아키텍처를 물리적으로 변경하므로 복잡한 파이프라인이 필요하다.  
**Soft Pruning은 학습 파이프라인을 그대로 유지하고, 마지막에 한 번만 Reducing을 실행한다.**

---

## 2. 전체 파이프라인

```
[1] 아키텍처 분석 & 메모리 프로파일링
     - timm 모델 구조 출력 (named_modules)
     - 그룹별 파라미터 수 측정 (Prunable vs Non-prunable)
     - target_compression 대비 예상 sparsity 계산
          ↓
[2] Soft Pruning Fine-tuning
     - timm.create_model(pretrained=True) 로드
     - EfficientViTPruner → VitPruner 로 교체 (아키텍처별 구현)
     - 매 optimizer.step() 직후 pruner.apply(model)
     - WandB로 sparsity, loss, accuracy 추적
          ↓
[3] Checkpoint 저장
     - model_best.pt (원본 아키텍처 shape 그대로, 0인 채널 포함)
          ↓
[4] Reducing
     - Soft Pruning 완료 모델의 0 채널을 물리적으로 제거
     - 작고 빠른 Dense 모델 생성
          ↓
[5] 검증
     - Forward pass shape 확인
     - 파라미터 수 압축률 확인 (≥ target_compression)
     - 정확도 측정 (val top-1)
```

---

## 3. Step 1 — 아키텍처 분석 & 메모리 프로파일링

### 3.1 timm ViT / DeiT 모듈 구조

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=False)

for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name}: {module.__class__.__name__}  weight={tuple(module.weight.shape)}")
```

**timm ViT의 대표적인 모듈 경로:**

```
patch_embed.proj           Conv2d(3, 768, k=16, s=16)      ← PatchEmbed
blocks.0.norm1             LayerNorm(768)
blocks.0.attn.qkv          Linear(768, 2304)  [= 3*768]    ← Q+K+V 합산
blocks.0.attn.proj         Linear(768, 768)                 ← Attention 출력 proj
blocks.0.norm2             LayerNorm(768)
blocks.0.mlp.fc1           Linear(768, 3072)  [= 768*4]    ← FFN expand ★ PRUNABLE
blocks.0.mlp.fc2           Linear(3072, 768)                ← FFN shrink ★ coupled with fc1
blocks.1.norm1             ...
...
norm                       LayerNorm(768)                   ← 최종 norm
head                       Linear(768, 1000)                ← Classifier
```

**DeiT 추가 요소:**

```
dist_token                 (1, 1, 768)  ← distillation token
head_dist                  Linear(768, 1000)  ← distillation head (별도)
```

### 3.2 Prunable / Non-Prunable 분류

| 위치 | 분류 | 이유 |
|------|------|------|
| `blocks[i].mlp.fc1` (out_features) | **Prunable — G_FFN** | 내부 hidden dim, 외부 채널 (embed_dim)과 무관 |
| `blocks[i].mlp.fc2` (in_features) | **Prunable — G_FFN (coupled)** | fc1 출력 인덱스와 동일 인덱스로 입력 컬럼 축소 |
| `blocks[i].attn.qkv` (Q, K 부분) | **선택적 Prunable — G_QK** | head_dim 내부 dim. 단, **fused weight 분리 필요** |
| `blocks[i].attn.qkv` (V 부분) | **Non-Prunable** | V → proj 출력이 embed_dim에 고정됨 |
| `blocks[i].attn.proj` | **Non-Prunable** | 출력 = embed_dim, residual connection과 연결 |
| `patch_embed.proj` | **비권장** | stride/kernel 크기 변경 없이 채널만 바꾸면 다음 blocks 입력 불일치 |
| `norm`, `head` | **Non-Prunable** | embed_dim 및 n_classes에 고정 |

> **1차 구현 권장**: `G_FFN`만 구현한다. 파라미터 비중이 가장 크고 (전체의 ~67%), 구현이 단순하며, G_QK는 fused QKV weight 분리가 필요해 복잡도가 높다.

### 3.3 메모리 프로파일링 구현

아래 함수를 `measure_memory.py`로 만들어 실행하면 그룹별 파라미터 분포를 확인할 수 있다.

```python
import torch
import torch.nn as nn
import timm


def measure_vit_memory(model_name: str) -> None:
    model = timm.create_model(model_name, pretrained=False)
    total = sum(p.numel() for p in model.parameters())

    groups = {
        "G_FFN":    [],  # mlp.fc1 + mlp.fc2
        "G_QKV":    [],  # attn.qkv (reference only)
        "G_PROJ":   [],  # attn.proj
        "G_NORM":   [],  # LayerNorm들
        "G_HEAD":   [],  # classifier head
        "G_EMBED":  [],  # patch_embed + pos_embed + cls_token
        "G_OTHER":  [],
    }

    accounted_ids = set()

    for name, module in model.named_modules():
        mid = id(module)
        if mid in accounted_ids:
            continue

        if name.endswith('.mlp'):
            for p in module.parameters():
                groups["G_FFN"].append(p.numel())
                accounted_ids.add(id(p))
        elif name.endswith('.attn.qkv'):
            for p in module.parameters():
                groups["G_QKV"].append(p.numel())
                accounted_ids.add(id(p))
        elif name.endswith('.attn.proj'):
            for p in module.parameters():
                groups["G_PROJ"].append(p.numel())
                accounted_ids.add(id(p))

    head = getattr(model, 'head', None)
    if head is not None:
        for p in head.parameters():
            groups["G_HEAD"].append(p.numel())
            accounted_ids.add(id(p))

    # 나머지
    for p in model.parameters():
        if id(p) not in accounted_ids:
            groups["G_OTHER"].append(p.numel())

    print(f"\n=== {model_name} parameter breakdown ===")
    print(f"{'group':<12}{'numel':>14}{'MB':>10}{'%':>9}")
    print("-" * 47)
    for g, nums in groups.items():
        n = sum(nums)
        print(f"{g:<12}{n:>14,}{n*4/1e6:>10.3f}{100*n/total:>8.2f}%")
    print("-" * 47)
    print(f"{'TOTAL':<12}{total:>14,}{total*4/1e6:>10.3f}{'100.00':>8}%")


if __name__ == "__main__":
    measure_vit_memory('vit_base_patch16_224')
    measure_vit_memory('deit_base_patch16_224')
```

**vit_base_patch16_224 예상 수치 (참고):**

```
group          numel          MB        %
G_FFN       57,885,696    231.543   67.41%   ← 가장 큰 Prunable 영역
G_QKV       14,155,776     56.623   16.49%
G_PROJ       4,722,432     18.890    5.50%
G_NORM           6,912      0.028    0.01%
G_HEAD         769,000      3.076    0.90%
G_EMBED        588,288      2.353    0.69%   ← pos_embed + cls_token 포함
G_OTHER      7,888,896     31.556    9.19%
TOTAL       86,017,000    344.068  100.00%
```

### 3.4 target_compression vs 예상 sparsity

G_FFN만 pruning 시 (vit_base_patch16_224, prunable ≈ 67.41%):

| target_compression | 예상 backbone sparsity | 비고 |
|-------------------|----------------------|------|
| 0.10 | ≈ 0.15 | 안전 |
| 0.20 | ≈ 0.30 | 권장 시작점 |
| 0.30 | ≈ 0.45 | 검증 필요 |
| 0.50 | ≈ 0.74 | 공격적 |
| 0.67 | ≈ 0.95 (상한) | 한계 |

> 정확한 값은 이진탐색([§6](#6-sparsity-이진탐색))으로 계산한다.  
> Secondary effect(fc1 출력 축소 → fc2 입력 컬럼도 같이 축소)를 반드시 포함해야 한다.

---

## 4. Step 2 — Soft Pruning (Fine-tuning 중)

### 4.1 핵심 구현 파일 구조

```
your_repo/
├── pruning/
│   ├── __init__.py
│   ├── vit_pruning.py      ← Soft Pruning 핵심 (이 섹션에서 설명)
│   └── vit_reducing.py     ← Reducing 핵심 (§5에서 설명)
├── train.py                ← 훈련 진입점 (DeiT engine.py 수정 또는 직접 작성)
└── reduce.py               ← Reducing CLI
```

### 4.2 핵심 수치 헬퍼

```python
MIN_SURVIVE = 4  # 어떤 그룹이든 최소 4채널은 살린다

def _calc_n_prune(n_total: int, sparsity: float) -> int:
    """sparsity 비율로 제거할 채널 수. MIN_SURVIVE 보장."""
    n_prune = round(n_total * sparsity)          # int() 아닌 round() — 소규모 그룹 편향 방지
    n_prune = min(n_prune, n_total - MIN_SURVIVE)
    return max(n_prune, 0)

def _topk_smallest_l2_idx(weight: torch.Tensor, k: int) -> torch.Tensor:
    """첫 차원(out_features/out_channels) 기준 L2 norm 하위 k개 인덱스."""
    n = weight.shape[0]
    norms = torch.norm(weight.detach().reshape(n, -1), dim=1)
    _, idx = torch.topk(norms, k, largest=False)
    return idx
```

### 4.3 G_FFN 마스킹 — `_prune_ffn`

timm ViT의 FFN 구조:
```
blocks[i].mlp.fc1: Linear(embed_dim, mlp_ratio * embed_dim)   ← expand
blocks[i].mlp.fc2: Linear(mlp_ratio * embed_dim, embed_dim)   ← shrink
blocks[i].norm2:   LayerNorm(embed_dim)                        ← FFN 앞 norm (입력, 건드리지 않음)
```

마스킹 대상: **fc1의 출력 행(row)** = fc2의 입력 열(col). 같은 인덱스를 양쪽에 적용.

```python
import torch
import torch.nn as nn

def _zero_ln_(ln: nn.LayerNorm, idx: torch.Tensor) -> None:
    """LayerNorm의 pruned 위치 weight/bias를 0으로."""
    # LayerNorm은 BN과 달리 running_mean/var가 없음 → weight/bias만 처리
    with torch.no_grad():
        if ln.weight is not None:
            ln.weight.data[idx] = 0.0
        if ln.bias is not None:
            ln.bias.data[idx] = 0.0


def _prune_ffn(mlp: nn.Module, sparsity: float) -> None:
    """timm ViT MLP block의 fc1(expand) hidden dim을 L2 norm 기준으로 마스킹.
    
    fc1 출력과 fc2 입력은 동일 채널을 공유하므로 같은 인덱스를 모두 0으로 설정.
    """
    fc1_weight = mlp.fc1.weight   # shape: (mlp_dim, embed_dim)
    mlp_dim = fc1_weight.shape[0]
    n_prune = _calc_n_prune(mlp_dim, sparsity)
    if n_prune <= 0:
        return

    idx = _topk_smallest_l2_idx(fc1_weight, n_prune)

    with torch.no_grad():
        # fc1: 출력 행(row) 마스킹
        mlp.fc1.weight.data[idx] = 0.0
        if mlp.fc1.bias is not None:
            mlp.fc1.bias.data[idx] = 0.0

        # fc2: 입력 열(col) 마스킹  ← 반드시 fc1과 동일 idx
        mlp.fc2.weight.data[:, idx] = 0.0
        # fc2.bias는 출력(embed_dim) 소속이므로 건드리지 않음

    # NOTE: timm ViT의 MLP에는 fc1/fc2 사이에 별도 norm이 없다.
    # 만약 fc1 다음에 norm이 있는 아키텍처라면 아래와 같이 처리:
    # if hasattr(mlp, 'norm') and isinstance(mlp.norm, nn.LayerNorm):
    #     _zero_ln_(mlp.norm, idx)
```

### 4.4 (선택) G_QK 마스킹

> **1차 구현에서는 건너뛰어도 된다.** fused QKV weight 분리가 필요해 복잡도가 높다.

timm ViT의 `attn.qkv`는 Q/K/V를 하나의 weight로 합친 fused projection이다:

```python
# attn.qkv.weight: shape (3 * num_heads * head_dim, embed_dim)
# = (3 * total_head_dim, embed_dim)
# 내부 배치: [Q_rows | K_rows | V_rows]  (각 num_heads * head_dim씩)
```

Q, K의 head_dim을 pruning하려면:
- Q, K는 동일한 인덱스를 사용해야 한다 (QK^T 차원 일치 필요)
- V는 건드리면 안 된다 (proj 출력이 embed_dim에 고정)
- fused weight를 `[Q | K | V]`로 분리한 뒤 Q+K 부분만 마스킹하고 재합산

```python
def _prune_attn_qk(attn: nn.Module, sparsity: float) -> None:
    """Q, K head_dim만 pruning. V는 보존. (선택 구현)"""
    total_dim = attn.qkv.weight.shape[0]   # 3 * num_heads * head_dim
    head_dim_total = total_dim // 3         # num_heads * head_dim

    # Q 기준으로 인덱스 계산
    q_weight = attn.qkv.weight[:head_dim_total]    # (head_dim_total, embed_dim)
    n_prune = _calc_n_prune(head_dim_total, min(sparsity, 0.90))  # QK는 0.90 상한 권장
    if n_prune <= 0:
        return
    idx = _topk_smallest_l2_idx(q_weight, n_prune)

    with torch.no_grad():
        # Q 행 마스킹
        attn.qkv.weight.data[:head_dim_total][idx] = 0.0
        # K 행 마스킹 — 반드시 Q와 동일 인덱스
        attn.qkv.weight.data[head_dim_total:2*head_dim_total][idx] = 0.0
        # bias 처리 (있을 경우)
        if attn.qkv.bias is not None:
            attn.qkv.bias.data[:head_dim_total][idx] = 0.0
            attn.qkv.bias.data[head_dim_total:2*head_dim_total][idx] = 0.0
        # V는 건드리지 않음 (attn.qkv.weight.data[2*head_dim_total:])
```

### 4.5 _PruneGroup 패턴 (성능 최적화)

매 step `model.modules()`를 순회하면 CPU 병목이 발생한다. **첫 `apply()` 호출 시 한 번만 그룹을 수집**하고, 이후에는 캐시된 텐서 레퍼런스만 사용하는 패턴이 효율적이다.

```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class _PruneGroup:
    criterion: torch.Tensor                        # ranking 기준 weight (fc1.weight)
    sparsity: float
    targets: List[Tuple[torch.Tensor, int, float]] # (tensor, dim, fill_value)
    _mask: torch.Tensor | None = field(default=None, repr=False)

    def refresh(self) -> None:
        """topk 인덱스를 재계산하고 마스크를 갱신."""
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
            shape = [1] * tensor.dim()
            shape[dim] = -1
            m = mask.view(shape)
            if fill == 0.0:
                tensor.data.mul_(m)
            else:
                # running_var 같은 경우: pruned → fill값(1.0), alive → 현재 값
                tensor.data.mul_(m).add_(fill * (1.0 - m))
```

`_PruneGroup`을 FFN 블록 하나에 적용하는 예:

```python
def _collect_ffn_group(mlp: nn.Module, sparsity: float) -> _PruneGroup:
    targets = []
    # fc1 출력 행 (dim=0)
    targets.append((mlp.fc1.weight, 0, 0.0))
    if mlp.fc1.bias is not None:
        targets.append((mlp.fc1.bias, 0, 0.0))
    # fc2 입력 열 (dim=1)
    targets.append((mlp.fc2.weight, 1, 0.0))
    # fc1 다음에 norm이 있다면:
    # if hasattr(mlp, 'norm') and mlp.norm is not None:
    #     if mlp.norm.weight is not None: targets.append((mlp.norm.weight, 0, 0.0))
    #     if mlp.norm.bias   is not None: targets.append((mlp.norm.bias,   0, 0.0))
    return _PruneGroup(criterion=mlp.fc1.weight, sparsity=sparsity, targets=targets)
```

### 4.6 Pruner 클래스 전체 구조

```python
class ViTPruner:
    """timm ViT/DeiT용 Soft Pruning 컨트롤러.

    사용법:
        pruner = ViTPruner(model, target_compression=0.30)
        # 학습 루프 안에서, optimizer.step() 직후:
        pruner.apply(model)
    """

    def __init__(
        self,
        model: nn.Module,
        target_compression: float,
        max_sparsity: float = 0.95,
        sparsity: float | None = None,
        index_refresh_steps: int = 100,
    ) -> None:
        self.target_compression = float(target_compression)
        self.max_sparsity = float(max_sparsity)
        self.index_refresh_steps = int(index_refresh_steps)
        self._step = 0
        self._groups: list[_PruneGroup] = []  # lazy init (apply 첫 호출 시)

        if sparsity is not None:
            self.sparsity = float(sparsity)
        else:
            # 이진탐색으로 per-group sparsity 결정 — §6 참고
            self.sparsity = _find_sparsity_by_bisection(
                model, self.target_compression, self.max_sparsity
            )

        # 로그
        total = sum(p.numel() for p in model.parameters())
        est = _estimate_total_removed(model, self.sparsity)
        rate = 100.0 * est / max(total, 1)
        print(
            f"[ViTPruner] target={self.target_compression*100:.1f}% "
            f"sparsity={self.sparsity:.4f} "
            f"estimated_compression={rate:.2f}%"
        )

    def _collect_all_groups(self, model: nn.Module) -> list[_PruneGroup]:
        groups = []
        for block in model.blocks:
            groups.append(_collect_ffn_group(block.mlp, self.sparsity))
        return groups

    def apply(self, model: nn.Module) -> None:
        if self.sparsity <= 0:
            return
        # Lazy init: 첫 apply()에서 model이 CUDA에 있을 때 수집
        # (init 시점에는 모델이 CPU에 있어 device mismatch 발생 가능)
        if not self._groups:
            self._groups = self._collect_all_groups(model)
            for g in self._groups:
                g.refresh()
            self._step += 1
            return

        need_refresh = (
            self.index_refresh_steps <= 0
            or self._step % self.index_refresh_steps == 0
        )
        if need_refresh:
            for g in self._groups:
                g.refresh()
        for g in self._groups:
            g.apply()
        self._step += 1

    @torch.no_grad()
    def log_sparsity(self, model: nn.Module) -> dict[str, float]:
        """실제 zero 비율을 반환. WandB 로깅용."""
        n_total, n_zero = 0, 0
        result = {}
        for name, module in model.named_modules():
            if hasattr(module, 'mlp') or not (
                hasattr(module, 'fc1') and hasattr(module, 'fc2')
            ):
                continue
            w = module.fc1.weight
            n = w.shape[0]
            norms = torch.norm(w.detach().reshape(n, -1), dim=1)
            n_z = int((norms == 0).sum().item())
            n_total += n
            n_zero += n_z
            safe_name = name.replace('.', '/')
            result[f"pruning/layer/{safe_name}"] = n_z / max(n, 1)

        result["pruning/zero_filters"] = n_zero
        result["pruning/prunable_filters"] = n_total
        result["pruning/actual_sparsity"] = n_zero / max(n_total, 1)
        result["pruning/target_sparsity"] = self.sparsity
        return result
```

---

## 5. Step 3 — Reducing (학습 완료 후)

### 5.1 핵심 원리

Soft Pruning 학습이 끝나면:
- `fc1.weight` 의 L2 norm이 정확히 0인 행 = 실질적으로 제거된 채널
- 이 행과 대응하는 fc2의 열을 물리적으로 삭제하면 → 실제로 작은 Dense 모델

```
Soft Pruning 완료 모델:
  fc1.weight[dead_rows]  ≈ 0.0  (L2 norm == 0)
  fc1.weight[live_rows]  ≠ 0.0

Reducing:
  survived = where(norm(fc1.weight) != 0)
  new_fc1 = Linear(embed_dim, len(survived))
  new_fc1.weight = fc1.weight[survived]          ← 살아남은 행만
  new_fc2 = Linear(len(survived), embed_dim)
  new_fc2.weight = fc2.weight[:, survived]        ← 대응하는 열만
```

### 5.2 Survived Index 추출

```python
@torch.no_grad()
def _survived_idx(weight: torch.Tensor) -> torch.Tensor:
    """첫 차원 기준 L2 norm이 0이 아닌 인덱스 (오름차순)."""
    n = weight.shape[0]
    norms = torch.norm(weight.reshape(n, -1), dim=1)
    # != 0 사용: Soft Pruning은 정확히 0으로 마스킹하므로 임계값 불필요
    return torch.nonzero(norms != 0, as_tuple=False).flatten()
```

### 5.3 FFN 블록 Reducing

```python
@torch.no_grad()
def _reduce_ffn(mlp: nn.Module) -> None:
    """mlp.fc1 / fc2를 survived 인덱스 기준으로 in-place 교체."""
    survived = _survived_idx(mlp.fc1.weight)
    n_new = survived.numel()
    if n_new == mlp.fc1.weight.shape[0]:
        return  # 줄일 것 없음

    embed_dim = mlp.fc2.weight.shape[0]   # 출력 = embed_dim (고정)
    dev = mlp.fc1.weight.device
    dtype = mlp.fc1.weight.dtype

    # fc1 교체: (mlp_dim, embed_dim) → (n_new, embed_dim)
    new_fc1 = nn.Linear(
        mlp.fc1.in_features, n_new,
        bias=(mlp.fc1.bias is not None)
    ).to(dev).to(dtype)
    new_fc1.weight.data.copy_(mlp.fc1.weight.data[survived])
    if mlp.fc1.bias is not None:
        new_fc1.bias.data.copy_(mlp.fc1.bias.data[survived])
    mlp.fc1 = new_fc1

    # fc2 교체: (embed_dim, mlp_dim) → (embed_dim, n_new)
    new_fc2 = nn.Linear(
        n_new, embed_dim,
        bias=(mlp.fc2.bias is not None)
    ).to(dev).to(dtype)
    new_fc2.weight.data.copy_(mlp.fc2.weight.data[:, survived])
    if mlp.fc2.bias is not None:
        # bias는 출력(embed_dim) 소속이므로 그대로 복사
        new_fc2.bias.data.copy_(mlp.fc2.bias.data)
    mlp.fc2 = new_fc2

    # fc1 다음에 norm이 있는 경우
    # if hasattr(mlp, 'norm') and isinstance(mlp.norm, nn.LayerNorm):
    #     old_ln = mlp.norm
    #     new_ln = nn.LayerNorm(n_new, eps=old_ln.eps).to(dev).to(dtype)
    #     new_ln.weight.data.copy_(old_ln.weight.data[survived])
    #     new_ln.bias.data.copy_(old_ln.bias.data[survived])
    #     mlp.norm = new_ln
```

### 5.4 모델 전체 Reducing

```python
@torch.no_grad()
def reduce_vit_model(model: nn.Module) -> nn.Module:
    """모델 전체를 in-place로 dense reduce. 반환은 동일 객체."""
    for block in model.blocks:
        _reduce_ffn(block.mlp)
    return model
```

### 5.5 Reducing CLI

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='timm model name')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-size', type=int, default=224)
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=False)

    # checkpoint 로드 (EMA 우선)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        # EMA weights 우선 사용 — Soft Pruning 학습에서 raw network weight는
        # 매 step 0으로 강제되어 degraded 상태. EMA가 실제 성능을 유지한다.
        # 저장 형식에 따라 키 이름이 다를 수 있으니 ckpt.keys()로 확인할 것.
        state_dict = ckpt.get('model_ema', ckpt.get('model', ckpt.get('state_dict', ckpt)))
    else:
        state_dict = ckpt
    # DDP module. prefix 제거
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    n_before = sum(p.numel() for p in model.parameters())
    print(f"BEFORE reduce: {n_before:,}")

    reduce_vit_model(model)

    n_after = sum(p.numel() for p in model.parameters())
    rate = 100.0 * (n_before - n_after) / n_before
    print(f"AFTER  reduce: {n_after:,}  ({rate:.2f}% removed)")

    # Forward 검증
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, args.input_size, args.input_size))
    print(f"Forward OK: output shape = {tuple(out.shape)}")

    torch.save({'state_dict': model.state_dict(), 'compression_rate': rate}, args.output)
    print(f"Saved to {args.output}")
```

### 5.6 EMA 우선 사용의 이유

Soft Pruning 학습에서 raw network의 weight는 매 step L2 norm 하위 X%가 강제로 0이 된다.  
EMA(Exponential Moving Average) shadow weight는 이 0-마스킹 전의 값들의 평균을 추적하므로  
**실제 성능은 EMA weight가 보존한다.** Reducing 시 반드시 EMA weight를 사용해야 한다.

timm의 `ModelEmaV2`나 DeiT의 EMA 구현 시 저장 키 이름이 다를 수 있으니  
checkpoint의 `ckpt.keys()`를 먼저 확인하고 적절한 키를 선택한다.

---

## 6. Sparsity 이진탐색

### 6.1 왜 이진탐색이 필요한가

단순 선형 계산:

```python
# 잘못된 방법
sparsity = target_compression * total_params / ffn_params
```

이 방법은 **secondary effect(이차 효과)**를 무시한다:
- fc1을 s% 제거 → fc2 입력도 s% 제거 (연동)
- 결과: 실제 압축률이 선형 계산보다 **더 크게** 나온다 (과소 추정)

### 6.2 정확한 제거량 추정 — FFN 한 블록

```python
def _estimate_removed_ffn(mlp: nn.Module, sparsity: float) -> int:
    """한 FFN 블록에서 sparsity로 제거되는 파라미터 수 (secondary effect 포함)."""
    fc1_w = mlp.fc1.weight         # (mlp_dim, embed_dim)
    mlp_dim = fc1_w.shape[0]
    embed_dim = fc1_w.shape[1]
    fc2_out = mlp.fc2.weight.shape[0]  # = embed_dim

    n_prune = _calc_n_prune(mlp_dim, sparsity)
    if n_prune <= 0:
        return 0

    removed = 0
    # fc1 weight: (mlp_dim, embed_dim) → n_prune 행 제거
    removed += n_prune * embed_dim
    # fc1 bias: n_prune개
    if mlp.fc1.bias is not None:
        removed += n_prune
    # fc2 weight: (embed_dim, mlp_dim) → n_prune 열 제거  ← secondary effect
    removed += fc2_out * n_prune
    # fc2 bias: 출력(embed_dim) 소속이므로 제거 없음

    # fc1 다음 norm이 있는 경우:
    # if hasattr(mlp, 'norm') and mlp.norm is not None:
    #     removed += n_prune * 2  # weight + bias

    return removed
```

### 6.3 이진탐색 구현

```python
def _estimate_total_removed(model: nn.Module, sparsity: float) -> int:
    total = 0
    for block in model.blocks:
        total += _estimate_removed_ffn(block.mlp, sparsity)
    return total


def _find_sparsity_by_bisection(
    model: nn.Module,
    target_compression: float,
    max_sparsity: float = 0.95,
    iters: int = 64,
) -> float:
    if target_compression <= 0:
        return 0.0
    total_params = sum(p.numel() for p in model.parameters())
    target_remove = target_compression * total_params
    lo, hi = 0.0, max_sparsity
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
```

64회 반복 → 약 `1e-19` 정밀도. 충분히 수렴.

---

## 7. WandB 연동

### 7.1 사전 준비

```bash
pip install wandb
wandb login        # API 키 입력 또는:
export WANDB_API_KEY=<your_key>
```

### 7.2 학습 시작 시 초기화

```python
import wandb

if args.wandb:
    run = wandb.init(
        project=args.wandb_project,    # e.g. "vit-pruning"
        name=args.wandb_run_name,      # e.g. "vit_base_prune30"
        id=args.wandb_run_id or None,  # resume 시 기존 run id
        resume="allow" if args.wandb_run_id else None,
        config=vars(args),
    )
    # gradient histogram 기록 (500 step 마다)
    wandb.watch(model, log='gradients', log_freq=500)
```

### 7.3 학습 루프 내 로깅

```python
# 매 epoch 끝에서 로깅
if args.wandb and wandb.run is not None:
    log_dict = {
        "train/loss": train_loss,
        "train/top1": train_acc1,
        "train/lr": optimizer.param_groups[0]['lr'],
        "val/top1": val_acc1,
        "val/top5": val_acc5,
        "val/loss": val_loss,
        "val/top1_best": best_acc1,
        "epoch": epoch,
    }
    # pruner가 활성화된 경우 sparsity 지표 추가
    if pruner is not None:
        sparsity_dict = pruner.log_sparsity(model)
        log_dict.update(sparsity_dict)
    wandb.log(log_dict)
```

### 7.4 로깅 항목 정의

| wandb 키 | 내용 | 비고 |
|----------|------|------|
| `train/loss` | epoch 평균 학습 손실 | |
| `train/top1` | epoch 평균 학습 Top-1 | |
| `train/lr` | 현재 learning rate | |
| `val/top1` | 검증 Top-1 | |
| `val/top5` | 검증 Top-5 | |
| `val/top1_best` | 현재까지 최고 val top-1 | |
| `pruning/actual_sparsity` | 전체 prunable 채널 대비 실제 zero 비율 | pruner 활성 시만 |
| `pruning/zero_filters` | zero 채널 수 | pruner 활성 시만 |
| `pruning/layer/<name>` | 블록별 zero 비율 | 계층별 세분화 |

### 7.5 학습 종료 시

```python
if args.wandb and wandb.run is not None:
    wandb.finish()
```

---

## 8. 훈련 루프 연동

### 8.1 timm 모델 + DeiT 훈련 스크립트

timm은 모델만 제공하므로 훈련 코드가 별도로 필요하다.  
**DeiT 공식 훈련 스크립트** (`facebookresearch/deit`)가 timm 모델을 그대로 사용하며  
`engine.py`의 `train_one_epoch` 함수 구조가 명확해 pruner 삽입이 용이하다.

### 8.2 DeiT engine.py 수정 (핵심 부분)

```python
# engine.py — train_one_epoch 함수 내

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    loss_scaler, max_norm=0, model_ema=None,
                    pruner=None,         # ← 추가
                    wandb_log=False):

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(
            loss, optimizer, clip_grad=max_norm,
            parameters=model.parameters(), create_graph=is_second_order
        )
        # loss_scaler 내부에서 backward + optimizer.step()이 수행된다.

        # ★ Soft Pruning Hook ★ — optimizer.step() 직후, model_ema.update() 이전
        if pruner is not None:
            # DDP인 경우 model.module로 전달
            actual_model = model.module if hasattr(model, 'module') else model
            pruner.apply(actual_model)

        torch.cuda.synchronize()

        # EMA 업데이트 — pruning 후 weight를 추적하도록 이후에 실행
        if model_ema is not None:
            model_ema.update(model)

        # 메트릭 업데이트 ...
```

### 8.3 훈련 진입점 (main.py) 수정

```python
# main.py 또는 train.py

# Soft Pruning 인자 추가
parser.add_argument('--target-compression', type=float, default=0.0,
                    help='target parameter compression (0.0 = disabled)')
parser.add_argument('--pruning-max-sparsity', type=float, default=0.95)
parser.add_argument('--prune-refresh-steps', type=int, default=100,
                    help='topk 재계산 간격 (0=매step, 100=100step마다)')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb-project', type=str, default='vit-pruning')
parser.add_argument('--wandb-run-name', type=str, default='')

# 모델 로드 (pretrained)
model = timm.create_model(args.model, pretrained=True)
model = model.to(device)

# Pruner 초기화 (target_compression=0이면 None)
if args.target_compression > 0:
    from pruning.vit_pruning import ViTPruner
    pruner = ViTPruner(
        model,
        target_compression=args.target_compression,
        max_sparsity=args.pruning_max_sparsity,
        index_refresh_steps=args.prune_refresh_steps,
    )
else:
    pruner = None

# 학습 루프 (예시)
for epoch in range(args.epochs):
    train_stats = train_one_epoch(
        model, criterion, data_loader_train, optimizer, device, epoch,
        loss_scaler, args.clip_grad, model_ema,
        pruner=pruner,               # ← 전달
        wandb_log=args.wandb,
    )
    val_stats = evaluate(data_loader_val, model, device)

    # WandB 로깅
    if args.wandb and wandb.run is not None:
        log_dict = {
            'epoch': epoch,
            'train/loss': train_stats['loss'],
            'val/top1': val_stats['acc1'],
            'val/top5': val_stats['acc5'],
        }
        if pruner is not None:
            log_dict.update(pruner.log_sparsity(
                model.module if hasattr(model, 'module') else model
            ))
        wandb.log(log_dict)

    # Best checkpoint 저장
    if val_stats['acc1'] > max_accuracy:
        max_accuracy = val_stats['acc1']
        torch.save({'model': model.state_dict(),
                    'model_ema': get_state_dict(model_ema),
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.output_dir, 'model_best.pt'))
```

### 8.4 삽입 위치의 원칙 (반드시 지킬 것)

```
optimizer.step()  →  loss_scaler.update()  →  ★ pruner.apply() ★  →  model_ema.update()  →  ...
                                                 ↑                 ↑
                                          gradient 반영 후    EMA가 pruning 후 weight 추적
```

- **`optimizer.step()` 이후**: gradient가 weight에 반영된 직후 마스킹
- **`model_ema.update()` 이전**: EMA가 pruning 후 weight를 추적하도록

---

## 9. 검증 체크리스트

### 학습 시작 시

- [ ] `[ViTPruner] target=X% sparsity=Y.YYYY estimated_compression=Z.ZZ%` 로그 출력 확인
- [ ] 첫 1~2 epoch 후 WandB `pruning/actual_sparsity` 가 target_sparsity에 근접한지 확인
- [ ] val/top1 이 너무 급격히 하락하지 않는지 확인 (epoch 1에서 5% 이상 하락이면 sparsity 낮추기)

### 학습 종료 후

- [ ] WandB에서 `val/top1_best` 와 baseline 비교
- [ ] `pruning/actual_sparsity` 가 target_sparsity와 일치하는지 확인

### Reducing 후

```python
# 반드시 실행
model.eval()
with torch.no_grad():
    out = model(torch.zeros(1, 3, 224, 224))
assert out.shape == (1, 1000), f"출력 shape 불일치: {out.shape}"

n_before = <original_param_count>
n_after = sum(p.numel() for p in model.parameters())
rate = 100 * (n_before - n_after) / n_before
assert rate >= target_compression * 100, f"압축률 미달: {rate:.2f}%"
print(f"압축률: {rate:.2f}% (target: {target_compression*100:.1f}%)")
```

- [ ] Forward pass shape 정상
- [ ] 압축률 ≥ target_compression × 100%
- [ ] Reduced 모델로 val top-1 재측정 (Soft Pruning 완료 모델과 비교)

---

## 10. 흔한 실수 & 주의사항

### ⚠️ LayerNorm vs BatchNorm — running_var 처리 차이

```python
# BatchNorm: running_var를 반드시 1.0으로 설정 (0이면 분모 0 문제)
bn.running_var[idx] = 1.0   # ← 0이면 BN(x) = (x-mean)/sqrt(0+eps) → 비정상

# LayerNorm: running_mean/var 없음 → weight/bias만 0으로 처리
ln.weight.data[idx] = 0.0
ln.bias.data[idx]   = 0.0
# running_mean, running_var는 존재하지 않으므로 건드리지 않는다
```

### ⚠️ EMA weights 반드시 사용

```python
# Reducing CLI에서 checkpoint 로드 시
ckpt = torch.load(path, map_location='cpu', weights_only=False)
# ckpt.keys()로 먼저 확인 후 EMA 키 선택
# DeiT style: 'model_ema' 키
# timm ModelEmaV2: 'state_dict_ema' 또는 별도 저장
state_dict = ckpt.get('model_ema', ckpt.get('state_dict', ckpt))
```

raw `model` weight는 Soft Pruning으로 인해 매 step 0으로 리셋된 상태다.  
**EMA weight가 실제 학습된 성능을 보존**하므로 반드시 EMA를 사용한다.

### ⚠️ DDP 환경에서의 pruner.apply()

```python
# DDP로 래핑된 경우
actual_model = model.module if hasattr(model, 'module') else model
pruner.apply(actual_model)   # ← 반드시 .module 전달
```

### ⚠️ int() 대신 round() 사용

```python
# 소규모 레이어에서 int()는 편향 발생
n_prune = int(16 * 0.30)    # = 4  (4.8을 버림)
n_prune = round(16 * 0.30)  # = 5  (정확)
```

### ⚠️ Lazy init — init 시점의 device mismatch

```python
# 잘못된 방법: __init__에서 그룹 수집
class Pruner:
    def __init__(self, model):
        self._groups = self._collect_all_groups(model)  # 모델이 CPU에 있음!
        # → BN running_mean/var가 CPU tensor로 수집됨
        # → model.cuda() 이후 device mismatch 발생

# 올바른 방법: apply() 첫 호출 시 (model이 CUDA에 있을 때)
class Pruner:
    def __init__(self, model):
        self._groups = []  # 비어있음

    def apply(self, model):
        if not self._groups:                  # 첫 호출 시
            self._groups = self._collect_all_groups(model)  # 이 시점에 CUDA
            for g in self._groups: g.refresh()
            return
        # 이후는 캐시 사용
```

### ⚠️ fc2.bias는 건드리지 않는다

```python
# fc2.bias shape: (embed_dim,) — 출력 소속이므로 pruning 대상 아님
# fc2를 새 Linear로 교체할 때:
new_fc2 = nn.Linear(n_new, embed_dim, bias=(mlp.fc2.bias is not None))
new_fc2.weight.data.copy_(mlp.fc2.weight.data[:, survived])
if mlp.fc2.bias is not None:
    new_fc2.bias.data.copy_(mlp.fc2.bias.data)   # ← 그대로 복사 (축소 없음)
```

### ⚠️ index_refresh_steps 권장값

| 단계 | 권장값 | 이유 |
|------|--------|------|
| 초기 수렴 전 (epoch 1~5) | 50 | 빠르게 중요 채널 결정 |
| 일반 학습 | 100 | CPU topk 병목 완화, 기본값 |
| 수렴 후 | 200~500 | 인덱스가 거의 변하지 않으므로 낭비 감소 |

### ⚠️ G_QK 구현 시 반드시 Q, K 동일 인덱스

```python
# 절대 금지: 각각 독립적으로 계산
q_idx = topk_smallest(q_norms, k)
k_idx = topk_smallest(k_norms, k)   # QK^T 차원 불일치!

# 올바른 방법: Q 기준으로 K에 강제 적용
q_idx = topk_smallest(q_norms, k)
k_idx = q_idx                       # 동일 인덱스
```

---

## 11. 레퍼런스 구현 (EfficientViT) 요약

이 문서의 방법론은 EfficientViT 분류 모델(`efficientvit-b1/b2/b3/l1/l2/l3`)에 먼저 적용되었다.  
동일한 구조로 timm ViT/DeiT에 재적용하는 것이 본 문서의 목적이다.

### Prunable 그룹 대응표

| EfficientViT | timm ViT / DeiT | 처리 원칙 |
|--------------|-----------------|-----------|
| `G_MBCONV` — `MBConv` 의 `inverted_conv → depth_conv → point_conv` | `G_FFN` — `mlp.fc1(expand) → mlp.fc2(shrink)` | expand-shrink 쌍: 동일 인덱스 |
| `G_STEM` — `input_stem` chain (Conv + DSConv 반복) | `patch_embed.proj` (독립 Conv) | 채널 mismatch 주의, 1차는 제외 권장 |
| `G_HEAD0/1` — ClsHead hidden dim | `head` Linear (embed_dim→n_classes) | head는 embed_dim 고정이므로 non-prunable |

### 핵심 수치 (EfficientViT-B1 기준, 참고용)

| 항목 | 수치 |
|------|------|
| 전체 파라미터 | 9.1M |
| Prunable (G_STEM + G_MBCONV) | 32% |
| Non-prunable (G_HEAD + G_LiteMLA) | 68% |
| target=15%, head_scale=0.5 → backbone sparsity | ≈ 0.22~0.27 |
| target=30%, head_scale=0.5 → backbone sparsity | ≈ 0.47~0.55 |

### 레퍼런스 파일 위치

```
efficientvit2026/
├── efficientvit/clscore/pruning/
│   ├── efficientvit_pruning.py     ← Pruner 클래스 (이 문서 §4 의 원본)
│   └── efficientvit_reducing.py    ← Reducer 함수 (이 문서 §5 의 원본)
├── applications/efficientvit_cls/
│   ├── train_efficientvit_cls_model.py  ← 학습 진입점 (argparse + pruner 연동)
│   ├── reduce_efficientvit_cls_model.py ← Reducing CLI
│   └── measure_memory.py               ← 메모리 프로파일러
├── PRUNING_IMPLEMENTATION_REPORT.md    ← EfficientViT 적용 상세 보고서
└── PRUNING_METHODOLOGY.md              ← 방법론 원본 문서
```

---

*문서 작성 기준: EfficientViT Soft Pruning 구현 완료 시점 (2026-06)*  
*적용 대상: timm 라이브러리 기반 ViT / DeiT 신규 레포*
