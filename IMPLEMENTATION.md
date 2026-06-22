# ViT / DeiT Soft Pruning 구현 보고서

> 작성 기준: 2026-06  
> 환경: timm 1.0.27 · torch 2.9.1 · Python 3.13.5  
> 서버: `root@59bfae69b3a9:/workspace/etri_iitp/JS/Server_Compression`  
> 레퍼런스: EfficientViT Soft Pruning (동일 방법론을 timm ViT/DeiT에 이식)

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [방법론 요약](#2-방법론-요약)
3. [환경 설정](#3-환경-설정)
4. [아키텍처 분석 결과](#4-아키텍처-분석-결과)
5. [구현 파일 설명](#5-구현-파일-설명)
6. [학습 실행 명령어](#6-학습-실행-명령어)
7. [Reducing 실행 명령어](#7-reducing-실행-명령어)
8. [Reduced 모델 로드 방법](#8-reduced-모델-로드-방법)
9. [WandB 모니터링 지표](#9-wandb-모니터링-지표)
10. [핵심 설계 결정 사항](#10-핵심-설계-결정-사항)
11. [주의사항 & 트러블슈팅](#11-주의사항--트러블슈팅)

---

## 1. 프로젝트 구조

```
Server_Compression/
├── pruning/
│   ├── __init__.py
│   ├── vit_pruning.py       ← ViTPruner: Soft Pruning 컨트롤러
│   └── vit_reducing.py      ← reduce_vit_model: Dense 변환
├── engine.py                ← train_one_epoch / evaluate
├── train.py                 ← 학습 진입점 (단일GPU / DDP)
├── reduce.py                ← Reducing CLI
├── measure_memory.py        ← 아키텍처 분석 & 파라미터 프로파일링
├── data/
│   └── imagenet/            ← ImageNet (서버에만 존재, gitignore)
│       ├── train/
│       └── val/
├── output/                  ← 체크포인트 저장 (gitignore)
├── TIMM_PRUNING_GUIDE.md    ← 방법론 상세 가이드
├── ARCHITECTURE_ANALYSIS.md ← 모델별 아키텍처 분석 결과
├── IMPLEMENTATION.md        ← 본 문서
└── .gitignore
```

---

## 2. 방법론 요약

### Soft Pruning → Reducing 2단계 파이프라인

```
[Soft Pruning — 학습 중]
  매 optimizer.step() 직후:
    fc1.weight 의 L2 norm 하위 X% 행(row) → 0으로 마스킹
    fc2.weight 의 동일 인덱스 열(col)     → 0으로 마스킹
  → 아키텍처 구조는 Dense 그대로 유지
  → 100 step마다 마스크 재계산 (중요 채널은 gradient로 살아날 수 있음)

[Reducing — 학습 완료 후]
  zero 채널을 물리적으로 제거 → 실제로 작은 Dense 모델
  fc1: (mlp_dim, embed_dim) → (n_survived, embed_dim)
  fc2: (embed_dim, mlp_dim) → (embed_dim, n_survived)
```

### Pruning 대상: G_FFN (FFN hidden dimension)

```
blocks[i].mlp.fc1: Linear(embed_dim → mlp_dim)   ★ Prunable
                          ↓ GELU
blocks[i].mlp.fc2: Linear(mlp_dim → embed_dim)   ★ Prunable (coupled)
```

- fc1 출력과 fc2 입력이 동일한 `mlp_dim` 채널을 공유 → 반드시 같은 인덱스로 제거
- Attention (qkv, proj), norm, head 는 1차 구현 제외

---

## 4. 아키텍처 분석 결과

### 모델 하이퍼파라미터

| 모델 | embed_dim | num_heads | num_layers | mlp_dim | 전체 파라미터 |
|------|:---------:|:---------:|:----------:|:-------:|:-----------:|
| **ViT-Tiny**  | 192  | 3  | 12 | 768   | 5.7M  |
| **ViT-Small** | 384  | 6  | 12 | 1,536 | 22.1M |

### 파라미터 그룹 비중

**ViT-Tiny:**
```
G_FFN     3,550,464  62.10%  ← Pruning 대상
G_QKV     1,334,016  23.33%
G_PROJ      444,672   7.78%
G_OTHER     388,264   6.79%
─────────────────────────────
TOTAL     5,717,416  100.00%
```

**ViT-Small:**
```
G_FFN    14,178,816  64.30%  ← Pruning 대상
G_QKV     5,322,240  24.14%
G_PROJ    1,774,080   8.05%
G_OTHER     775,528   3.51%
─────────────────────────────
TOTAL    22,050,664  100.00%
```

### target_compression → G_FFN sparsity (이진탐색, secondary effect 포함)

| target | Tiny sparsity | Small sparsity |
|:------:|:---:|:---:|
| 10% | 0.1608 | 0.1553 |
| 20% | 0.3223 | 0.3109 |
| 30% | 0.4837 | 0.4665 |
| **50%** | **0.8053** | **0.7777** |

> secondary effect: fc1을 s% 제거하면 fc2 입력도 s% 함께 감소  
> → 단순 선형 계산보다 실제 압축률이 더 크므로 이진탐색으로 보정

---

## 5. 구현 파일 설명

### `pruning/vit_pruning.py` — ViTPruner

```python
pruner = ViTPruner(
    model,
    target_compression=0.50,      # 목표 압축률
    max_sparsity=0.95,            # per-group sparsity 상한
    index_refresh_steps=100,      # 마스크 재계산 주기 (step)
)

# 학습 루프 내: optimizer.step() 직후, model_ema.update() 이전
pruner.apply(model)

# epoch 끝 WandB 로깅용
metrics = pruner.log_sparsity(model)
# → {'pruning/actual_sparsity': 0.76, 'pruning/zero_filters': 27840, ...}
```

**동작 원리:**
- `apply()` 는 매 step 호출되어 하위 sparsity% 채널을 0으로 덮어씀
- 100 step마다 `refresh()` 로 마스크 재계산 (gradient가 채널을 살릴 수 있음)
- lazy init: 첫 `apply()` 호출 시 model이 CUDA에 있을 때 그룹 수집 (device mismatch 방지)

### `pruning/vit_reducing.py` — reduce_vit_model

```python
from pruning.vit_reducing import (
    reduce_vit_model,
    get_reduced_config,
    apply_reduced_config,
    transfer_pruning_mask,   # EMA reducing 시 필수
)

# EMA reducing 순서:
#   1) raw model로 dead 채널 결정 (norm == 0 정확)
#   2) EMA model에 zero 패턴 이식
#   3) reduce
transfer_pruning_mask(raw_model, ema_model)
reduce_vit_model(ema_model)
mlp_dims = get_reduced_config(ema_model)  # 저장용
```

**transfer_pruning_mask 가 필요한 이유:**

| weights | dead 채널 값 | 정확히 0? |
|---------|------------|:---:|
| raw model | pruner.apply()로 매 step 리셋 | **예** |
| EMA model | `decay^N × 초기값` 으로 수렴 중 | 아니오 |

→ EMA weights를 그대로 `_survived_idx` (norm != 0 판정)에 넣으면 dead 채널이 살아남음  
→ `transfer_pruning_mask`로 raw의 정확한 zero 패턴을 EMA에 이식한 뒤 reduce

### `engine.py` — 학습/검증 루프

```python
train_stats = train_one_epoch(
    model, criterion, train_loader,
    optimizer, scaler, device, epoch,
    model_ema=model_ema,     # EMA 업데이트 (pruning 후 weight 기준)
    pruner=pruner,           # None이면 일반 학습
    amp=True,
    clip_grad=1.0,
)
val_stats = evaluate(val_loader, model_ema.module, device)
```

pruner 삽입 위치:
```
optimizer.step() → pruner.apply() → model_ema.update()
                        ↑                  ↑
                  gradient 반영 후    pruning 후 weight 추적
```

### `train.py` — 학습 진입점

주요 인자:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data-path` | (필수) | ImageNet 루트 (train/ val/ 포함) |
| `--model` | vit_tiny_patch16_224 | timm 모델 이름 |
| `--epochs` | 30 | 학습 epoch 수 |
| `--batch-size` | 256 | GPU 당 배치 크기 |
| `--lr` | 5e-5 | AdamW learning rate |
| `--target-compression` | 0.0 | 압축률 (0=pruning 비활성) |
| `--prune-refresh-steps` | 100 | 마스크 재계산 주기 |
| `--warmup-epochs` | 5 | LR warmup epoch 수 |
| `--resume` | "" | 체크포인트 재개 경로 |
| `--wandb` | False | WandB 로깅 활성 |
| `--output-dir` | ./output | 체크포인트 저장 디렉터리 |

체크포인트 저장:
- `checkpoint_last.pt` — 매 epoch 덮어씀
- `checkpoint_best.pt` — val top-1 갱신 시만 저장

### `reduce.py` — Reducing CLI

```bash
python reduce.py \
  --model <timm_model_name> \
  --checkpoint <checkpoint_best.pt 경로> \
  --output <reduced.pt 저장 경로>
```

---

## 6. 학습 실행 명령어

### 데이터 경로

```
/workspace/etri_iitp/JS/Server_Compression/data/imagenet/
├── train/   (1,281,167 images, 1000 classes)
└── val/     (50,000 images, 1000 classes)
```

---

### VRAM 기준 배치 사이즈 (GPU 1개당 11GB)

| 모델 | 파라미터 | 권장 batch/GPU | 총 배치 (×4 GPU) | 예상 VRAM |
|------|:-------:|:--------------:|:---------------:|:--------:|
| ViT-Tiny  | 5.7M  | **256** | 1,024 | ~5GB |
| ViT-Small | 22.1M | **128** | 512   | ~8GB |

> AMP(FP16) 기준. OOM 발생 시 절반으로 줄이기.

---

### ViT-Small 50% Pruning (GPU 4,5,6,7 — 현재 진행 중)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --model vit_small_patch16_224 \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --epochs 30 \
  --batch-size 128 \
  --target-compression 0.50 \
  --output-dir ./output/vit_small_prune50 \
  --wandb
```

> 예상: G_FFN sparsity ≈ 0.7777, 제거 파라미터 ≈ 11.0M (22.1M → 11.1M)  
> 총 배치: 128 × 4 = 512

---

### 기타 실험 조합

**ViT-Tiny 50% (GPU 4,5,6,7):**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --model vit_tiny_patch16_224 \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --epochs 30 --batch-size 256 \
  --target-compression 0.50 \
  --output-dir ./output/vit_tiny_prune50 \
  --wandb
```

> 예상: G_FFN sparsity ≈ 0.8053, 제거 파라미터 ≈ 2.9M (5.7M → 2.8M)  
> 총 배치: 256 × 4 = 1024

**ViT-Tiny 30% (GPU 4,5,6,7):**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --model vit_tiny_patch16_224 \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --epochs 30 --batch-size 256 \
  --target-compression 0.30 \
  --output-dir ./output/vit_tiny_prune30 \
  --wandb
```

**ViT-Small 30% (GPU 4,5,6,7):**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --model vit_small_patch16_224 \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --epochs 30 --batch-size 128 \
  --target-compression 0.30 \
  --output-dir ./output/vit_small_prune30 \
  --wandb
```

<!-- 현재 미사용 모델 (필요 시 주석 해제, --model 인자만 변경)
  vit_base_patch16_224  → batch-size 64  (11GB GPU 기준)
  deit_tiny/small_patch16_224 → ViT와 동일 batch-size 사용
-->

**학습 재개 (checkpoint_last.pt 로부터):**
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --model vit_small_patch16_224 \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --epochs 30 --batch-size 128 \
  --target-compression 0.50 \
  --output-dir ./output/vit_small_prune50 \
  --resume ./output/vit_small_prune50/checkpoint_last.pt \
  --wandb
```

---

## 7. Reducing 실행 명령어

학습 완료 후 `checkpoint_best.pt` 를 Dense 모델로 변환:

```bash
python reduce.py \
  --model vit_base_patch16_224 \
  --checkpoint ./output/vit_base_prune50/checkpoint_best.pt \
  --output ./output/vit_base_prune50/reduced.pt
```

실행 결과 예시:
```
[Reducer] EMA weights 사용  (dead 채널 마스크는 raw model 기준)
BEFORE reduce: 86,567,656 params
[Reducer] removed 43,269,624 params across 12 blocks
AFTER  reduce: 43,298,032 params  (49.98% removed)
Forward OK: output shape = (1, 1000)

블록별 reduced mlp_dim:
  block  0: 744
  block  1: 743
  ...
  block 11: 741
```

---

## 8. Reduced 모델 로드 방법

`reduced.pt` 에는 `state_dict`, `model_name`, `mlp_dims` 가 저장된다.  
`mlp_dims` 로 timm 원본 모델 구조를 축소한 뒤 state_dict 를 로드한다.

```python
import torch
import timm
from pruning.vit_reducing import apply_reduced_config

ckpt  = torch.load("reduced.pt", map_location="cpu")
model = timm.create_model(ckpt["model_name"], pretrained=False)
apply_reduced_config(model, ckpt["mlp_dims"])   # 구조 축소
model.load_state_dict(ckpt["state_dict"])        # 가중치 복원
model.eval()

# 추론
with torch.no_grad():
    out = model(torch.zeros(1, 3, 224, 224))     # (1, 1000)
```

---

## 9. WandB 모니터링 지표

| 키 | 내용 |
|----|------|
| `train/loss` | 배치 평균 학습 loss |
| `train/top1` | 배치 평균 학습 Top-1 |
| `train/lr` | 현재 learning rate |
| `val/loss` | 검증 loss |
| `val/top1` | 검증 Top-1 **(핵심 지표)** |
| `val/top5` | 검증 Top-5 |
| `val/top1_best` | 현재까지 최고 val Top-1 |
| `pruning/actual_sparsity` | 전체 prunable 채널 중 실제 zero 비율 |
| `pruning/zero_filters` | zero 채널 수 (절대값) |
| `pruning/target_sparsity` | 설정된 목표 sparsity |
| `pruning/layer/blocks/N/mlp` | 블록별 zero 비율 |

**초기 수렴 확인 체크리스트:**
- epoch 1~2: `pruning/actual_sparsity` 가 `pruning/target_sparsity` 에 근접하는지 확인
- epoch 1: `val/top1` 이 baseline 대비 5% 이상 급락하면 `--target-compression` 낮추기
- epoch 10 이후: `pruning/layer/*` 의 블록별 sparsity가 균등한지 확인

---

## 10. 핵심 설계 결정 사항

### G_FFN 전용 1차 구현
파라미터 비중이 65%로 가장 크고 구현이 단순. G_QKV (fused weight 분리 필요)는 2차 구현.

### 이진탐색 sparsity 계산
secondary effect (fc2 입력도 같이 제거) 를 포함하여 정확한 sparsity 계산.  
단순 선형 계산 `target × total / ffn` 보다 더 정확하다.

### transfer_pruning_mask
EMA weights의 dead 채널은 `decay^N × 초기값` 으로 부동소수점 상 정확히 0이 아님.  
raw model의 zero 패턴을 EMA에 이식 후 reduce. 이 과정 없이 EMA로 reduce하면 모든 채널이 survived로 판정된다.

### index_refresh_steps = 100
매 step 재계산하면 CPU topk 병목 발생. 100 step마다 재계산하면:
- 충분히 자주 업데이트되어 채널 부활/사망이 반영됨
- CPU 병목 완화

### Lazy init (첫 apply() 시 그룹 수집)
`__init__` 시점에는 모델이 CPU에 있고, 학습 시작 시점에는 CUDA로 이동.  
첫 `apply()` 호출 시 그룹을 수집하면 device mismatch 방지.

---

## 11. 주의사항 & 트러블슈팅

### ❶ DDP 환경에서 pruner.apply()
```python
# engine.py에서 이미 처리됨
actual = model.module if hasattr(model, "module") else model
pruner.apply(actual)   # ← 반드시 .module 전달
```

### ❷ Reducing 시 EMA 체크포인트 키 이름 확인
```python
ckpt = torch.load("checkpoint_best.pt", map_location="cpu")
print(ckpt.keys())
# → dict_keys(['model', 'model_ema', 'optimizer', 'lr_scheduler', 'scaler', 'pruner', 'epoch', 'best_acc1', 'args'])
```
`model_ema` 키가 있으면 EMA 자동 사용. 없으면 `model` (raw) 사용.

### ❸ val/top1 급락 시
epoch 1에서 5% 이상 하락하면 sparsity 과도:
```bash
# 0.50 → 0.30 으로 낮추거나
--target-compression 0.30

# warmup을 늘려 초기 학습을 안정화
--warmup-epochs 10
```

### ❹ 아키텍처 분석 재실행
```bash
python measure_memory.py
```

### ❺ 환경 재현
```bash
pip install timm==1.0.27
# torch 2.9.1 + torchvision 0.24.1 은 서버 기존 환경 유지
```

---

*작성: 2026-06 | 서버: `root@59bfae69b3a9` | GPU: 4,5,6,7*
