# ViT Soft Pruning 구현 보고서

> 작성 기준: 2026-06  
> 환경: timm 1.0.27 · torch 2.9.1 · Python 3.13.5  
> 서버: `root@59bfae69b3a9:/workspace/etri_iitp/JS/Server_Compression`  
> 레퍼런스: EfficientViT Soft Pruning (동일 방법론을 timm ViT에 이식)

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [방법론 요약](#2-방법론-요약)
3. [환경 설정](#3-환경-설정)
4. [아키텍처 & 파라미터 분석](#4-아키텍처--파라미터-분석)
5. [구현 파일 설명](#5-구현-파일-설명)
6. [학습 실행 명령어](#6-학습-실행-명령어)
7. [Baseline Evaluation](#7-baseline-evaluation)
8. [Reducing 실행 명령어](#8-reducing-실행-명령어)
9. [ONNX 변환](#9-onnx-변환)
10. [Reduced 모델 로드 방법](#10-reduced-모델-로드-방법)
11. [WandB 모니터링 지표](#11-wandb-모니터링-지표)
12. [핵심 설계 결정 사항](#12-핵심-설계-결정-사항)
13. [주의사항 & 트러블슈팅](#13-주의사항--트러블슈팅)

---

## 1. 프로젝트 구조

```
Server_Compression/
├── configs/                     ← 실험별 YAML config (NEW)
│   ├── vit_tiny_prune50.yaml
│   ├── vit_tiny_prune30.yaml
│   ├── vit_small_prune50.yaml
│   └── vit_small_prune30.yaml
├── pruning/
│   ├── __init__.py
│   ├── vit_pruning.py           ← ViTPruner: Soft Pruning 컨트롤러
│   └── vit_reducing.py          ← reduce_vit_model: Dense 변환
├── engine.py                    ← train_one_epoch / evaluate
├── train.py                     ← 학습 진입점 (단일GPU / DDP, --config 지원)
├── reduce.py                    ← Reducing CLI
├── eval_baseline.py             ← Pruning 전 pretrained 모델 baseline 평가 (NEW)
├── export_onnx.py               ← Reduced 모델 → ONNX 변환 (NEW)
├── measure_memory.py            ← 아키텍처 분석 & 파라미터 프로파일링
├── data/
│   └── imagenet/                ← ImageNet (서버에만 존재, gitignore)
│       ├── train/               (1,281,167 images, 1000 classes)
│       └── val/                 (50,000 images, 1000 classes)
├── output/                      ← 체크포인트 저장 (gitignore)
├── TIMM_PRUNING_GUIDE.md
├── ARCHITECTURE_ANALYSIS.md
├── IMPLEMENTATION.md
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
  → 100 step마다 마스크 재계산 (gradient로 살아난 채널은 제외 가능)

[Reducing — 학습 완료 후]
  zero 채널을 물리적으로 제거 → 실제로 작은 Dense 모델 생성
  fc1: (mlp_dim, embed_dim) → (n_survived, embed_dim)
  fc2: (embed_dim, mlp_dim) → (embed_dim, n_survived)
```

### Pruning 대상: G_FFN (FFN hidden dimension)

```
blocks[i].mlp.fc1: Linear(embed_dim → mlp_dim)   ★ Prunable
                          ↓ GELU
blocks[i].mlp.fc2: Linear(mlp_dim → embed_dim)   ★ Prunable (coupled)
```

- fc1 출력과 fc2 입력이 동일한 `mlp_dim` 채널을 공유 → 같은 인덱스로 동시 제거
- Attention (qkv, proj), norm, head 는 1차 구현 제외

---

## 3. 환경 설정

```bash
# 의존성 설치
pip install timm==1.0.27
pip install wandb
pip install onnx onnxruntime   # ONNX 변환 시 필요

# 모델 사전 다운로드 (최초 1회, 서버에서 실행)
python -c "
import timm
timm.create_model('vit_tiny_patch16_224',  pretrained=True)
timm.create_model('vit_small_patch16_224', pretrained=True)
print('done')
"
```

> **중요**: timm pretrained 모델은 모델별로 mean/std/crop_pct 가 다름.  
> `timm.data.resolve_model_data_config(model)` 로 모델 권장값을 사용해야 함.  
> vit_tiny/small 은 ImageNet 표준 `(0.485, 0.456, 0.406)` 이 아닌 `(0.5, 0.5, 0.5)` 사용.

---

## 4. 아키텍처 & 파라미터 분석

> 분석 스크립트: `python measure_memory.py`

### 모델 기본 스펙

| 모델 | embed_dim | mlp_dim | num_heads | blocks | 전체 파라미터 |
|------|:---------:|:-------:|:---------:|:------:|:-----------:|
| ViT-Tiny  | 192 | 768   | 3  | 12 | **5,717,416** |
| ViT-Small | 384 | 1,536 | 6  | 12 | **22,050,664** |

### 파라미터 그룹 분류 (measure_memory.py 실측값)

**ViT-Tiny:**
```
G_FFN     3,550,464   62.10%  ← Pruning 대상 (fc1.weight/bias + fc2.weight)
G_QKV     1,334,016   23.33%
G_PROJ      444,672    7.78%
G_NORM        9,600    0.17%
G_HEAD      193,000    3.38%
G_EMBED     147,648    2.58%
G_OTHER      38,016    0.66%
──────────────────────────────
TOTAL     5,717,416  100.00%
```

**ViT-Small:**
```
G_FFN    14,178,816   64.30%  ← Pruning 대상 (fc1.weight/bias + fc2.weight)
G_QKV     5,322,240   24.14%
G_PROJ    1,774,080    8.05%
G_NORM       19,200    0.09%
G_HEAD      385,000    1.75%
G_EMBED     295,296    1.34%
G_OTHER      76,032    0.34%
──────────────────────────────
TOTAL    22,050,664  100.00%
```

### 50% 압축 달성을 위한 FFN Sparsity (정확한 파라미터 역산값)

> 이진탐색 (64회 반복) + secondary effect (fc2 column도 동시 감소) 포함한 정확한 계산값.  
> 채널 하나당 제거 파라미터 = 2 × embed_dim + 1 (fc1.weight행 + fc1.bias + fc2.weight열)

| 모델 | embed_dim | 채널당 제거 params | n_prune (50% 목표) | FFN sparsity | 실제 제거 params | 실제 압축률 |
|------|:---------:|:------------------:|:------------------:|:------------:|:---------------:|:---------:|
| ViT-Tiny  | 192 | 2×192+1 = **385** | 618 / 768  | **0.8053** | 2,855,160 | **49.94%** |
| ViT-Small | 384 | 2×384+1 = **769** | 1195 / 1536 | **0.7777** | 11,027,460 | **50.01%** |

**전체 target_compression 테이블 (실측값):**

| target | Tiny sparsity | Tiny 제거 params | Small sparsity | Small 제거 params |
|:------:|:---:|:---:|:---:|:---:|
| 10% | 0.1608 | 568,260 | 0.1553 | 2,196,264 |
| 20% | 0.3223 | 1,145,760 | 0.3109 | 4,401,756 |
| 30% | 0.4837 | 1,714,020 | 0.4665 | 6,616,476 |
| **50%** | **0.8053** | **2,855,160** | **0.7777** | **11,027,460** |

> 실제 압축 후 모델 크기:  
> ViT-Tiny 50%: 5.72M → **2.86M** params  
> ViT-Small 50%: 22.1M → **11.0M** params

---

## 5. 구현 파일 설명

### `configs/*.yaml` — 실험별 Config

```yaml
# 예: configs/vit_tiny_prune50.yaml
model: vit_tiny_patch16_224
data_path: /workspace/etri_iitp/JS/Server_Compression/data/imagenet
epochs: 50
batch_size: 256        # per GPU
lr: 5.0e-5
target_compression: 0.50
output_dir: ./output/vit_tiny_prune50
wandb: true
```

모든 하이퍼파라미터가 YAML에 집중되며, CLI 인자로 개별 override 가능:
```bash
# config 기본값 사용
torchrun ... train.py --config configs/vit_tiny_prune50.yaml

# epochs만 override
torchrun ... train.py --config configs/vit_tiny_prune50.yaml --epochs 30
```

---

### `pruning/vit_pruning.py` — ViTPruner

```python
pruner = ViTPruner(
    model,
    target_compression=0.50,
    max_sparsity=0.95,
    index_refresh_steps=100,
)

# 학습 루프: optimizer.step() 직후, model_ema.update() 이전
pruner.apply(model)

# WandB 로깅
metrics = pruner.log_sparsity(model)
# → {'pruning/actual_sparsity': 0.8053, 'pruning/zero_filters': 8856, ...}
```

---

### `pruning/vit_reducing.py` — reduce_vit_model

```python
# EMA reducing 순서
transfer_pruning_mask(raw_model, ema_model)  # ★ raw의 zero 패턴 이식
reduce_vit_model(ema_model)
mlp_dims = get_reduced_config(ema_model)
```

| 항목 | raw model | EMA model |
|------|:---:|:---:|
| dead 채널 값 | 정확히 0 (매 step pruner 적용) | `decay^N × 초기값` (근사 0) |
| `_survived_idx` 판정 | 정확 | 오판 가능 → 모든 채널 survived 처리됨 |

→ `transfer_pruning_mask` 로 raw 의 zero 패턴을 EMA 에 이식한 뒤 reduce.

---

### `train.py` — 학습 진입점

주요 인자:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--config` | "" | YAML config 경로 |
| `--model` | vit_tiny_patch16_224 | timm 모델 이름 |
| `--epochs` | 50 | 학습 epoch 수 |
| `--batch-size` | 256 | GPU 당 배치 크기 |
| `--lr` | 5e-5 | AdamW learning rate |
| `--target-compression` | 0.0 | 압축률 (0=pruning 비활성) |
| `--prune-refresh-steps` | 100 | 마스크 재계산 주기 |
| `--warmup-epochs` | 5 | LR warmup epoch 수 |
| `--resume` | "" | 체크포인트 재개 경로 |
| `--wandb` | False | WandB 로깅 활성 |

체크포인트:
- `checkpoint_last.pt` — 매 epoch 덮어씀
- `checkpoint_best.pt` — val top-1 갱신 시만 저장

---

### `eval_baseline.py` — Pruning 전 Baseline 평가

Pruning 전 pretrained 모델의 정확도를 WandB에 기록. 모델별 권장 data config 자동 적용.

---

### `reduce.py` — Reducing CLI

### `export_onnx.py` — ONNX 변환

---

## 6. 학습 실행 명령어

### 데이터 경로

```
/workspace/etri_iitp/JS/Server_Compression/data/imagenet/
├── train/   (1,281,167 images, 1000 classes)
└── val/     (50,000 images, 1000 classes)
```

### GPU & 배치 사이즈 (GPU 1개당 11GB VRAM)

| 모델 | batch/GPU | GPU 구성 | 총 배치 |
|------|:---------:|:--------:|:-------:|
| ViT-Tiny  | 256 | 4,5,6,7 (×4) | 1,024 |
| ViT-Small | 128 | 4,5,6,7 (×4) | 512   |

---

### ViT-Tiny 50% Pruning (GPU 4,5,6,7)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --config configs/vit_tiny_prune50.yaml
```

> FFN sparsity: **0.8053** (실측) | 제거: 2,855,160 params | 5.72M → **2.86M**  
> 총 배치: 256 × 4 = 1,024

---

### ViT-Small 50% Pruning (GPU 4,5,6,7)

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --config configs/vit_small_prune50.yaml
```

> FFN sparsity: **0.7777** (실측) | 제거: 11,027,460 params | 22.1M → **11.0M**  
> 총 배치: 128 × 4 = 512

---

### 기타 실험

```bash
# ViT-Tiny 30%
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --config configs/vit_tiny_prune30.yaml

# ViT-Small 30%
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --config configs/vit_small_prune30.yaml
```

<!-- 현재 미사용 (필요 시 --model 인자만 변경)
  vit_base_patch16_224  → batch-size 64 (11GB 기준)
  deit_tiny/small_patch16_224 → 동일 코드 사용 가능
-->

---

### 학습 재개

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py \
  --config configs/vit_tiny_prune50.yaml \
  --resume ./output/vit_tiny_prune50/checkpoint_last.pt
```

---

## 7. Baseline Evaluation

Pruning 전 pretrained 모델의 baseline top-1/top-5 를 WandB에 기록.

```bash
# 두 모델 동시 평가
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 eval_baseline.py \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --batch-size 256 \
  --wandb

# 단일 모델
python eval_baseline.py \
  --model vit_tiny_patch16_224 \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --wandb
```

**기대 baseline 수치 (timm pretrained):**

| 모델 | top-1 | top-5 | mean/std |
|------|:-----:|:-----:|:--------:|
| ViT-Tiny  | ~75.5% | ~92.4% | (0.5, 0.5, 0.5) |
| ViT-Small | ~81.4% | ~95.8% | (0.5, 0.5, 0.5) |

---

## 8. Reducing 실행 명령어

학습 완료 후 `checkpoint_best.pt` 를 Dense 모델로 변환:

```bash
# ViT-Tiny
python reduce.py \
  --model vit_tiny_patch16_224 \
  --checkpoint ./output/vit_tiny_prune50/checkpoint_best.pt \
  --output ./output/vit_tiny_prune50/reduced.pt

# ViT-Small
python reduce.py \
  --model vit_small_patch16_224 \
  --checkpoint ./output/vit_small_prune50/checkpoint_best.pt \
  --output ./output/vit_small_prune50/reduced.pt
```

실행 결과 예시 (ViT-Tiny 50%):
```
[Reducer] EMA weights 사용
BEFORE: 5,717,416 params
AFTER:  2,862,256 params  (49.94% removed)

블록별 survived mlp_dim:
  block  0: 150 / 768
  block  1: 152 / 768
  ...
```

---

## 9. ONNX 변환

```bash
pip install onnx onnxruntime

python export_onnx.py \
  --reduced ./output/vit_tiny_prune50/reduced.pt \
  --output  ./output/vit_tiny_prune50/reduced.onnx \
  --verify
```

- `--dynamic`: 배치 차원 가변 (기본 활성)
- `--verify`: onnxruntime vs PyTorch 출력값 비교 (최대 차이 출력)
- opset 17, constant folding 적용

---

## 10. Reduced 모델 로드 방법

```python
import torch, timm
from pruning.vit_reducing import apply_reduced_config

ckpt  = torch.load("reduced.pt", map_location="cpu")
model = timm.create_model(ckpt["model_name"], pretrained=False)
apply_reduced_config(model, ckpt["mlp_dims"])   # 구조 축소 후
model.load_state_dict(ckpt["state_dict"])        # 가중치 복원
model.eval()

with torch.no_grad():
    out = model(torch.zeros(1, 3, 224, 224))    # (1, 1000)
```

---

## 11. WandB 모니터링 지표

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
| `baseline/top1` | Pruning 전 pretrained 기준 top-1 |

**수렴 확인 체크리스트:**
- epoch 1: `pruning/actual_sparsity` 가 target에 근접하는지 확인
- epoch 5: LR warmup 종료 후 val/top1 회복 추세 확인
- epoch 10+: 블록별 sparsity가 균등한지 확인

---

## 12. 핵심 설계 결정 사항

### 정확한 sparsity 계산 (이진탐색 + secondary effect)

채널 하나를 제거할 때 실제로 제거되는 파라미터:
```
fc1.weight 1행: embed_dim 개
fc1.bias   1개: 1 개
fc2.weight 1열: embed_dim 개 (← secondary effect)
─────────────────────────────
합계: 2 × embed_dim + 1 개
```

단순 선형 계산(`target × total / G_FFN`) 대비 이진탐색이 더 정확.

### transfer_pruning_mask

EMA weights의 dead 채널은 `decay^N × 초기값` (정확히 0이 아님).  
raw model의 zero 패턴을 EMA에 이식한 뒤 reduce.

### resolve_model_data_config

timm 모델마다 권장 normalization이 다름:
- ViT-Tiny/Small (AugReg): `mean=std=(0.5, 0.5, 0.5)`, `crop_pct=0.9`
- 하드코딩 시 정확도가 크게 하락함 (vit_tiny 기준 ~75% → ~44%)

### Lazy init

`ViTPruner.__init__` 시점엔 model이 CPU에 있음.  
첫 `apply()` 호출 시 그룹 수집 → device mismatch 방지.

---

## 13. 주의사항 & 트러블슈팅

### ❶ Normalization 불일치

```python
# 잘못된 방법 — 모든 모델에 ImageNet 표준 적용
mean=IMAGENET_DEFAULT_MEAN  # (0.485, 0.456, 0.406)

# 올바른 방법 — 모델 권장값 사용 (train.py, eval_baseline.py 모두 적용됨)
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
```

### ❷ DDP 환경에서 pruner.apply()

```python
# engine.py 에서 처리됨
actual = model.module if hasattr(model, "module") else model
pruner.apply(actual)   # 반드시 .module 전달
```

### ❸ val/top1 급락 시 (50% 압축 초기는 정상)

epoch 0: top1 ~2% 는 정상 (FFN 80% 제거 직후 충격)  
epoch 5 이후에도 20% 미만이면 압축률 낮추기:
```bash
# config 내 target_compression: 0.30 으로 변경
# 또는 CLI override:
--target-compression 0.30
```

### ❹ 체크포인트 키 확인

```python
ckpt = torch.load("checkpoint_best.pt")
print(ckpt.keys())
# → ['model', 'model_ema', 'optimizer', 'lr_scheduler', 'scaler', 'pruner', 'epoch', 'best_acc1', 'args']
```

### ❺ 아키텍처 분석 재실행

```bash
python measure_memory.py
```

---

*작성: 2026-06 | 서버: `root@59bfae69b3a9` | GPU: 4,5,6,7*
