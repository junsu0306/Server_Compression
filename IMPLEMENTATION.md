# ViT Soft Pruning 구현 보고서

> 작성 기준: 2026-07  
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
├── configs/
│   ├── vit_tiny_prune50.yaml                    ← Global + KD
│   ├── vit_tiny_prune30.yaml
│   ├── vit_small_prune50.yaml
│   ├── vit_small_prune30.yaml
│   ├── vit_tiny_prune50_progressive.yaml        ← Global + KD + Progressive + Taylor EMA
│   └── vit_small_prune50_progressive.yaml       ← Global + KD + Progressive + Taylor EMA
├── pruning/
│   ├── __init__.py
│   ├── vit_pruning.py           ← ViTPruner: Soft Pruning 컨트롤러
│   └── vit_reducing.py          ← reduce_vit_model: Dense 변환
├── engine.py                    ← train_one_epoch / evaluate
├── train.py                     ← 학습 진입점 (단일GPU / DDP, --config 지원)
├── reduce.py                    ← Reducing CLI
├── eval_baseline.py             ← Pruning 전 pretrained 모델 baseline 평가
├── eval_reduced.py              ← Reduced 모델 val 평가 → WandB test 기록
├── export_onnx.py               ← Reduced 모델 → ONNX 변환
├── measure_memory.py            ← 아키텍처 분석 & 파라미터 프로파일링
├── data/
│   └── imagenet/                ← ImageNet (서버에만 존재, gitignore)
│       ├── train/               (1,281,167 images, 1000 classes)
│       └── val/                 (50,000 images, 1000 classes)
├── output/                      ← 체크포인트 저장 (gitignore)
└── IMPLEMENTATION.md
```

---

## 2. 방법론 요약

### Soft Pruning → Reducing 2단계 파이프라인

```
[Soft Pruning — 학습 중]
  매 optimizer.step() 직후:
    fc1.weight의 중요도 하위 X% 행(row) → 0으로 마스킹
    fc2.weight의 동일 인덱스 열(col)    → 0으로 마스킹
  → 아키텍처 구조는 Dense 그대로 유지
  → 100 step마다 마스크 재계산

[Reducing — 학습 완료 후]
  zero 채널을 물리적으로 제거 → 실제로 작은 Dense 모델 생성
  fc1: (mlp_dim, embed_dim) → (n_survived, embed_dim)  ← 블록마다 n_survived 다름
  fc2: (embed_dim, mlp_dim) → (embed_dim, n_survived)
```

### 1 step 전체 학습 순서 (engine.py)

매 배치마다 아래 순서로 실행된다. 순서가 바뀌면 EMA가 pruning 이전 weight를 학습하거나 pruning이 무효화된다.

```
① samples, targets 로드

② [Student forward]  output = model(samples)
   └─ 이 시점 FFN 채널의 일부는 이미 0인 상태 (progressive: 점진적 증가)

③ CE Loss 계산
   loss = CrossEntropyLoss(output, targets)

④ [Teacher forward]  teacher_logits = teacher(samples)  ← torch.no_grad()
   └─ frozen 원본 모델, gradient 없음

⑤ KD Loss 계산 + 합산
   kd_loss = KL(student/T ‖ teacher/T) × T²
   loss = 0.5 × CE + 0.5 × KD

⑥ optimizer.zero_grad()
   └─ 이전 step gradient 초기화

⑦ Backward
   loss.backward()
   └─ gradient는 Student에만 흐름, Teacher 쪽 없음
   └─ param.grad에 현재 배치 기준 gradient 저장

⑧ scaler.unscale_(optimizer)  ← clip_grad > 0일 때
   └─ AMP scale 제거 → param.grad가 실제 gradient 값이 됨

⑨ optimizer.step()
   └─ gradient 반영 → weight 갱신
   └─ 이 시점: dead 채널이 gradient에 의해 일시적으로 살아날 수 있음

⑩ pruner.apply()  ★ 핵심 ★
   └─ _channel_importance() 호출 → Taylor EMA 업데이트 (importance=taylor 시)
   └─ fc1.weight, fc1.bias, fc2.weight의 마스크 대상 위치를 다시 0으로 강제
   └─ tensor.data.mul_(mask) — autograd를 우회해 직접 덮어씀

⑪ model_ema.update()
   └─ pruning 후 weight 기준으로 shadow weight 갱신
   └─ 검증 및 최종 reduce에 이 EMA weight 사용
```

> ⑨→⑩이 핵심: optimizer가 dead 채널을 살리더라도 pruner가 즉시 다시 0으로 덮어써
> "soft"하게 죽은 상태를 유지한다. 100 step마다 마스크를 재계산하므로
> gradient 신호가 꾸준히 강한 채널은 마스크에서 살아남을 수 있다.

---

### Pruning 모드: Uniform vs Global (Non-uniform)

| 모드 | 동작 | 특징 |
|------|------|------|
| `uniform` | 각 블록 독립적으로 하위 sparsity% 제거 | 모든 블록 동일 비율 |
| `global` (**기본값**) | 전체 블록 채널을 global 중요도 랭킹으로 선택 | 중요 블록은 덜 잘리고, 중복 많은 블록은 더 잘림 |

```
[Global mode 동작]
  1. 모든 블록의 fc1.weight row 중요도 계산 (L2 or Taylor EMA)
     block 0: [0.82, 0.03, 1.24, ...]
     block 5: [0.91, 0.02, 0.07, ...]
     ...  (총 12 × 768 = 9,216개 score)

  2. 전체를 한번에 정렬 → 하위 N개를 globally 선택
     단, 블록당 max_sparsity(0.95) 상한 적용
     → 상한 초과분은 다른 블록에서 추가 제거

  3. 결과: 블록마다 다른 sparsity (자동 non-uniform)
     block 0: 58% 제거  ← 중요, 덜 잘림
     block 5: 92% 제거  ← 중복 많음
     block 11: 71% 제거
     전체 총 제거 채널 수 = uniform과 동일 (압축률 보장)
```

---

### Knowledge Distillation (KD)

Soft Pruning과 병행하여 압축 후 정확도를 높이기 위해 KD를 추가 지원.

```
[KD Loss]
  Teacher: 원본 pretrained 모델 (frozen, eval mode)
  Student: 현재 학습 중인 pruned 모델

  loss = (1 - α) × CE(student, hard_label)
       +      α  × KL(student_logits/T ‖ teacher_logits/T) × T²
              ↑                                               ↑
           KD 가중치 (α=0.5 권장)              Temperature scaling 보정

T (temperature): 높을수록 teacher softmax 분포가 부드러워짐
  → 클래스 간 유사성 정보가 student에게 더 잘 전달됨
  → 권장: T=4.0
```

---

### Progressive Pruning

기존 방식은 epoch 0 첫 step에서 목표 sparsity를 즉시 적용하여 모델이 초기에 큰 충격을 받는다.
Progressive Pruning은 sparsity를 점진적으로 증가시켜 이 문제를 해소한다.

```
[기존 방식]
  epoch 0 첫 배치: FFN 80% 즉시 제거
  → top-1 ~2% 급락 후 50 epoch 내내 회복에 소비

[Progressive Pruning — Zhu & Gupta 2018 cubic schedule]
  epoch 0~4:   sparsity = 0%      (LR warmup과 동기화, 정상 학습)
  epoch 5~24:  sparsity: 0% → target (cubic ease-out으로 점진 증가)
  epoch 25~49: sparsity = target  (수렴 단계)

cubic ease-out 수식:
  progress = (epoch - warmup) / ramp_epochs    ← 0~1
  sparsity = target × (1 - (1 - progress)³)   ← 초반 빠르게, 후반 완만하게
```

진행 상황은 매 epoch 로그에 출력된다:
```
[ViTPruner] epoch=10  sparsity: 0.3500 → 0.5200  (66.8% of target)
```

---

### Taylor Criterion + Gradient EMA (채널 중요도 기준)

#### L2 vs Taylor 비교

| 기준 | 수식 | 의미 | 특징 |
|------|------|------|------|
| L2 norm | `‖fc1.weight‖₂` 채널별 | weight 크기 | 빠르고 안정적 |
| Taylor | `\|w × ∇w\|` 채널합 | loss에 대한 기여도 1차 근사 | 정확하지만 noisy |

Taylor는 "이 채널을 제거하면 loss가 얼마나 바뀌는가"를 gradient × weight로 근사한다.
L2는 weight가 크더라도 gradient가 0이면 loss에 기여가 없음을 포착하지 못한다.

#### Gradient EMA (Taylor 안정화)

Single-batch gradient는 배치 구성에 따라 크게 흔들린다.
특히 ViT-Small (batch=128, embed_dim=384)처럼 배치가 작고 모델이 클수록 노이즈가 심하다.

```
[Gradient EMA 동작]
  매 _channel_importance() 호출 시 (=100 step마다):

  taylor_now = |fc1.weight × ∇fc1.weight|.sum(dim=1)  ← 현재 배치 기준
  grad_ema  = β × grad_ema + (1-β) × taylor_now        ← EMA 누적

  반환값 = grad_ema  (누적 평균 기준으로 채널 중요도 결정)

β = 0.9: 최근 10 step 정도의 gradient를 가중 평균
```

**추가 연산 없음**: backprop gradient를 재활용하므로 extra forward/backward 패스 불필요.
**AMP scale 안전**: AMP GradScaler의 scale 값은 모든 채널에 공통이므로 상대 랭킹에 영향 없음.
**Resume 시 초기화**: `load_state_dict()` 시 `_grad_ema`는 초기화되고 첫 step부터 재누적.

#### ViT-Small에서 Taylor 필요성

| 모델 | embed_dim | batch/GPU | Taylor score 합산 차원 | gradient 노이즈 수준 |
|------|:---------:|:---------:|:----------------------:|:-------------------:|
| Tiny | 192 | 256 | 192 | 낮음 → L2와 큰 차이 없음 |
| Small | 384 | 128 | 384 | 높음 → EMA 없이 epoch 20에서 붕괴 |

Tiny는 모델이 작아 채널 중복이 많으므로 잘못 제거해도 회복 가능하다.
Small은 채널 간 역할 분담이 뚜렷해서 noisy Taylor로 중요 채널을 제거하면 cascading failure가 발생한다.

---

## 3. 환경 설정

```bash
pip install timm==1.0.27
pip install wandb
pip install onnx onnxruntime

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

### 50% 압축 달성을 위한 FFN Sparsity

> 이진탐색 (64회 반복) + secondary effect (fc2 column도 동시 감소) 포함.  
> 채널 하나당 제거 파라미터 = 2 × embed_dim + 1 (fc1.weight행 + fc1.bias + fc2.weight열)

| 모델 | 채널당 제거 params | n_prune (50% 목표) | FFN sparsity | 실제 압축률 |
|------|:------------------:|:------------------:|:------------:|:---------:|
| ViT-Tiny  | 2×192+1 = **385** | 618 / 768  | **0.8053** | **49.94%** |
| ViT-Small | 2×384+1 = **769** | 1195 / 1536 | **0.7777** | **50.01%** |

**전체 target_compression 테이블:**

| target | Tiny sparsity | Small sparsity |
|:------:|:---:|:---:|
| 10% | 0.1608 | 0.1553 |
| 20% | 0.3223 | 0.3109 |
| 30% | 0.4837 | 0.4665 |
| **50%** | **0.8053** | **0.7777** |

> 실제 압축 후 모델 크기:  
> ViT-Tiny 50%: 5.72M → **2.86M** params  
> ViT-Small 50%: 22.1M → **11.0M** params

---

## 5. 구현 파일 설명

### `configs/*.yaml` — 실험별 Config

```yaml
# configs/vit_tiny_prune50_progressive.yaml (현재 권장 설정)
model: vit_tiny_patch16_224
epochs: 50
batch_size: 256
lr: 5.0e-5
target_compression: 0.50
pruning_max_sparsity: 0.95
pruning_mode: global
pruning_importance: taylor    # L2 → Taylor EMA (gradient × weight)
prune_warmup_epochs: 5        # epoch 0~4: pruning 없이 정상 학습
prune_ramp_epochs: 20         # epoch 5~24: 0% → target 점진적 증가
kd_alpha: 0.5
kd_temperature: 4.0
output_dir: ./output/vit_tiny_prune50_progressive_taylor
wandb_run_name: "vit_tiny_prune50_progressive_taylor"
```

---

### `pruning/vit_pruning.py` — ViTPruner

```python
pruner = ViTPruner(
    model,
    target_compression=0.50,
    max_sparsity=0.95,
    index_refresh_steps=100,
    mode="global",             # "global"(non-uniform) | "uniform"
    importance="taylor",       # "l2"(magnitude) | "taylor"(gradient EMA)
    grad_ema_beta=0.9,         # Taylor EMA 감쇠율 (최근 ~10 step 평균)
    warmup_epochs=5,           # progressive: 유예 epoch
    ramp_epochs=20,            # progressive: 점진 증가 epoch
)

# 에포크 시작 전 (progressive sparsity 업데이트)
pruner.set_epoch(epoch)

# 학습 루프: optimizer.step() 직후, model_ema.update() 이전
pruner.apply(model)

# WandB 로깅
metrics = pruner.log_sparsity(model)
```

**내부 구조 — `_PruneGroup` (블록 1개당 1개 생성):**

```python
_PruneGroup(
    criterion = mlp.fc1.weight,        # 중요도 계산 기준 텐서

    targets = [
        (mlp.fc1.weight, dim=0, 0.0),  # fc1 행(row) 마스킹
        (mlp.fc1.bias,   dim=0, 0.0),  # fc1 bias 마스킹
        (mlp.fc2.weight, dim=1, 0.0),  # fc2 열(col) 마스킹 ← secondary effect
    ]
)
```

**`_channel_importance()` 동작 흐름:**

```
importance="taylor" 이고 grad 있음:
  taylor_now = |fc1.weight × ∇fc1.weight|.sum(dim=1)
  grad_ema[id] = 0.9 × grad_ema[id] + 0.1 × taylor_now
  반환: grad_ema[id]

importance="taylor" 이고 grad 없음 (첫 step 또는 epoch 시작):
  grad_ema[id] 가 있으면: 기존 EMA 반환
  grad_ema[id] 없으면:   L2 fallback

importance="l2":
  반환: ‖fc1.weight‖₂  채널별
```

**마스크 적용 메커니즘 (`_PruneGroup.apply()`):**

```python
tensor.data.mul_(mask)   # ★ .data: autograd 우회, in-place로 직접 덮어씀
```

`.data`를 쓰는 이유: `tensor *= mask`는 autograd graph에 연산을 추가하지만,
`tensor.data.mul_(mask)`는 graph를 건드리지 않고 메모리 값만 덮어쓴다.
optimizer.step() 이후에 gradient graph 손상 없이 weight를 강제로 0으로 만들 수 있다.

**Progressive sparsity 스케줄:**

```python
def _scheduled_sparsity(self, epoch):
    if epoch < warmup_epochs:
        return 0.0
    progress = (epoch - warmup_epochs) / ramp_epochs   # 0~1
    return target × (1 - (1 - progress)³)              # cubic ease-out
```

**Global mode per-block cap 동작:**
```
블록 5의 max_prune = round(768 × 0.95) = 729개
  → 블록 5가 730개를 잘라야 한다면 729개까지만 잘림
  → 초과 1개는 다른 블록(여유 있는 블록)의 낮은 중요도 채널이 대신 제거됨
  → 총 제거 채널 수 = 동일하게 유지 (압축률 보장)
```

---

### `pruning/vit_reducing.py` — reduce_vit_model

```python
# EMA reducing 순서
transfer_pruning_mask(raw_model, ema_model)  # raw의 zero 패턴 이식
reduce_vit_model(ema_model)
mlp_dims = get_reduced_config(ema_model)
```

| 항목 | raw model | EMA model |
|------|:---:|:---:|
| dead 채널 값 | 정확히 0 (매 step pruner 적용) | `decay^N × 초기값` (근사 0) |
| `_survived_idx` 판정 | 정확 | 오판 가능 → 모든 채널 survived 처리됨 |

→ `transfer_pruning_mask`로 raw의 zero 패턴을 EMA에 이식한 뒤 reduce.

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
| `--pruning-mode` | global | `global`=non-uniform \| `uniform`=균일 |
| `--pruning-importance` | l2 | `l2`=weight크기 \| `taylor`=gradient EMA |
| `--pruning-max-sparsity` | 0.95 | 블록당 최대 sparsity 상한 |
| `--prune-refresh-steps` | 100 | 마스크 재계산 주기 |
| `--prune-warmup-epochs` | 0 | pruning 유예 epoch (0=즉시 적용) |
| `--prune-ramp-epochs` | 0 | sparsity 점진 증가 epoch (0=즉시 target) |
| `--kd-alpha` | 0.0 | KD loss 가중치 (0=비활성, 0.5 권장) |
| `--kd-temperature` | 4.0 | KD soft label 온도 (권장: 3~5) |
| `--kd-teacher` | "" | Teacher 모델명 (비어있으면 student와 동일) |
| `--warmup-epochs` | 5 | LR warmup epoch 수 |
| `--resume` | "" | 체크포인트 재개 경로 |
| `--wandb` | False | WandB 로깅 활성 |

체크포인트:
- `checkpoint_last.pt` — 매 epoch 덮어씀
- `checkpoint_best.pt` — val top-1 갱신 시만 저장

---

### `eval_reduced.py` — Reduced 모델 평가

`reduce.py`로 생성한 `reduced.pt`를 ImageNet val로 평가하고 WandB에 `test/*` 지표 기록.

```bash
CUDA_VISIBLE_DEVICES=4 python eval_reduced.py \
  --reduced   ./output/vit_tiny_prune50_progressive_taylor/reduced.pt \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --wandb \
  --wandb-run-name vit_tiny_prune50_progressive_taylor_test
```

기록 지표: `test/top1`, `test/top5`, `test/loss`, `test/n_params`, `test/compression_pct`

---

## 6. 학습 실행 명령어

### 데이터 경로

```
/workspace/etri_iitp/JS/Server_Compression/data/imagenet/
├── train/   (1,281,167 images, 1000 classes)
└── val/     (50,000 images, 1000 classes)
```

### GPU & 배치 사이즈

| 모델 | batch/GPU | GPU 구성 | 총 배치 |
|------|:---------:|:--------:|:-------:|
| ViT-Tiny  | 256 | 6,7 (×2) | 512 |
| ViT-Small | 128 | 4,5 (×2) | 256 |

---

### [현재 권장] Progressive + Taylor EMA

```bash
# ViT-Tiny (GPU 6,7)
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train.py \
  --config configs/vit_tiny_prune50_progressive.yaml

# ViT-Small (GPU 4,5)
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train.py \
  --config configs/vit_small_prune50_progressive.yaml
```

output: `./output/vit_{tiny,small}_prune50_progressive_taylor/`

---

### [기존] Global + KD (즉시 full sparsity)

```bash
# ViT-Tiny 50%
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train.py \
  --config configs/vit_tiny_prune50.yaml

# ViT-Small 50%
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 train.py \
  --config configs/vit_small_prune50.yaml
```

---

### 학습 재개

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train.py \
  --config configs/vit_tiny_prune50_progressive.yaml \
  --resume ./output/vit_tiny_prune50_progressive_taylor/checkpoint_last.pt
```

---

## 7. Baseline Evaluation

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 eval_baseline.py \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --batch-size 256 \
  --wandb
```

**기대 baseline 수치 (timm pretrained):**

| 모델 | top-1 | top-5 | mean/std |
|------|:-----:|:-----:|:--------:|
| ViT-Tiny  | ~75.5% | ~92.4% | (0.5, 0.5, 0.5) |
| ViT-Small | ~81.4% | ~95.8% | (0.5, 0.5, 0.5) |

---

## 8. Reducing 실행 명령어

학습 완료 후 `checkpoint_best.pt`를 Dense 모델로 변환:

```bash
# ViT-Tiny
python reduce.py \
  --model vit_tiny_patch16_224 \
  --checkpoint ./output/vit_tiny_prune50_progressive_taylor/checkpoint_best.pt \
  --output     ./output/vit_tiny_prune50_progressive_taylor/reduced.pt

# ViT-Small
python reduce.py \
  --model vit_small_patch16_224 \
  --checkpoint ./output/vit_small_prune50_progressive_taylor/checkpoint_best.pt \
  --output     ./output/vit_small_prune50_progressive_taylor/reduced.pt
```

실행 결과 예시 (ViT-Tiny 50%, global mode):
```
[Reducer] EMA weights 사용
BEFORE: 5,717,416 params
AFTER:  2,862,256 params  (49.94% removed)

블록별 survived mlp_dim (non-uniform):
  block  0: 320 / 768  ← 중요, 많이 살아남음
  block  5:  58 / 768  ← 중복 많음, 많이 제거됨
  ...
```

---

## 9. ONNX 변환

```bash
python export_onnx.py \
  --reduced ./output/vit_tiny_prune50_progressive_taylor/reduced.pt \
  --output  ./output/vit_tiny_prune50_progressive_taylor/reduced.onnx \
  --verify
```

- `--dynamic`: 배치 차원 가변 (기본 활성)
- `--verify`: onnxruntime vs PyTorch 출력값 비교
- `--num-threads N`: 추론 스레드 수 (0=auto, 기본값)
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
| `pruning/current_sparsity` | 현재 적용 중인 scheduled sparsity (progressive에서 변함) |
| `pruning/target_sparsity` | 최종 목표 sparsity (이진탐색 기준값) |
| `pruning/zero_filters` | zero 채널 수 (절대값) |
| `pruning/layer/blocks/N/mlp` | 블록별 zero 비율 (global mode → 블록마다 다름) |
| `pruning/survived/blocks/N/mlp` | 블록별 생존 채널 수 (절대값) |
| `pruning/layer_sparsity` | 블록별 sparsity 한눈에 보기 (bar chart) |

**Progressive Pruning 확인 포인트:**
- `pruning/current_sparsity` 가 epoch마다 증가하는지 확인 (epoch 5~24)
- epoch 25부터 `current_sparsity ≈ target_sparsity` 로 고정되는지 확인
- val/top1이 epoch 0~4 구간에서 상대적으로 안정적인지 확인 (warmup 효과)

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

단순 선형 계산 대비 이진탐색이 더 정확 (특히 small embed_dim에서 차이 큼).

### transfer_pruning_mask

EMA weights의 dead 채널은 `decay^N × 초기값` (정확히 0이 아님).  
raw model의 zero 패턴을 EMA에 이식한 뒤 reduce.

### resolve_model_data_config

timm 모델마다 권장 normalization이 다름:
- ViT-Tiny/Small (AugReg): `mean=std=(0.5, 0.5, 0.5)`, `crop_pct=0.9`
- 하드코딩 시 정확도가 크게 하락함 (vit_tiny 기준 ~75% → ~44%)

### Global Non-uniform Pruning

```
핵심 아이디어:
  전체 블록 채널의 중요도를 한번에 비교 → 전역 하위 N개 제거
  → 중요한 블록(높은 중요도)은 채널을 더 많이 보존
  → 중복이 많은 블록(낮은 중요도)은 더 많이 제거

per-block 상한 (max_sparsity):
  cap 초과 채널의 score → inf로 마킹 → 전역 선택에서 자동 제외
  초과분은 여유 있는 다른 블록의 낮은 score 채널이 대신 채움
  → 총 제거 채널 수 = uniform과 동일하게 유지 (압축률 보장)
```

### Knowledge Distillation (KD)

Teacher는 student와 동일한 아키텍처의 pretrained 모델(frozen).  
KL divergence에 `T²` 보정을 곱해야 gradient scale이 CE loss와 동등해짐.

```python
kd_loss = F.kl_div(
    F.log_softmax(output / T, dim=1),
    F.softmax(teacher_logits / T, dim=1),
    reduction="batchmean",
) * (T * T)   # T²: /T를 하면 gradient가 1/T²로 작아지므로 복원
```

**Temperature가 하는 일:**
```
T=1: 고양이: 0.98  개: 0.01  여우: 0.005  → 클래스 관계 정보 거의 없음
T=4: 고양이: 0.61  개: 0.18  여우: 0.09   → 클래스 간 유사도 student에게 전달
```

### Progressive Pruning (Zhu & Gupta cubic schedule)

```python
progress = (epoch - warmup_epochs) / ramp_epochs
sparsity = target × (1 - (1 - progress)³)
```

cubic ease-out 선택 이유:
- 초반(progress 0~0.5): 빠르게 증가 → 모델이 낮은 sparsity에서 적응 시작
- 후반(progress 0.5~1): 완만하게 증가 → target 근처에서 세밀한 수렴

### Taylor Criterion + Gradient EMA

```python
# _channel_importance() 내부
taylor_now = (w * g).abs().sum(dim=1)                          # 현재 배치
grad_ema[id] = β × grad_ema[id] + (1-β) × taylor_now          # EMA 누적
return grad_ema[id]
```

EMA가 필요한 이유: single-batch gradient는 배치 구성(어떤 클래스가 들어왔는지)에 따라
크게 달라진다. ViT-Small처럼 배치 작고 모델 클 때 epoch 20에서 붕괴 현상이 관찰됐다.
β=0.9 EMA가 약 10 step의 gradient를 평균화하여 이를 해소한다.

### Soft Pruning — `tensor.data` vs `tensor`

```python
tensor.data.mul_(mask)   # ✓ autograd graph 우회, in-place 덮어씀
tensor *= mask           # ✗ autograd에 연산 추가 → optimizer.step() 이후 graph 손상
```

### Lazy init

`ViTPruner.__init__` 시점엔 model이 CPU에 있음.  
첫 `apply()` 호출 시 그룹 수집 → device mismatch 방지.

---

## 13. 주의사항 & 트러블슈팅

### ❶ Normalization 불일치

```python
# 잘못된 방법
mean=IMAGENET_DEFAULT_MEAN  # (0.485, 0.456, 0.406)

# 올바른 방법 (train.py, eval_baseline.py 모두 적용됨)
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
```

### ❷ DDP 환경에서 pruner.apply()

```python
# engine.py에서 처리됨
actual = model.module if hasattr(model, "module") else model
pruner.apply(actual)   # 반드시 .module 전달
```

### ❸ val/top1 급락 시

**Progressive 적용 전 (즉시 full sparsity)**: epoch 0 top-1 ~2%는 정상.  
epoch 5 이후에도 20% 미만이면 압축률 낮추기:
```yaml
target_compression: 0.30
```

**Progressive 적용 후**: epoch 0~4 구간에서 급락하면 warmup 연장:
```yaml
prune_warmup_epochs: 10   # 5 → 10
```

### ❹ Taylor EMA 불안정 (Small 모델 epoch 20 붕괴)

증상: val/top1이 epoch 20 근처에서 급락 후 회복  
원인: Single-batch Taylor gradient의 노이즈  
해결: `pruning_importance: taylor` + `grad_ema_beta: 0.9` (configs에 이미 설정됨)

beta를 올리면 더 오래된 gradient를 반영 (더 안정적이지만 반응 느림):
```yaml
# 여전히 불안정하면
grad_ema_beta: 0.95   # 기본 0.9 → 강화
```

### ❺ KD 비활성화

```yaml
kd_alpha: 0.0
```

### ❻ 체크포인트 키 확인

```python
ckpt = torch.load("checkpoint_best.pt", weights_only=False)
print(ckpt.keys())
# → ['model', 'model_ema', 'optimizer', 'lr_scheduler', 'scaler', 'pruner', 'epoch', 'best_acc1', 'args']
```

### ❼ 아키텍처 분석 재실행

```bash
python measure_memory.py
```

---

*작성: 2026-07 | 서버: `root@59bfae69b3a9` | GPU: Tiny→6,7 / Small→4,5*
