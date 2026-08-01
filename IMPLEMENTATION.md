# ViT Soft Pruning 구현 보고서

> 작성 기준: 2026-07 (Stage 2 Token Pruning 추가 반영)  
> 환경: timm 1.0.27 · torch 2.9.1 · Python 3.13.5  
> 서버: `root@59bfae69b3a9:/workspace/etri_iitp/JS/Server_Compression`  
> 레퍼런스:
>   - Stage 1 (channel pruning): EfficientViT Soft Pruning (동일 방법론을 timm ViT에 이식)
>   - Stage 2 (token pruning): EViT — Liang et al., ICLR 2022,
>     [youweiliang/evit](https://github.com/youweiliang/evit)
>     (알고리즘만 참고해 timm 1.0.x 호환 형태로 재구현 — repo 코드 직접 이식 아님, §14.1 참고)

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
14. [Stage 2 — EViT Token Pruning](#14-stage-2--evit-token-pruning)

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
│   ├── vit_small_prune50_progressive.yaml       ← Global + KD + Progressive + Taylor EMA
│   ├── vit_tiny_token_prune70.yaml              ← Stage 2: EViT Token Pruning, tiny 50% 기반 (§14)
│   ├── vit_small_token_prune70.yaml             ← Stage 2: EViT Token Pruning, small 50% 기반 (§14)
│   ├── vit_tiny_30_token_prune70.yaml           ← Stage 2: EViT Token Pruning, tiny 30% 기반 (§14)
│   └── vit_small_30_token_prune70.yaml          ← Stage 2: EViT Token Pruning, small 30% 기반 (§14)
├── pruning/
│   ├── __init__.py
│   ├── vit_pruning.py           ← ViTPruner: Soft Pruning 컨트롤러 (channel, Stage 1)
│   ├── vit_reducing.py          ← reduce_vit_model: Dense 변환 (Stage 1 → 완료 후)
│   ├── token_pruning.py         ← EvitTokenPruner: EViT Token Pruning (sequence, Stage 2)
│   └── vit_flops.py             ← FLOPs / Activation Footprint 분석적 추정 (§5, §11)
├── engine.py                    ← train_one_epoch / evaluate (Stage 1/2 공용)
├── train.py                     ← Stage 1 학습 진입점 (단일GPU / DDP, --config 지원)
├── reduce.py                    ← Reducing CLI (Stage 1 완료 후)
├── eval_baseline.py             ← Pruning 전 pretrained 모델 baseline 평가
├── eval_reduced.py              ← Stage 1 Reduced 모델 val 평가 → WandB test 기록
├── train_token_pruning.py       ← Stage 2 학습 진입점 (reduced.pt 입력)
├── eval_token_pruned.py         ← Stage 2 Token Pruned 모델 val 평가 → WandB test 기록
├── export_onnx.py               ← Reduced / Token Pruned 모델 → ONNX 변환 (자동 판별)
├── measure_memory.py            ← 아키텍처 분석 & 파라미터 프로파일링
├── data/
│   └── imagenet/                ← ImageNet (서버에만 존재, gitignore)
│       ├── train/               (1,281,167 images, 1000 classes)
│       └── val/                 (50,000 images, 1000 classes)
├── output/                      ← 체크포인트 저장 (gitignore)
└── IMPLEMENTATION.md
```

> **네이밍 컨벤션**: 각 모델×압축률 조합 중 `progressive_taylor`(Stage 1 권장 설정)로
> 학습한 것이 최종 채택 버전이다. 서버에서는 이 4개(tiny 30/50%, small 30/50%)를
> `vit_{tiny,small}_{30,50}_final`로 rename해서 관리한다
> (예: `output/vit_tiny_30_final/`). 이 문서의 실행 명령어 예시 중 일부는 구버전
> 이름(`vit_tiny_prune50_progressive_taylor` 등)을 쓰지만, 실제 서버 경로는
> `_final` 컨벤션을 따른다 — §14.7 참고.

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
`--gpu`는 상대 인덱스라 특정 GPU 하나만 쓰고 싶으면 `CUDA_VISIBLE_DEVICES` 없이
`--gpu <N>`만 지정하면 된다 (DDP 아님, 단일 프로세스 평가).

```bash
python eval_reduced.py \
  --reduced   ./output/vit_tiny_30_final/reduced.pt \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --gpu 4 \
  --wandb \
  --wandb-run-name vit_tiny_30_final_reduced_eval
```

기록 지표: `test/top1`, `test/top5`, `test/loss`, `test/n_params`, `test/compression_pct`,
그리고 `pruning/vit_flops.py` 기반 FLOPs/activation footprint 지표(§11) —
params뿐 아니라 실제 연산량/메모리 절감이 얼마나 되는지 baseline과 비교해서 같이 기록한다.

---

### `pruning/vit_flops.py` — FLOPs / Activation Footprint 분석적 추정

Params 압축률만으로는 실제 연산량(FLOPs)이나 추론 시 메모리(activation footprint)가
얼마나 줄었는지 알 수 없다 — channel pruning이 FFN만 건드리고 attention(QKV/proj,
§4의 G_QKV+G_PROJ ≈ 31%)은 그대로 두기 때문에, params 절감률과 FLOPs/activation
절감률이 서로 다르게 나온다. 이걸 `eval_reduced.py`에서 baseline(원본 구조) vs
reduced(block별 non-uniform mlp_dim)로 비교해서 WandB에 시각화한다.

```python
from pruning.vit_flops import analyze_vit_compute, compute_reduction

baseline_compute = analyze_vit_compute(embed_dim, num_heads, baseline_mlp_dims, n_patches)
reduced_compute  = analyze_vit_compute(embed_dim, num_heads, ckpt["mlp_dims"], n_patches)
reduction = compute_reduction(baseline_compute, reduced_compute)
```

**측정이 아니라 구조로부터 계산(analytical)이다** — hook 기반 프로파일러가 아닌 이유:
timm 1.0.x의 `Attention`이 `fused_attn=True`일 때 `F.scaled_dot_product_attention`
하나로 attention 전체를 처리해서, nn.Module 단위 forward hook으로는 Q@K^T /
softmax / attn@V FLOPs를 분해해서 잡을 수 없다. 이미 알고 있는 구조 정보
(embed_dim, num_heads, block별 mlp_dim, 토큰 개수)로 표준 공식을 이용해 직접
계산하는 쪽이 오히려 간단하고 정확하다.

- **FLOPs**: block별 `qkv + proj + attn(QK^T, attn@V) + mlp(fc1, fc2)` MACs 합 × 2
- **Activation footprint**: block 하나의 forward 동안 존재하는 주요 중간 텐서
  (qkv, attention matrix, mlp hidden 등) 크기 합의 **block별 최댓값(peak)** — 실제
  peak memory profiler가 아니라 상대 비교용 추정치임을 명시 (docstring 참고).
  attention 관련 항(`H*N*N`)은 channel pruning으로 안 줄어들므로, activation
  절감률이 params 절감률보다 작게 나오는 게 정상이다.

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

> **주의**: `export_onnx.py`의 `--reduced` 인자는 Stage 2(§14) 추가 시
> `--input`으로 이름이 바뀌었다 (reduced.pt / token_pruned_*.pt 공용 로더).

`--output`을 생략하면 체크포인트 메타데이터(모델명·압축률·keep_rate)로
**자동 네이밍**한다 — `reduced.onnx`/`token_pruned.onnx`처럼 모델마다 똑같은
이름이 나오면 폴더 밖으로 꺼냈을 때 구분이 안 되는 문제를 피하기 위함이다.
폴더 이름을 파싱하는 게 아니라 체크포인트 안의 값(`compression_rate`,
`n_params_before/after`, `token_pruning.base_keep_rate`)만 쓰므로 어디로
옮겨도 파일명만으로 구분된다.

```bash
# --output 생략 → 자동 네이밍
python export_onnx.py --input ./output/vit_tiny_30_final/reduced.pt --verify
#   → ./output/vit_tiny_30_final/vit_tiny_c30_reduced.onnx

python export_onnx.py \
  --input ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt --verify
#   → ./output/vit_tiny_30_final/token_prune70/vit_tiny_c30_token70.onnx
```

- `--output`: 직접 지정하면 자동 네이밍 대신 그 경로를 그대로 씀
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

### `eval_reduced.py` 전용 (test/* + FLOPs/Activation, §5)

| 키 | 내용 |
|----|------|
| `test/n_params`, `test/compression_pct` | 압축 후 파라미터 수 / 절감률 |
| `test/flops_baseline`, `test/flops_reduced`, `test/flops_reduction_pct` | FLOPs 원본 대비 절감률 |
| `test/activation_baseline_bytes`, `test/activation_reduced_bytes`, `test/activation_reduction_pct` | activation footprint(peak, 추정치) 절감률 |
| `compression_summary` | params/FLOPs/activation 절감률 한눈 비교 bar chart |
| `flops_per_block` | block별 FLOPs 절감률 bar chart — attention은 안 줄어서 block마다 편차가 큼 |
| `activation_per_block` | block별 activation footprint 절감률 bar chart |

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

### ❽ `checkpoint_best.pt`가 pruning 적용 전(warmup 구간) epoch에서 저장되는 문제 ★중요★

**증상**: `reduce.py`를 `checkpoint_best.pt`로 돌렸는데 `removed 0 params (0.00% removed)`가
나오거나, 압축률이 목표치보다 훨씬 낮게 나옴.

**원인**: `train.py`의 `is_best` 판정이 pruning 진행 상태와 무관하게 순수 `val_top1`
최고값만 본다. Progressive pruning은 `prune_warmup_epochs` 구간(예: epoch 0~4)에는
sparsity=0(=pruning 전, 원본 정확도)이라 이 구간의 val_top1이 이후 pruning이 걸린
epoch들보다 높게 나오기 쉽다. 그 경우 `checkpoint_best.pt`는 **pruning이 거의 안
걸린 상태의 체크포인트**가 되어버린다.

**진단**:
```python
import torch
ckpt = torch.load("checkpoint_best.pt", map_location="cpu", weights_only=False)
args = ckpt["args"]
warmup, ramp = args.get("prune_warmup_epochs", 0), args.get("prune_ramp_epochs", 0)
print(f"best epoch={ckpt['epoch']}  ramp_end={warmup+ramp}  "
      f"→ {'⚠ pruning 적용 전' if ckpt['epoch'] < warmup+ramp else 'OK'}")
```

**즉시 대안**: 이미 완료된 run이면 `checkpoint_last.pt`(마지막 epoch, sparsity=target
유지 구간)로 `reduce.py`를 돌린다. 최고 정확도 지점은 아니지만 최소한 목표
압축률은 반영된 결과를 얻는다.

**근본 수정 (권장, 아직 `train.py`에는 미반영)**: `is_best` 판정을 `epoch >=
prune_warmup_epochs + prune_ramp_epochs`(=target sparsity 도달 이후) 조건으로
gate해야 한다. `train_token_pruning.py`(Stage 2)에는 이미 이 수정이 적용돼 있다
(§14.4-1) — 동일한 패턴을 `train.py`에도 적용할 수 있다.

---

## 14. Stage 2 — EViT Token Pruning

> 구현 파일: `pruning/token_pruning.py`, `train_token_pruning.py`, `eval_token_pruned.py`,
> `export_onnx.py` (공용화)  
> config 예시: `configs/vit_tiny_token_prune70.yaml`, `configs/vit_small_token_prune70.yaml`

### 14.1 왜, 그리고 무엇을 이식했는가

Stage 1(channel pruning)은 FFN의 **폭(width)**을 줄인다. Stage 2는 **시퀀스 길이
(패치 토큰 개수)**를 줄인다 — 서로 직교하는 압축 축이라 같은 backbone에 순차
적용 가능하다.

레퍼런스는 [youweiliang/evit](https://github.com/youweiliang/evit) (EViT,
Liang et al., ICLR 2022)이지만, **저장소를 그대로 이식하지 않았다.** 원 저장소는
`torch==1.9.0`, `timm==0.4.12` 기준으로 timm의 `Attention`/`Block`을 통째로
포크해서 고쳐놓은 코드라, 지금 쓰는 `timm==1.0.27`과 구조가 크게 다르다
(LayerScale, `fused_attn`/`F.scaled_dot_product_attention` 도입 등). 그래서
**알고리즘(CLS-attention 기반 top-k 선택 + fused token)만 가져오고, 구현은
현재 timm 버전에 맞게 새로 짰다.**

가장 중요한 차이점 하나: timm 1.0.x의 `Attention.forward`는 기본적으로
`fused_attn=True`라 `F.scaled_dot_product_attention`을 쓰며, 이 경로는 attention
행렬을 아예 만들지 않는다. 즉 EViT가 필요로 하는 "CLS→patch attention score"를
얻을 수 없다. 여기서 두 가지 선택지가 있었다:

1. `fused_attn`을 강제로 끄고 `Attention.forward` 전체를 eager 모드로 재구현
2. `Attention.forward`는 그대로 두고, CLS attention score만 별도로 계산

**2번을 선택했다.** 1번은 원본 attention 출력(x)을 손으로 재현해야 하는데,
사소한 구현 실수(reshape/permute 순서, scale 위치 등)가 조용히 정확도를 깎아먹을
위험이 있다. 2번은 `self.attn.qkv` Linear를 한 번 더 통과시켜 `q, k`만 뽑고
CLS row(`q[:, :, 0:1, :] @ k.T`)만 계산하는 것이라, 원본 attention 경로를 전혀
건드리지 않는다. 추가 비용은 O(N) 크기의 작은 matmul 하나뿐 — 전체 attention의
O(N²) 대비 무시할 수준이다.

### 14.2 파이프라인 순서 — 왜 reduced 모델을 대상으로 하는가

```
[Stage 1] Soft channel pruning + KD (train.py)
    → checkpoint_best.pt

[Reduce] reduce.py
    → reduced.pt   (FFN이 물리적으로 축소된 순수 Dense 모델)

[Stage 2] EViT Token Pruning fine-tuning (train_token_pruning.py)
    → reduced.pt를 시작점으로 로드
    → checkpoint_last/best.pt (재개용) + token_pruned_best.pt (배포용)
```

동시 진행(같은 학습 루프에서 channel pruning + token pruning)이 아니라 **순차
2단계**로 설계했다. 이유:

- Channel pruning만으로도 ViT-Small에서 epoch 20 근처 collapse가 관찰됐고
  (§13-❹), Taylor EMA로 겨우 안정화했다. 여기에 토큰 단위 동적 변화까지 같은
  루프에서 동시에 켜면 val/top1 하락의 원인이 sparsity 스케줄 때문인지 keep_rate
  스케줄 때문인지 구분할 수 없다.
- `reduce.py` 완료 후 결과물은 이미 검증된 안정적인 Dense 모델이다. EViT 원 논문도
  잘 학습된 dense backbone 위에 token pruning을 얹어 짧게 fine-tuning하는 방식을
  쓴다 — 여기서는 그 backbone 자리에 "channel-pruned dense 모델"을 놓은 것뿐이다.
- 두 메커니즘은 서로 다른 텐서 축(FFN width vs. sequence length)에서 작동하고,
  channel pruning은 QKV/proj를 건드리지 않으므로(§4의 G_QKV, G_PROJ는 pruning
  대상이 아님) embed_dim이 그대로 유지돼 순서 종속성 문제가 없다.

### 14.3 알고리즘 — CLS Attention 기반 Token 선택 + Fusion

선택된 일부 block(기본: `depth // 4` 등분 지점, 12-block 모델이면 block 3, 6, 9)에서:

```
① CLS attention score 계산 (attn.forward는 건드리지 않고 별도 계산)
   x_norm = norm1(x)
   attn_out = attn(x_norm)                    ← 원본 그대로, fused_attn 유지
   cls_attn = CLS→patch attention score        ← qkv Linear 재사용해 별도 계산
                                                  (B, N-1), head 평균, softmax

② 잔차 연결 (원본과 동일)
   x = x + drop_path1(ls1(attn_out))

③ Token 선택 (keep_rate < 1인 block만)
   n_keep = ceil((N-1) × keep_rate)            ← 모든 입력에 대해 동일한 상수!
   top-k index = topk(cls_attn, n_keep)         ← "어떤" 토큰인지는 입력마다 다름
   x_kept = gather(x[:, 1:], top-k index)

④ Fusion (fuse_token=True 시)
   버려지는 (N-1-n_keep)개 토큰을 각자의 cls_attn 가중치로 가중합 →
   fused_token 1개로 합쳐서 유지 (완전 폐기 대비 정보 손실 최소화)
   x = cat([CLS, x_kept, fused_token])
   → 다음 block으로 넘어가는 시퀀스 길이 = 1(CLS) + n_keep + 1(fused) = n_keep + 2

⑤ MLP (원본과 동일, 단 짧아진 시퀀스에 대해)
   x = x + drop_path2(ls2(mlp(norm2(x))))
```

`keep_rate`가 고정 비율이고 패치 개수 N도 고정(입력 해상도 224 고정)이므로
**n_keep은 모든 입력에 대해 동일한 컴파일타임 상수**다 — 텐서 shape은 완전히
정적이고, `topk`가 고르는 인덱스 "값"만 입력마다 다르다. DynamicViT류(threshold
기반이라 샘플마다 남는 토큰 "개수" 자체가 다름)보다 ONNX/NPU 컴파일에 훨씬
우호적인 이유가 이것이다 (§14.6).

### 14.4 Progressive Keep Rate 스케줄

Stage 1의 progressive sparsity(§2, Zhu & Gupta cubic ease-out)와 동일한 형태를
keep_rate에 적용했다:

```python
# pruning/token_pruning.py: EvitTokenPruner._scheduled_keep_rate()
if epoch < warmup_epochs:
    keep_rate = 1.0                              # pruning 없음, 정상 학습
elif epoch < warmup_epochs + ramp_epochs:
    progress = (epoch - warmup_epochs) / ramp_epochs
    drop = 1.0 - base_keep_rate
    keep_rate = 1.0 - drop × (1 - (1-progress)³)  # cubic ease-out
else:
    keep_rate = base_keep_rate                    # 목표치 유지
```

`ViTPruner`(channel pruning)와의 핵심 차이: `EvitTokenPruner`는 **weight를
마스킹하지 않는다.** 순수하게 forward 시점의 시퀀스 길이만 바꾸는 것이라
`optimizer.step()` 이후에 호출할 `.apply()`가 없다 — 매 epoch 시작 시
`set_epoch()`만 호출하면 된다 (`pruner.apply()` 같은 매 step 마스크 재적용이
필요 없음).

#### 14.4-1 `is_best` 판정 — §13-❽와 동일한 함정을 Stage 2에서도 발견

`vit_tiny_30_final`로 첫 Stage 2 run을 돌려보니, `val/top1_best`가 epoch 3
(=`keep_rate_warmup_epochs` 직후, keep_rate가 아직 1.0에 가까운 시점)에서
73.6%로 찍힌 뒤 나머지 26 epoch 내내 한 번도 안 움직였다 — §13-❽에서 발견한
channel pruning의 `checkpoint_best.pt` 함정이 token pruning에도 그대로
재현된 것이다. `train_token_pruning.py`도 원래는 `train.py`와 동일하게
"keep_rate 상태 무관하게 그냥 val_top1 최고값"만 봤기 때문이다.

**수정** ([train_token_pruning.py](../train_token_pruning.py) 학습 루프):
```python
ramp_end = token_pruner.warmup_epochs + token_pruner.ramp_epochs
fully_pruned = epoch >= ramp_end
is_best = fully_pruned and acc1 > best_acc1   # ramp 끝난 이후 epoch만 후보
```
`keep_rate_warmup_epochs`/`keep_rate_ramp_epochs`가 모두 0(non-progressive)이면
`ramp_end=0`이라 모든 epoch이 후보가 되어 기존과 동일하게 동작한다.

**엣지 케이스**: `epochs <= warmup_epochs + ramp_epochs`로 설정하면 `fully_pruned`인
epoch이 학습 루프 안에 하나도 안 생겨서 `is_best`가 끝까지 한 번도 안 켜지고,
`checkpoint_best.pt`/`token_pruned_best.pt` 자체가 생성되지 않는다
(`checkpoint_last.pt`는 항상 저장됨). 지금 쓰는 4개 config는 `epochs=30`,
`ramp_end=15`라 15 epoch 여유가 있어 안전하다.

**이미 이 문제로 오염된 run 복구**: `token_pruned_best.pt`가 이미 (수정 전 코드로)
epoch 3 상태로 저장돼버린 경우, `checkpoint_last.pt`(마지막 epoch, keep_rate=target
도달)의 EMA weight로 배포용 아티팩트를 직접 재구성해야 한다:
```python
import torch
ckpt         = torch.load("checkpoint_last.pt", map_location="cpu", weights_only=False)
reduced_ckpt = torch.load("../reduced.pt", map_location="cpu", weights_only=False)  # Stage 1 산출물
tp = ckpt["token_pruner"]
torch.save({
    "state_dict":      ckpt["model_ema"],
    "model_name":      reduced_ckpt["model_name"],
    "mlp_dims":        reduced_ckpt["mlp_dims"],
    "token_pruning": {
        "prune_layers":   tp["prune_layers"],
        "base_keep_rate": tp["keep_rate"],       # 실제 도달한 값(=target)
        "fuse_token":     tp["fuse_token"],
    },
    "n_params_before": reduced_ckpt.get("n_params_before", 0),
    "n_params_after":  reduced_ckpt.get("n_params_after", 0),
}, "token_pruned_last.pt")
```
이렇게 만든 `token_pruned_last.pt`는 `eval_token_pruned.py`/`export_onnx.py`가
`token_pruned_best.pt`와 동일하게 로드할 수 있는 포맷이다.

### 14.5 Knowledge Distillation — Teacher 선택

Stage 2는 학습 가능한 파라미터를 추가하지 않는다(DynamicViT의 learned predictor와
달리, EViT는 기존 attention을 그대로 재활용하는 training-free 선택 방식). 대신
KD로 정확도 회복을 돕는다. `--kd-teacher-mode`로 두 가지를 지원:

| 모드 | Teacher | 특징 |
|------|---------|------|
| `reduced` (기본값) | token pruning 적용 **전**의 동일 reduced 모델 (전체 토큰 사용) | Self-distillation. Stage 1 정확도를 기준점으로 삼아 token pruning으로 인한 손실만 회복하도록 유도 |
| `original` | 원본 pretrained dense 모델 | 더 강한 teacher지만 이미 Stage 1에서 한 번 압축된 student와 capacity gap이 커서 신호가 덜 직접적 |

기본값(`reduced`)을 권장한다 — "이 reduce된 모델이 토큰을 줄이기 전엔 냈던
성능"을 직접 타깃으로 삼는 게 가장 직접적인 신호이기 때문이다.

### 14.6 NPU 배포 관점 — TopK + Gather

전체 파이프라인에서 가장 리스크가 큰 지점이다. Channel pruning 결과물(reduced
모델)은 그냥 더 작은 표준 ViT라 ONNX/NPU 변환에 특별한 리스크가 없지만, token
pruning은 그래프에 `TopK`와 **런타임에 계산된 인덱스로 하는 Gather**를 추가한다.
이건 conv/matmul/elementwise 위주로 설계된 edge NPU 컴파일러 상당수가 지원하지
않거나 CPU fallback으로 빠지는 연산 패턴이다.

다만 §14.3에서 설명했듯 **텐서 shape 자체는 완전히 정적**이다(keep_rate가 고정
비율이라 n_keep이 상수). 이 조건 덕분에 Mobilint NPU 컴파일러 통과 여부를 사전에
확인했다 — TopK/Gather op 자체가 컴파일 가능함을 확인한 뒤에 이 Stage 2 구현에
착수했다.

**향후 다른 NPU 타겟으로 이식할 때는 반드시 이 순서를 지킬 것**: 전체
fine-tuning을 다 돌리기 전에, TopK+Gather만 들어간 최소 toy ONNX 그래프를 먼저
그 컴파일러에 넣어보고 통과 여부(및 실제 온칩 실행인지, CPU fallback인지)를
확인한다. 통과하지 않으면:

- **속도 이득 포기, 정확도만 취하는 대안**: 토큰을 gather로 제거하지 않고
  0-마스킹만 적용. Shape이 안 바뀌므로 NPU엔 안전하지만 N 전체를 계속 연산하므로
  속도 이득이 없다.
- **입력 비의존적 static pruning 대안**: 매 입력마다 고정된 위치의 토큰을 제거.
  TopK/동적 Gather가 없어져 NPU엔 안전하지만, EViT의 핵심 강점(이미지마다 다른
  중요 패치를 적응적으로 고름)을 잃는다.

### 14.6-1 WandB 프로젝트 분리

Stage 1(channel pruning, `train.py`/`eval_reduced.py`)과 Stage 2(token pruning)는
서로 다른 WandB 프로젝트를 쓴다 — 압축 축이 달라서 같은 프로젝트에 섞으면 비교가
헷갈리기 때문:

| 스크립트 | 기본 `--wandb-project` |
|----------|------------------------|
| `train.py`, `eval_reduced.py` | `vit-pruning` (기존) |
| `train_token_pruning.py`, `eval_token_pruned.py` | `vit-token-pruning` (신규, 기본값) |

`configs/*_token_prune70.yaml` 4개 모두 `wandb_project: vit-token-pruning`이 이미
박혀 있어서 별도 인자 없이 `--config`만 써도 자동으로 새 프로젝트로 간다.

### 14.7 실행 명령어

`_final` 네이밍 컨벤션(프로젝트 루트 노트 참고) 기준. 동일 서버에서 여러 job을
동시에 돌릴 때는 `torchrun`에 `--master_port`를 job마다 다르게 지정해야
포트 충돌이 안 난다 (기본 29500 하나만 쓰면 두 번째 torchrun이 바인딩 실패).

```bash
# Stage 1 → Reduce (checkpoint_best.pt가 §13-❽ 문제로 오염됐으면 checkpoint_last.pt 사용)
python reduce.py --model vit_tiny_patch16_224 \
  --checkpoint ./output/vit_tiny_30_final/checkpoint_last.pt \
  --output     ./output/vit_tiny_30_final/reduced.pt

# Stage 2 — Token Pruning fine-tuning (GPU 여러 개 동시 돌릴 땐 --master_port 다르게)
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 train_token_pruning.py \
  --config configs/vit_tiny_30_token_prune70.yaml

# 평가 (§14.4-1 문제로 token_pruned_best.pt가 오염됐으면 token_pruned_last.pt 사용)
python eval_token_pruned.py \
  --token-pruned ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --gpu 4 --wandb

# ONNX 변환 (reduced.pt / token_pruned_*.pt 공용, "token_pruning" 키로 자동 판별)
# --output 생략 시 vit_tiny_c30_reduced.onnx / vit_tiny_c30_token70.onnx로 자동 네이밍 (§9)
python export_onnx.py --input ./output/vit_tiny_30_final/reduced.pt --verify

python export_onnx.py \
  --input ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt --verify
```

### 14.8 `token_pruned_best.pt` / `token_pruned_last.pt` 로드 방법

두 파일 다 포맷이 동일하다 (`best`는 정상적으로 post-ramp에서 갱신된 경우,
`last`는 §14.4-1 문제로 수동 재구성한 경우) — 로드 코드는 똑같다.

```python
import torch, timm
from pruning.vit_reducing import apply_reduced_config
from pruning.token_pruning import apply_token_pruning

ckpt   = torch.load("token_pruned_best.pt", map_location="cpu")  # 또는 token_pruned_last.pt
model  = timm.create_model(ckpt["model_name"], pretrained=False)
apply_reduced_config(model, ckpt["mlp_dims"])                 # Stage 1 구조 축소

tp_cfg = ckpt["token_pruning"]
apply_token_pruning(
    model,
    prune_layers=tp_cfg["prune_layers"],
    base_keep_rate=tp_cfg["base_keep_rate"],
    fuse_token=tp_cfg["fuse_token"],
)                                                               # Stage 2 forward 패치
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

### 14.9 주의사항

- **표준 단일 CLS 토큰 ViT만 지원.** `dist_token`이 있는 distilled 모델이나
  `no_embed_class` 변형은 `pruning/token_pruning.py`의 `_validate_model()`에서
  즉시 예외를 던진다 (조용히 잘못된 결과를 내지 않도록).
- **timm 버전이 바뀌면 `Attention` 내부 속성명(`qkv`, `num_heads`, `q_norm`,
  `k_norm`)이 달라질 수 있다.** `_validate_model()`이 `hasattr` 체크로 조기
  실패하지만, 정확한 CLS attention score 계산을 보장하려면 timm 업그레이드 시
  `_cls_attention_scores()`를 다시 확인해야 한다.
- **실제로 겪은 timm 버전 호환성 문제**: 서버에 설치된 timm의 `Attention.forward()`가
  `is_causal` 키워드 인자를 안 받아서 `TypeError: Attention.forward() got an
  unexpected keyword argument 'is_causal'`가 났다 (GitHub의 최신 `vision_transformer.py`
  소스와 실제 설치 버전이 미묘하게 다름). 이 repo의 ViT 분류 파이프라인은
  `attn_mask`/`is_causal`을 애초에 안 쓰므로(항상 `None`/`False`),
  `_evit_block_forward()`에서 이 kwarg들을 `self.attn(...)`에 아예 전달하지
  않도록 고쳐서 해결했다 — 대신 `attn_mask is not None or is_causal`이면
  `NotImplementedError`로 조기 실패한다. **timm을 업그레이드하거나 다른 서버에
  이식할 때 이 부분이 다시 깨질 수 있으니, 처음 실행할 때는 아래 스모크 테스트로
  먼저 검증할 것** (실제 학습 3 epoch을 기다리지 않고 forward+backward만 즉시 확인):
  ```python
  import torch, timm
  from pruning.token_pruning import apply_token_pruning

  model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
  apply_token_pruning(model, prune_layers=[3, 6, 9], base_keep_rate=0.7, fuse_token=True)

  model.eval()
  with torch.no_grad():
      out = model(torch.randn(2, 3, 224, 224))
  print("forward OK:", tuple(out.shape))

  model.train()
  model(torch.randn(2, 3, 224, 224)).sum().backward()
  print("backward OK:", model.blocks[3].attn.qkv.weight.grad is not None)
  ```
- **`is_best`가 keep_rate ramp 종료 이전 epoch을 잘못 채택하는 문제 — 겪었고
  수정함.** §14.4-1 참고. 이미 오염된 run은 `checkpoint_last.pt`에서
  `token_pruned_last.pt`를 수동으로 재구성해야 한다.
- **`export_onnx.py`의 CLI 인자가 `--reduced` → `--input`으로 바뀌었다** (Stage 1
  전용이던 로더를 Stage 1/2 공용으로 바꾸면서 이름을 일반화함). 기존 스크립트/
  문서에 `--reduced`로 남아있는 부분은 `--input`으로 교체해야 한다.
- **Stage 2에는 channel pruning(`ViTPruner`)이 관여하지 않는다.** `mlp_dims`는
  Stage 1에서 결정된 그대로 유지되고, Stage 2는 순수하게 시퀀스 길이만 바꾼다.

---

*작성: 2026-07 | 서버: `root@59bfae69b3a9` | GPU: Tiny→6,7 / Small→4,5*
*업데이트: 2026-07 | Stage 2 EViT Token Pruning 추가 (§14)*
*업데이트: 2026-08 | FLOPs/Activation footprint 분석(§5, §11), `checkpoint_best.pt`/*
*`token_pruned_best.pt`가 pruning 적용 전 epoch에서 저장되는 문제 발견 및 수정(§13-❽, §14.4-1),*
*timm `is_causal` 호환성 이슈 수정(§14.9), WandB 프로젝트 분리(§14.6-1), tiny/small 30% config 추가*
