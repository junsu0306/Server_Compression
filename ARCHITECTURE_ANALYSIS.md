# ViT / DeiT 아키텍처 분석 보고서

> timm 1.0.27, torch 2.9.1 기준  
> `measure_memory.py` 실행 결과를 바탕으로 작성 (2026-06)

---

## 목차

1. [분석 대상 모델](#1-분석-대상-모델)
2. [공통 아키텍처 개요](#2-공통-아키텍처-개요)
3. [모델별 하이퍼파라미터](#3-모델별-하이퍼파라미터)
4. [모듈 구조 상세 (vit_base 기준)](#4-모듈-구조-상세-vit_base-기준)
5. [파라미터 그룹 분류](#5-파라미터-그룹-분류)
6. [Sparsity / Compression 분석](#6-sparsity--compression-분석)
7. [ViT vs DeiT 차이점](#7-vit-vs-deit-차이점)
8. [Pruning 관점 핵심 요약](#8-pruning-관점-핵심-요약)

---

## 1. 분석 대상 모델

| 모델 | timm 이름 | 전체 파라미터 | 크기(FP32) | 대상 |
|------|-----------|-------------|-----------|:---:|
| ViT-Tiny | `vit_tiny_patch16_224` | 5,717,416 | 22.9 MB | **✓** |
| ViT-Small | `vit_small_patch16_224` | 22,050,664 | 88.2 MB | **✓** |
| ~~ViT-Base~~ | ~~`vit_base_patch16_224`~~ | ~~86,567,656~~ | ~~346.3 MB~~ | — |

> **현재 구현 대상: ViT-Tiny, ViT-Small.**  
> ViT-Base 및 DeiT 계열은 현재 제외. 동일 코드로 실행 가능하나 `--model` 인자만 변경하면 됨.

---

## 2. 공통 아키텍처 개요

ViT(Vision Transformer)의 전체 데이터 흐름:

```
입력 이미지 (B, 3, 224, 224)
        │
        ▼
[PatchEmbed]
  Conv2d(3 → embed_dim, kernel=16×16, stride=16)
  → (B, 196, embed_dim)          ← 14×14 = 196 패치
        │
        ▼
[Token Concat]
  cls_token (1, 1, embed_dim) 앞에 붙임
  → (B, 197, embed_dim)
        │
        ▼
[Positional Embedding]
  pos_embed (1, 197, embed_dim) 덧셈 (학습 가능)
  → (B, 197, embed_dim)
        │
        ▼ ×num_layers
[Transformer Block]  ← 이 블록이 핵심 (아래 상세 설명)
        │
        ▼
[Final LayerNorm]  norm: LayerNorm(embed_dim)
        │
        ▼
[Classifier Head]  head: Linear(embed_dim → num_classes)
  cls_token 위치([0]) 만 사용
        │
        ▼
출력 (B, 1000)
```

### Transformer Block 내부 구조

각 블록은 Multi-Head Self-Attention + FFN의 Pre-Norm 구조:

```
입력 x (B, 197, embed_dim)
│
├─ [Pre-Norm 1]  norm1: LayerNorm(embed_dim)
│         │
│   [MHSA]  Multi-Head Self-Attention
│   ┌─────────────────────────────────────────────┐
│   │  qkv: Linear(embed_dim → 3×embed_dim)       │  ← Q, K, V 동시 계산 (fused)
│   │  reshape → (B, num_heads, 197, head_dim)    │
│   │  Attention: softmax(QKᵀ / √head_dim) × V   │
│   │  proj: Linear(embed_dim → embed_dim)        │  ← 출력 projection
│   └─────────────────────────────────────────────┘
│         │
│   + residual (skip connection)
│
├─ [Pre-Norm 2]  norm2: LayerNorm(embed_dim)
│         │
│   [FFN]  Feed-Forward Network
│   ┌─────────────────────────────────────────────┐
│   │  fc1:  Linear(embed_dim → mlp_dim)          │  ★ PRUNABLE (G_FFN)
│   │  act:  GELU()                               │
│   │  fc2:  Linear(mlp_dim → embed_dim)          │  ★ PRUNABLE (G_FFN, coupled)
│   └─────────────────────────────────────────────┘
│         │
└─ + residual (skip connection)
        │
출력 x' (B, 197, embed_dim)
```

**핵심 관계**: fc1의 출력 채널(`mlp_dim`)과 fc2의 입력 채널(`mlp_dim`)은 같은 차원이다.  
→ fc1 출력 행(row) 제거 시 반드시 fc2 입력 열(col)도 같은 인덱스로 제거해야 한다.

---

## 3. 모델별 하이퍼파라미터

| 항목 | ViT/DeiT-Tiny | ViT/DeiT-Small | ViT/DeiT-Base |
|------|:---:|:---:|:---:|
| `embed_dim` | 192 | 384 | 768 |
| `num_heads` | 3 | 6 | 12 |
| `head_dim` | 64 | 64 | 64 |
| `num_layers` | 12 | 12 | 12 |
| `mlp_dim` (=4×embed_dim) | 768 | 1,536 | 3,072 |
| `mlp_ratio` | 4.0× | 4.0× | 4.0× |
| `patch_size` | 16×16 | 16×16 | 16×16 |
| `num_patches` | 196 (14×14) | 196 | 196 |
| `seq_len` (+cls) | 197 | 197 | 197 |
| `mlp_act` | GELU | GELU | GELU |
| `dist_token` | 없음 | 없음 | 없음 |
| **블록 당 파라미터** | **444,864** | **1,774,464** | **7,087,872** |
| — attn (qkv+proj) | 148,224 | 591,360 | 2,362,368 |
| — ffn (fc1+fc2) | 295,872 | 1,181,568 | 4,722,432 |
| **전체 파라미터** | **5,717,416** | **22,050,664** | **86,567,656** |

> `head_dim = embed_dim / num_heads = 64` — 세 모델 모두 동일.  
> 스케일 차이는 `embed_dim`과 그에 연동된 `mlp_dim`에서만 발생한다.

---

## 4. 모듈 구조 상세 (vit_base 기준)

`vit_base_patch16_224` 전체 named_modules (weight 보유 레이어만):

```
이름                                      클래스        weight shape         그룹
─────────────────────────────────────────────────────────────────────────────
patch_embed.proj                          Conv2d       (768, 3, 16, 16)     G_EMBED
─────────────────────────────────────────────────────────────────────────────
blocks.0.norm1                            LayerNorm    (768,)               G_NORM
blocks.0.attn.qkv                         Linear       (2304, 768)          G_QKV     ← 2304 = 3×768
blocks.0.attn.proj                        Linear       (768, 768)           G_PROJ
blocks.0.norm2                            LayerNorm    (768,)               G_NORM
blocks.0.mlp.fc1                          Linear       (3072, 768)          G_FFN ★
blocks.0.mlp.fc2                          Linear       (768, 3072)          G_FFN ★
─────────────────────────────────────────────────────────────────────────────
blocks.1 ~ blocks.11                      (위와 동일 구조 반복, ×12)
─────────────────────────────────────────────────────────────────────────────
norm                                      LayerNorm    (768,)               G_NORM
head                                      Linear       (1000, 768)          G_HEAD
─────────────────────────────────────────────────────────────────────────────
cls_token  (nn.Parameter, 모델 직접 등록)   —           (1, 1, 768)          G_EMBED
pos_embed  (nn.Parameter, 모델 직접 등록)   —           (1, 197, 768)        G_EMBED
```

### 모델 크기별 weight shape 비교

| 레이어 | Tiny | Small | Base |
|--------|------|-------|------|
| `patch_embed.proj` | (192, 3, 16, 16) | (384, 3, 16, 16) | (768, 3, 16, 16) |
| `attn.qkv` | (576, 192) | (1152, 384) | (2304, 768) |
| `attn.proj` | (192, 192) | (384, 384) | (768, 768) |
| `mlp.fc1` ★ | **(768, 192)** | **(1536, 384)** | **(3072, 768)** |
| `mlp.fc2` ★ | **(192, 768)** | **(384, 1536)** | **(768, 3072)** |
| `head` | (1000, 192) | (1000, 384) | (1000, 768) |
| `cls_token` | (1, 1, 192) | (1, 1, 384) | (1, 1, 768) |
| `pos_embed` | (1, 197, 192) | (1, 197, 384) | (1, 197, 768) |

### FFN 채널 수 (Pruning 대상)

| 모델 | `mlp_dim` (fc1 out / fc2 in) | 블록 수 | 전체 Prunable 채널 수 |
|------|:---:|:---:|:---:|
| Tiny | 768 | 12 | 9,216 |
| Small | 1,536 | 12 | 18,432 |
| Base | 3,072 | 12 | 36,864 |

---

## 5. 파라미터 그룹 분류

### 그룹 정의

| 그룹 | 포함 레이어 | Prunable |
|------|------------|:---:|
| **G_FFN** | `blocks.i.mlp.fc1`, `blocks.i.mlp.fc2` | **예 (1차 구현 대상)** |
| G_QKV | `blocks.i.attn.qkv` (Q+K+V fused) | 선택적 (2차) |
| G_PROJ | `blocks.i.attn.proj` | 아니오 (embed_dim 고정) |
| G_NORM | 모든 LayerNorm | 아니오 |
| G_HEAD | `head` (Linear) | 아니오 (num_classes 고정) |
| G_EMBED | `patch_embed`, `cls_token`, `pos_embed` | 아니오 |

> **G_QKV Pruning 복잡도**: `attn.qkv` weight는 Q/K/V가 fused되어 있다.  
> Q와 K의 head_dim만 pruning 가능하며 반드시 동일 인덱스를 사용해야 한다.  
> V는 `attn.proj`의 입력이므로 embed_dim에 고정 → V는 pruning 불가.

### ViT-Tiny 파라미터 분포

```
그룹         파라미터 수       MB      비중
─────────────────────────────────────────
G_FFN        3,550,464     14.202   62.10%  ← Prunable
G_QKV        1,334,016      5.336   23.33%
G_PROJ         444,672      1.779    7.78%
G_NORM           9,600      0.038    0.17%
G_HEAD         193,000      0.772    3.38%
G_EMBED        185,664      0.743    3.25%  ← patch_embed + cls_token + pos_embed
─────────────────────────────────────────
TOTAL        5,717,416     22.870  100.00%
```

### ViT-Small 파라미터 분포

```
그룹         파라미터 수       MB      비중
─────────────────────────────────────────
G_FFN       14,178,816     56.715   64.30%  ← Prunable
G_QKV        5,322,240     21.289   24.14%
G_PROJ       1,774,080      7.096    8.05%
G_NORM          19,200      0.077    0.09%
G_HEAD         385,000      1.540    1.75%
G_EMBED        371,328      1.485    1.68%
─────────────────────────────────────────
TOTAL       22,050,664     88.203  100.00%
```

### ViT-Base 파라미터 분포

```
그룹         파라미터 수        MB      비중
──────────────────────────────────────────
G_FFN       56,669,184     226.677   65.46%  ← Prunable
G_QKV       21,261,312      85.045   24.56%
G_PROJ       7,087,104      28.348    8.19%
G_NORM          38,400       0.154    0.04%
G_HEAD         769,000       3.076    0.89%
G_EMBED        742,656       2.971    0.86%
──────────────────────────────────────────
TOTAL       86,567,656     346.271  100.00%
```

> **G_EMBED 내역 (vit_base 예시)**  
> - `patch_embed.proj` weight+bias: 590,592 + (768) ≈ 591,360  
> - `cls_token` (nn.Parameter): 768  
> - `pos_embed` (nn.Parameter): 151,296 (= 197 × 768)  
>
> `cls_token`, `pos_embed`는 `model.named_modules()` 에서는 컨테이너로 잡히지 않고  
> root VisionTransformer 모듈의 직접 Parameter로 등록되어 있다.

---

## 6. Sparsity / Compression 분석

G_FFN만 pruning 시 이진탐색으로 계산한 per-group sparsity.  
**Secondary effect 포함**: fc1을 s% 제거하면 fc2 입력도 s% 감소하므로  
단순 선형 계산보다 실제 압축률이 더 크다.

### ViT/DeiT-Tiny (전체 5.72M)

| target_compression | G_FFN sparsity | 제거 파라미터 수 | 실제 압축률 |
|:---:|:---:|---:|:---:|
| 10% | 0.1608 | 568,260 | 9.94% |
| 20% | 0.3223 | 1,145,760 | 20.04% |
| 30% | 0.4837 | 1,714,020 | 29.98% |
| 50% | 0.8053 | 2,855,160 | 49.94% |

### ViT/DeiT-Small (전체 22.05M)

| target_compression | G_FFN sparsity | 제거 파라미터 수 | 실제 압축률 |
|:---:|:---:|---:|:---:|
| 10% | 0.1553 | 2,196,264 | 9.96% |
| 20% | 0.3109 | 4,401,756 | 19.96% |
| 30% | 0.4665 | 6,616,476 | 30.01% |
| 50% | 0.7777 | 11,027,460 | 50.01% |

### ViT/DeiT-Base (전체 86.57M)

| target_compression | G_FFN sparsity | 제거 파라미터 수 | 실제 압축률 |
|:---:|:---:|---:|:---:|
| 10% | 0.1528 | 8,668,680 | 10.01% |
| 20% | 0.3055 | 17,318,916 | 20.01% |
| 30% | 0.4585 | 25,969,152 | 30.00% |
| 50% | 0.7638 | 43,269,624 | 49.98% |

### sparsity 해석

```
G_FFN sparsity 0.30 이란?
  → 각 블록의 mlp_dim 채널 중 30%를 제거 (L2 norm 하위 30%)

  vit_base 예시:
  mlp_dim = 3072 채널 중 round(3072 × 0.4585) = 1409 채널 제거
  살아남는 채널: 3072 - 1409 = 1663 채널
  fc1: (3072, 768) → (1663, 768)
  fc2: (768, 3072) → (768, 1663)
  제거 파라미터: 1409×768 + 1409 + 768×1409 = 1,081,536 + 1,409 + 1,081,312 = 2,164,257
  12 블록: 2,164,257 × 12 = 25,971,084 ≈ 25,969,152 (이진탐색 결과와 일치)
```

---

## 7. ViT vs DeiT 차이점

### timm 1.0.27 기준

| 항목 | ViT | DeiT (`*_patch16_224`) | DeiT-distilled (`*_distilled_*`) |
|------|-----|------------------------|----------------------------------|
| 아키텍처 구조 | 동일 | **ViT와 완전 동일** | cls_token + dist_token (2개) |
| `dist_token` | 없음 | 없음 | 있음 (1, 1, embed_dim) |
| `head_dist` | 없음 | 없음 | 있음 (Linear) |
| 출력 | (B, 1000) | (B, 1000) | (B, 1000) — eval 시 평균 |
| 전체 파라미터 | 동일 | 동일 | 미세하게 다름 |
| 학습 방식 | 자기지도/지도 | 교사 모델 증류 | 교사 모델 + dist_token 증류 |

**결론**: `deit_*_patch16_224` (non-distilled)는 ViT와 **아키텍처가 완전히 동일**하다.  
Pruning 코드를 공유할 수 있으며, 별도 분기 처리가 불필요하다.

`deit_*_distilled_patch16_224`는 `dist_token`과 `head_dist`가 추가되므로  
forward 검증 시 출력 처리가 달라진다 (현재 구현 범위 외).

---

## 8. Pruning 관점 핵심 요약

### 구현 결정 사항

| 항목 | 결정 | 근거 |
|------|------|------|
| 1차 Pruning 대상 | G_FFN 전용 | 전체의 62~65%, 구현 단순, 효과 최대 |
| G_QKV Pruning | 2차 이후 | fused weight 분리 필요, 1차 제외 |
| G_PROJ / G_NORM / G_HEAD | 제외 | embed_dim / num_classes 에 고정 |
| 시작 모델 권장 | `vit_tiny_patch16_224` | 5.7M params, 빠른 검증 |

### Pruning 시 주의사항

```
1. fc1 출력 행(row) ↔ fc2 입력 열(col) 반드시 동일 인덱스
   fc1.weight[idx, :] = 0   →   fc2.weight[:, idx] = 0
   fc1.bias[idx]      = 0   →   fc2.bias는 건드리지 않음 (embed_dim 소속)

2. LayerNorm에는 running_mean / running_var 없음
   weight[idx] = 0, bias[idx] = 0 만 처리
   (BatchNorm과 달리 running_var를 1.0으로 설정할 필요 없음)

3. timm ViT의 MLP에는 fc1 / fc2 사이에 별도 norm 없음
   (일부 다른 아키텍처와 다름 — 확인 완료)

4. Reducing 시 EMA weights 사용 필수
   raw network weight는 매 step pruning으로 0 상태
   EMA가 실제 학습 성능을 보존
```

### 모델 선택 가이드

| 목적 | 권장 모델 | 이유 |
|------|---------|------|
| 빠른 파이프라인 검증 | `vit_tiny_patch16_224` | 23MB, CPU에서도 forward 빠름 |
| 실용 성능 검증 | `vit_small_patch16_224` | 88MB, 정확도-속도 균형 |
| 최종 배포 타겟 | `vit_base_patch16_224` | 346MB, ImageNet SOTA급 |

---

*작성 기준: timm 1.0.27, torch 2.9.1, Python 3.13.5 (macOS)*  
*모델 캐시: `~/.cache/huggingface/hub/`*
