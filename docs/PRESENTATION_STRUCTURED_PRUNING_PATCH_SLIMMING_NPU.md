# ViT Structured Pruning · Patch Slimming · NPU 평가 발표 자료

이 문서는 현재 프로젝트의 최종 구현과 실험 결과를 바탕으로 6~9장의 발표 슬라이드를
제작하기 위한 원고다. 권장 구성은 총 9장이다.

- Structured pruning/reducing: 3장
- Patch Slimming: 3장
- NPU 프로파일링 및 결과: 3장

발표 시간이 짧으면 각 절의 첫 두 장을 합쳐 총 6장으로 줄일 수 있다. 표의 수치는 별도
표기가 없는 한 실제 저장된 최종 실험 결과다.

---

## 0. 전체 발표 흐름

전체 압축 및 배포 과정은 다음과 같다.

```text
ImageNet pretrained ViT-Tiny / ViT-Small
  → Global L2 structured pruning + knowledge distillation
  → FFN channel physical reducing
  → 정적 Patch Slimming 탐색
  → block reconstruction fine-tuning
  → 전체 모델 recovery fine-tuning
  → static NHWC ONNX export
  → MXQ INT8 계열 양자화·컴파일
  → Aries2 NPU ImageNet 정확도·속도·메모리·전력 평가
```

이 파이프라인의 두 압축 축은 서로 다르다.

| 단계 | 줄이는 축 | 실제로 줄어드는 대상 | 주된 효과 |
|---|---|---|---|
| Structured pruning/reducing | FFN hidden channel | FC1 행, FC1 bias, FC2 열 | 파라미터와 FFN 연산 감소 |
| Patch Slimming | token sequence | block별 입력 token tensor 길이 | attention·activation·연산 감소 |

---

# Part I. Global L2 Structured Pruning & Reducing

## 슬라이드 1. 왜 ViT FFN 채널을 줄이는가?

### 슬라이드 제목

**ViT의 주요 파라미터 병목: Transformer FFN**

### 한 문장 핵심 메시지

> ViT block의 FFN은 embedding을 크게 확장하므로 파라미터 비중이 높고, hidden channel을
> 구조적으로 제거하면 NPU가 실행할 수 있는 더 작은 dense 행렬을 만들 수 있다.

### 본문에 넣을 내용

ViT의 각 Transformer block은 Multi-Head Self-Attention과 FFN으로 구성된다. FFN은 token
embedding 차원 `D`를 더 큰 hidden 차원 `M`으로 확장한 뒤 다시 `D`로 축소한다.

```text
Token [B, N, D]
    │
    ├─ FC1: D → M
    ├─ GELU
    └─ FC2: M → D
```

일반적인 ViT에서는 `M ≈ 4D`이기 때문에 FFN의 두 weight가 상당한 파라미터를 차지한다.

```text
FC1 weight: [M, D]
FC1 bias:   [M]
FC2 weight: [D, M]
```

hidden channel `j` 하나를 제거하려면 다음 세 요소를 같은 index로 함께 제거해야 한다.

- `FC1.weight[j, :]`: hidden channel을 생성하는 행
- `FC1.bias[j]`: 해당 hidden channel의 bias
- `FC2.weight[:, j]`: 해당 hidden channel을 다시 출력에 반영하는 열

이렇게 연결된 행과 열을 함께 제거해야 연산의 수학적 연결이 유지되고, 최종 행렬 크기도
`M → M'`으로 실제 축소된다.

### 추천 도식

```text
기존 FFN                          Reduced FFN

D → [ M hidden channels ] → D    D → [ M' channels ] → D
      ││×│×││×│                         ││││
      중요도가 낮은 채널 제거          실제 작은 Dense matrix
```

### 발표 멘트

“우리는 unstructured sparsity처럼 weight 일부만 0으로 만들지 않았습니다. FFN hidden
channel 단위로 FC1의 행과 FC2의 열을 함께 제거했습니다. 따라서 최종 모델은 sparse
kernel이 없어도 일반적인 dense 연산으로 실행할 수 있습니다.”

---

## 슬라이드 2. Global L2 pruning과 KD 기반 학습

### 슬라이드 제목

**Global L2 Channel Ranking과 Physical Reducing**

### 한 문장 핵심 메시지

> 모든 block의 FFN 채널을 하나의 중요도 공간에서 비교하고, KD로 정확도를 회복한 뒤
> mask를 실제 구조 축소로 변환한다.

### 채널 중요도

block `l`의 hidden channel `j`를 생성하는 `FC1.weight` 행의 L2 norm으로 중요도를
평가한다. FC2 weight는 중요도 산식에는 포함하지 않지만, 선택된 channel과 연결된 FC2
열은 reducing할 때 함께 제거한다.

\[
I_{l,j}=
\left\|W^{(l)}_{FC1}[j,:]\right\|_2
\]

- score가 작을수록 해당 hidden feature를 생성하는 FC1 행의 weight magnitude가 작다.
- 따라서 상대적으로 영향이 작은 channel로 간주해 우선 제거한다.
- 최종 제출 실험은 Progressive Taylor를 사용하지 않고 **L2 norm**을 사용했다.

### Global ranking

각 block에서 동일 비율을 제거하지 않는다. 12개 block에서 계산한 모든 FFN channel score를
모아 전역으로 정렬한다.

```text
Block 1 scores ─┐
Block 2 scores ─┤
      ...       ├─ Global sorting → 목표 parameter 감소량까지 선택
Block 12 scores ┘
```

이 방식은 block마다 정보 중요도가 다를 수 있다는 점을 반영한다. 중요한 block은 더 많은
channel을 유지하고, 덜 중요한 block은 더 큰 폭으로 줄어들 수 있다.

### 학습 중 pruning과 KD

학습 중에는 선택된 channel weight에 mask를 적용하고, optimizer update 후에도 제거 대상이
다시 살아나지 않도록 mask를 재적용한다.

```text
forward/backward
  → optimizer.step()
  → pruning mask 재적용
  → EMA update
```

정확도 회복에는 동일 아키텍처의 pretrained teacher를 사용했다.

- Tiny student ← pretrained Tiny teacher
- Small student ← pretrained Small teacher
- KD alpha: `0.5`
- temperature: `4.0`
- 데이터: ImageNet-1K training set

\[
L=(1-\alpha)L_{CE}+\alpha T^2L_{KL}
\]

### Physical reducing

학습 완료 후 zero mask를 그대로 배포하지 않고, 살아 있는 channel index만 추출한다.

```text
Masked checkpoint
  → alive FFN channel 검출
  → FC1 row / bias / FC2 column 복사
  → block별 MLP hidden dimension 재구성
  → reduced.pt 저장
```

이 단계 후에는 zero channel이 포함되지 않으며, 각 block은 실제로 폭이 작은 dense FFN을
갖는다. 구현의 핵심 파일은 다음과 같다.

- `pruning/vit_pruning.py`: L2 score, global ranking, mask 관리
- `pruning/train.py`: ImageNet pruning+KD 학습
- `pruning/engine.py`: 공용 학습·평가 loop
- `pruning/vit_reducing.py`: block별 FFN 구조 재생성
- `pruning/reduce.py`: reduced checkpoint 생성 CLI

### 발표 멘트

“Global이라는 말은 모든 layer를 같은 비율로 자른다는 뜻이 아닙니다. 반대로 모든 layer의
채널을 함께 순위화해 모델 전체에서 가장 덜 중요한 채널부터 제거한다는 의미입니다.”

---

## 슬라이드 3. Structured pruning/reducing 결과

### 슬라이드 제목

**30%·50% 물리적 모델 축소와 ImageNet 정확도**

### 평가 조건

- 모델: `vit_tiny_patch16_224`, `vit_small_patch16_224`
- 데이터: ImageNet-1K validation 50,000장
- 결과 단계: Patch Slimming 적용 전 `reduced.pt`
- 학습: Global L2 structured pruning + same-architecture KD

### 결과 표

| 모델 | 실제 파라미터 감소 | Parameters | FP32 parameter 용량 | Reduced Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|
| Tiny baseline | 0% | 5.717 M | 21.81 MiB | timm 기준 약 75%대 | 약 92%대 |
| Tiny30 reduced | **29.98%** | **4.003 M** | **15.27 MiB** | **72.796%** | **91.42%** |
| Tiny50 reduced | **49.94%** | **2.862 M** | **10.92 MiB** | **64.954%** | **86.68%** |
| Small baseline | 0% | 22.051 M | 84.12 MiB | timm 기준 약 81%대 | 약 95%대 |
| Small30 reduced | **30.01%** | **15.434 M** | **58.88 MiB** | **80.218%** | **95.35%** |
| Small50 reduced | **50.01%** | **11.023 M** | **42.05 MiB** | **75.448%** | **92.94%** |

Baseline의 동일 조건 실측 비교는 뒤의 NPU 결과 표에 제시한다. 위 표에서 baseline
정확도는 발표 문맥을 위한 timm reference이고, reduced 정확도는 현재 서버 실험의 실측값이다.

### 해석

1. 목표 압축률과 실제 parameter 감소율이 거의 정확히 일치했다.
2. Small30은 parameter를 약 30% 제거하면서 Top-1 `80.218%`를 유지해 가장 균형이 좋았다.
3. 50% 압축에서는 특히 Tiny의 정확도 하락이 커져 작은 모델의 redundancy가 제한적임을
   보여준다.
4. 결과물이 작은 dense 모델이므로 ONNX export와 NPU 컴파일에 별도 sparse operator가
   필요하지 않다.

### 추천 시각화

- 왼쪽: baseline→30%→50% parameter 용량 막대그래프
- 오른쪽: 각 모델의 Reduced Top-1 선 또는 막대그래프
- Small30 결과에 “Best trade-off” 표시

### 슬라이드 결론 문구

> Global L2 + KD로 목표한 물리적 압축률을 달성했으며, 특히 ViT-Small 30%에서 높은
> 정확도와 약 30%의 모델 축소를 동시에 확보했다.

---

# Part II. Static Patch Slimming

## 슬라이드 4. Patch Slimming이 줄이는 것은 무엇인가?

### 슬라이드 제목

**정적 Patch Slimming: 레이어가 깊어질수록 Token Sequence 축소**

### 한 문장 핵심 메시지

> Structured pruning이 FFN 폭을 줄였다면, Patch Slimming은 중요도가 낮은 patch token을
> 실제 tensor에서 제거해 attention과 activation을 줄인다.

### 기본 개념

224×224 이미지를 patch size 16으로 분할하면 14×14=`196`개 patch token이 생성된다.
CLS token을 포함한 초기 sequence 길이는 `197`이다.

```text
[CLS] + 196 patch tokens = 197 tokens
```

Self-attention의 attention matrix는 token 수 `N`에 대해 `[B,H,N,N]` 크기를 갖기 때문에
token 감소는 특히 `N²` 항을 줄인다.

```text
Attention score: QKᵀ → [B, H, N, N]
N 감소 → attention matrix와 QKV/MLP activation 감소
```

### Dynamic token pruning과의 차이

| Dynamic token pruning | 현재 Static Patch Slimming |
|---|---|
| 입력마다 runtime importance 계산 | 데이터셋으로 patch ID를 오프라인 검색 |
| TopK/Gather와 동적 shape 필요 | layer별 index와 shape가 상수 |
| 입력마다 token 위치가 바뀜 | 모든 입력이 동일 token ID를 유지 |
| 현재 NPU compiler에 부적합 | 고정 graph ONNX/MXQ 생성 가능 |

현재 모델은 token을 0으로 마스킹하는 방식이 아니다. 선택된 행만 남겨 tensor shape 자체를
`[B,N_in,D] → [B,N_out,D]`로 줄인다.

### 추천 도식

```text
Input patches       Middle blocks       Deep blocks
●●●●●               ●●●●                ●●
●●●●●    →          ●●●       →         ●●
●●●●●               ●●●                 ●

197 tokens          smaller N           still smaller N
```

### 발표 멘트

“0을 곱하는 masking이라면 attention matrix 크기는 그대로입니다. 우리는 선택 후 실제
sequence dimension을 줄였기 때문에 이후 block의 QKV, attention, MLP activation이 모두
작아집니다.”

---

## 슬라이드 5. 논문 방식 검색·복원·정적 ONNX 구현

### 슬라이드 제목

**Paper-style Patch Search와 NPU-safe Static Graph**

### 전체 적용 순서

```text
Global-L2 reduced model
  → attention 기반 path-energy patch score 계산
  → 낮은 score patch를 step 단위로 제거
  → 다음 block 출력에서 reconstruction error 평가
  → 후보 block 3 epochs fine-tuning
  → ε 조건을 만족하는 최대 축소 선택
  → 전체 compact model 30 epochs recovery fine-tuning
  → static ONNX export
```

### Patch 중요도

논문의 attention path-energy 아이디어를 구현했다. 여러 layer와 head를 거쳐 classifier로
전달되는 attention contribution을 동적계획법(DP)으로 누적하여 global patch score를 만든다.

- score mode: `paper_path_energy_dp`
- 모든 head 조합을 직접 열거하지 않고 DP로 누적
- score는 layer 내부에서 patch 제거 우선순위를 결정
- CLS token은 항상 유지
- layer가 깊어질수록 이전 layer에서 남은 token의 부분집합만 선택

### 후보 판정 오차

현재 block 출력만 비교하지 않고, pruning된 출력을 **다음 Transformer block까지 통과**시킨
결과로 reconstruction error를 계산한다. attention mixing 이후 발생하는 손상까지 반영하기
위함이다.

사용 metric은 sample-relative squared Frobenius error의 batch 평균이다.

\[
E = \frac{1}{B}\sum_{b=1}^{B}
\frac{\|\hat{Z}_{b}-Z_{b}\|_F^2}
{\|Z_{b}\|_F^2+\delta}
\]

- global tolerance `ε = 0.02`
- search step: 10 tokens
- 후보마다 block reconstruction fine-tuning: 3 epochs
- 최종 compact model recovery fine-tuning: 30 epochs

### 논문 방식으로 검색된 token schedule

각 배열은 embedding 입력 197개부터 각 block 출력, 마지막 CLS-only 출력까지 나타낸다.

```text
Tiny30
[197,151,151,151,141,131,111,111,111,111,111,111,1]

Tiny50
[197,121,121,121,121,121,111,111,111,111,111,111,1]

Small30
[197,197,197,197,197,101,101,101,101,101,101,101,1]

Small50
[197,197,197,197,141,101,101,101,101,101,101,101,1]
```

### NPU-safe ONNX 구현

- 입력: static NHWC `[1,224,224,3]`, FP32
- 출력: `[1,1000]`
- opset 17
- runtime TopK/Gather 없음
- 실제 token 감소 위치: 고정 상수 selection matrix MatMul
- token 수가 동일한 위치: identity selection 연산 제거
- 최종 CLS-only 선택: static slice/squeeze

초기에는 모든 layer에 identity selection MatMul이 남아 MXQ 양자화 오차가 누적됐다. 따라서
실제 token 수가 바뀌지 않는 selection은 ONNX graph에서 제거했다. Float ONNX와 PyTorch
출력의 최대 오차는 약 `4×10⁻⁶~9×10⁻⁶` 수준이었다.

### 구현 파일

- `patch_slimming/psvit/scoring.py`: patch score
- `patch_slimming/psvit/reconstruct.py`: 다음 block reconstruction
- `patch_slimming/psvit/search.py`: ε 기반 탐색
- `patch_slimming/psvit/compact.py`: 실제 가변 sequence forward
- `patch_slimming/psvit/slim_block.py`: selection 및 rectangular attention
- `patch_slimming/scripts/finetune.py`: 30-epoch recovery
- `patch_slimming/scripts/export_onnx.py`: NPU-safe export

---

## 슬라이드 6. Patch Slimming 결과와 GPU activation 분석

### 슬라이드 제목

**FP32 정확도 회복과 Activation 감소**

### ImageNet 정확도

아래 Patch Top-1은 논문 방식 Patch Slimming을 적용하고 30 epochs recovery fine-tuning한
PyTorch checkpoint 결과다. ONNX Runtime 출력은 PyTorch와 사실상 동일했다.

| 모델 | Reduced Top-1 | Patch Top-1 | Patch로 인한 변화 |
|---|---:|---:|---:|
| Tiny30 | 72.796% | **69.386%** | -3.410%p |
| Tiny50 | 64.954% | **60.384%** | -4.570%p |
| Small30 | 80.218% | **78.678%** | -1.540%p |
| Small50 | 75.448% | **73.330%** | -2.118%p |

### ONNX 변환 전 CUDA/activation 측정 조건

- 최종 `compact_finetuned_best.pt`
- batch 1, FP32, `[1,3,224,224]` random input
- `torch.inference_mode()`
- NPU export와 같은 `select-mode=matmul`
- warmup 10회, 측정 forward 50회
- random input을 사용했으므로 정확도 측정이 아니라 forward memory/shape 비교

### Reduced 대비 activation 결과

| 모델 | Reduced hook activation | Patch hook activation | 감소 | Attention memory 감소 |
|---|---:|---:|---:|---:|
| Tiny30 | 41.16 MiB | **21.38 MiB** | **48.1%** | **58.9%** |
| Tiny50 | 32.25 MiB | **14.60 MiB** | **54.7%** | **66.1%** |
| Small30 | 83.22 MiB | **46.42 MiB** | **44.2%** | **49.2%** |
| Small50 | 65.98 MiB | **34.93 MiB** | **47.1%** | **52.8%** |

`Hook activation`은 module별 출력 tensor volume의 합이며 동시 생존 peak는 아니다.
PyTorch CUDA allocator peak는 selection 임시 tensor와 caching 영향으로 reduced 대비 약간
증가했지만, layer별 tensor volume과 attention matrix는 명확히 감소했다.

### 완화형 3·6·9 참고 결과

정확도 회복을 위해 layer 3·6·9 뒤에서만 `171→151→131`로 줄이는 완화형도 평가했다.

| 모델 | 논문 방식 Top-1 | 완화형 Top-1 |
|---|---:|---:|
| Tiny30 | 69.386% | **70.942%** |
| Tiny50 | 60.384% | **63.862%** |
| Small30 | 78.678% | **79.014%** |
| Small50 | 73.330% | **74.152%** |

다만 뒤의 NPU INT8/MXQ 결과에서는 완화형 정확도가 개선되지 않았다. FP32 결과와 양자화
후 결과를 반드시 구분해서 발표해야 한다.

### 슬라이드 결론 문구

> FP32에서는 Patch Slimming 후에도 recovery fine-tuning으로 상당한 정확도를 유지하면서,
> attention 및 module activation tensor volume을 약 44~55% 줄였다.

---

# Part III. MXQ 변환 및 실제 NPU 프로파일링

## 슬라이드 7. NPU 평가 파이프라인과 측정 조건

### 슬라이드 제목

**ONNX → MXQ → Aries2 실제 장치 종합 평가**

### 한 문장 핵심 메시지

> 동일한 ImageNet 50,000장과 전처리 조건에서 정확도뿐 아니라 latency, NPU memory,
> utilization, power를 실제 장치에서 함께 측정했다.

### 배포 흐름

```text
Server_Compression
  baseline / reduced / patch ONNX 생성
        ↓
npu repository
  timm 방식 calibration data 생성
        ↓
  qbcompiler global8 MXQ 컴파일
        ↓
  qbruntime로 Aries2 NPU inference
        ↓
  mblt_tracker + process sampler로 telemetry 수집
```

### 최종 평가 조건

| 항목 | 설정 |
|---|---|
| 모델 수 | 16개: 4 group × baseline/reduced/paper patch/relaxed patch |
| Dataset | ImageNet validation 50,000장/모델 |
| 총 추론 | 800,000회 |
| 전처리 | timm validation preprocessing |
| Core mode | `global8` |
| Device | Aries2 device 0 |
| Warmup | 모델당 10회 |
| Accuracy | Top-1, Top-5, baseline 예측 일치율 |
| Latency | mean, P50, P90, P95, P99, max, FPS |
| NPU telemetry | memory, utilization, power, temperature |
| Host memory | RSS, USS, PSS, VMS 및 peak |
| 오류 | 16/16 모델 정상 실행, 오류 이미지 0장 |

NPU memory/utilization/power/temperature는 `mobilint-cli status`를 `mblt_tracker`로 500 ms
주기 관측했다. 프로세스 메모리는 100 ms 주기로 관측했다.

### 전처리 검증의 중요성

초기 legacy 전처리는 baseline Top-1을 약 `74%→44%`로 떨어뜨렸다. 이후 ONNX 생성에
사용한 timm 설정과 동일하게 resize, center crop, normalization을 맞춰 baseline 성능을
회복했다. 따라서 모든 최종 모델은 같은 image manifest와 timm 전처리를 사용했다.

### 발표 멘트

“속도만 한 이미지로 재지 않고 각 모델마다 ImageNet validation 5만 장을 모두 실행했습니다.
동일 manifest를 사용해 모델별 정확도와 latency, NPU telemetry를 직접 비교했습니다.”

---

## 슬라이드 8. NPU 속도와 메모리 결과

### 슬라이드 제목

**실제 NPU에서 최대 1.38× 가속, 최대 50% 메모리 감소**

### 논문 방식 Patch Slimming 핵심 결과

| 모델 | Baseline latency | Patch latency | 가속 | Baseline NPU peak | Patch NPU peak | 메모리 감소 |
|---|---:|---:|---:|---:|---:|---:|
| Tiny30 | 3.110 ms | **2.473 ms** | **1.26×** | 12 MB | **7 MB** | **41.7%** |
| Tiny50 | 3.070 ms | **2.263 ms** | **1.36×** | 12 MB | **6 MB** | **50.0%** |
| Small30 | 5.810 ms | **4.493 ms** | **1.29×** | 35 MB | **23 MB** | **34.3%** |
| Small50 | 5.820 ms | **4.213 ms** | **1.38×** | 35 MB | **19 MB** | **45.7%** |

### Reduced와 Patch Slimming의 역할 비교

| Group | Reduced latency | Paper Patch latency | Reduced NPU peak | Paper Patch NPU peak |
|---|---:|---:|---:|---:|
| Tiny30 | 2.945 ms | **2.473 ms** | 11 MB | **7 MB** |
| Tiny50 | 2.864 ms | **2.263 ms** | 10 MB | **6 MB** |
| Small30 | 5.545 ms | **4.493 ms** | 28 MB | **23 MB** |
| Small50 | 5.319 ms | **4.213 ms** | 24 MB | **19 MB** |

Structured reducing만으로도 parameter와 일부 NPU memory가 줄지만, attention sequence는
197개로 유지된다. Patch Slimming을 추가하면 attention과 activation이 함께 줄어 latency와
장치 memory가 추가로 개선된다.

### 정적 ONNX 기준 주요 연산량 감소

| 모델 | Reduced FLOPs 감소 | Paper Patch FLOPs 감소 |
|---|---:|---:|
| Tiny30 | 26.9% | **55.1%** |
| Tiny50 | 44.7% | **70.1%** |
| Small30 | 28.3% | **54.7%** |
| Small50 | 47.2% | **66.6%** |

이 FLOPs는 ONNX의 Conv/MatMul/Gemm static shape에서 `1 MAC=2 FLOPs`로 계산한 이론값이다.
elementwise, normalization, softmax, compiler fusion 및 tiling은 포함하지 않으므로 실제
NPU 연산량 자체와 동일하다고 해석하면 안 된다. latency와 NPU memory가 실제 장치 실측값이다.

### 추천 시각화

- 왼쪽: 네 모델의 baseline/reduced/patch latency grouped bar chart
- 오른쪽: baseline/reduced/patch NPU peak memory grouped bar chart
- 그래프 위에 `1.26~1.38×`, `34.3~50.0%↓`를 크게 표시

### 슬라이드 결론 문구

> 정적 token 축소가 실제 NPU graph에서도 동작했으며, 이론적 FLOPs 감소가 실제 latency와
> NPU memory 감소로 이어지는 것을 확인했다.

---

## 슬라이드 9. NPU 정확도 결과, 한계와 결론

### 슬라이드 제목

**효율 개선은 확인, Patch Slimming 양자화 정확도는 후속 과제**

### ImageNet 50,000장 NPU 정확도

| 모델 | Baseline Top-1 | Reduced Top-1 | Paper Patch Top-1 | Relaxed Patch Top-1 |
|---|---:|---:|---:|---:|
| Tiny30 | 74.238% | **71.738%** | 53.390% | 52.572% |
| Tiny50 | 74.238% | **64.188%** | 37.610% | 36.986% |
| Small30 | 80.550% | **79.948%** | 71.174% | 70.298% |
| Small50 | 80.550% | **74.726%** | 64.914% | 63.812% |

### 결과 해석

1. **Structured reduced MXQ는 정상적으로 정확도를 보존했다.**
   - Small30은 baseline 대비 `-0.602%p`로 가장 우수했다.
   - Tiny30은 `-2.500%p`였다.
2. **Patch Slimming은 속도와 메모리를 크게 개선했지만 정확도 하락이 남았다.**
   - FP32 PyTorch/ONNX는 정상이고 출력 일치도 검증됐다.
   - 따라서 주된 문제는 patch 선택 자체만이 아니라 MXQ 양자화·컴파일 경계와
     rectangular attention이 결합될 때 발생하는 추가 오차로 판단한다.
3. **Identity selection 제거로 심각한 붕괴는 일부 회복했다.**
   - 모든 layer에 남아 있던 identity MatMul은 불필요한 quantization boundary를 만들었다.
   - 이를 제거해 초기 약 9% 수준의 Tiny30 Top-1에서 50% 이상으로 회복했지만, FP32
     수준에는 도달하지 못했다.
4. **완화형 3·6·9도 NPU 정확도를 개선하지 못했다.**
   - FP32에서는 논문 방식보다 높았지만 MXQ에서는 오히려 소폭 낮았다.
   - 단순히 token을 더 남기는 것만으로 양자화 민감도를 해결할 수 없음을 보여준다.

### 기술적 원인 가설

- token selection 이후 activation distribution 변화
- static selection MatMul 전후에 삽입되는 quantization boundary
- layer마다 달라지는 rectangular attention shape
- calibration에서 각 stage의 activation range를 충분히 표현하지 못했을 가능성
- attention/LayerNorm/residual 경로의 scale mismatch 누적

Gather 기반 ONNX는 현재 compiler에서 unsupported이며, 상수 MatMul toy model은 정상
동작했다. 따라서 “상수 선택행렬 하나가 틀렸다”기보다 전체 ViT graph에서 selection과
양자화가 반복될 때 생기는 누적 오차가 더 유력하다.

### 후속 개선 방향

- Patch Slimming graph를 대상으로 quantization-aware fine-tuning 적용
- layer/stage별 calibration sample 및 range 분석
- selector 전후 tensor를 ONNX/quantized simulator/NPU에서 단계별 비교
- sensitive layer를 FP16 또는 higher precision으로 유지하는 mixed precision 검토
- compiler가 지원한다면 Gather/Slice 기반 selection과 selection fusion 검토
- 최종 목적에 따라 Small30 reduced를 안정적인 정확도 중심 모델로 사용하고,
  Patch Slimming은 효율 중심 실험 모델로 구분

### 최종 결론

```text
Global L2 structured pruning
  → 물리적 dense 모델 축소와 비교적 안정적인 NPU 정확도 달성

Static Patch Slimming
  → 실제 NPU latency·memory 추가 절감 달성
  → 그러나 MXQ 양자화 정확도 하락은 아직 해결 필요
```

발표에서는 “Patch Slimming이 완전히 성공했다”고 표현하기보다 다음과 같이 정리하는 것이
정확하다.

> Static Patch Slimming의 고정 shape NPU 실행 가능성과 실제 메모리·속도 개선은 검증했다.
> 다만 FP32에서는 유지되던 정확도가 MXQ 변환 후 크게 하락하여, 배포를 위해서는
> quantization-aware 최적화가 추가로 필요하다.

---

# 부록 A. 6장으로 줄이는 방법

발표 시간이 짧을 때는 다음처럼 합친다.

1. FFN 병목 + L2 중요도/Global ranking
2. KD/physical reducing + 결과
3. Patch Slimming 개념 + 논문 방식 탐색
4. Token schedule/ONNX 구현 + FP32 결과
5. NPU 측정 방법 + 속도·메모리 결과
6. NPU 정확도·한계 + 최종 결론

---

# 부록 B. 그래프 제작용 핵심 숫자

## B.1 Reduced 정확도

```text
model,parameter_reduction_pct,top1,top5
tiny30,29.9789,72.796,91.42
tiny50,49.9379,64.954,86.68
small30,30.0058,80.218,95.35
small50,50.0097,75.448,92.94
```

## B.2 FP32 Patch Slimming 정확도

```text
model,reduced_top1,paper_patch_top1,relaxed_patch_top1
tiny30,72.796,69.386,70.942
tiny50,64.954,60.384,63.862
small30,80.218,78.678,79.014
small50,75.448,73.330,74.152
```

## B.3 NPU latency와 memory

```text
model,baseline_ms,reduced_ms,paper_patch_ms,baseline_MB,reduced_MB,paper_patch_MB
tiny30,3.110,2.945,2.473,12,11,7
tiny50,3.070,2.864,2.263,12,10,6
small30,5.810,5.545,4.493,35,28,23
small50,5.820,5.319,4.213,35,24,19
```

## B.4 NPU 정확도

```text
model,baseline_top1,reduced_top1,paper_patch_top1,relaxed_patch_top1
tiny30,74.238,71.738,53.390,52.572
tiny50,74.238,64.188,37.610,36.986
small30,80.550,79.948,71.174,70.298
small50,80.550,74.726,64.914,63.812
```

---

# 부록 C. 수치 출처와 해석 주의사항

## Server-side 결과

- Structured/Patch 결과:
  `output/final_global_kd_l2_paper_20260814/FINAL_RESULTS.md`
- 논문 방식 architecture:
  `output/final_global_kd_l2_paper_20260814/<run>/03_patch_slimming_paper/architecture.json`
- PyTorch CUDA/activation profile:
  `output/final_global_kd_l2_paper_20260814/<run>/05_memory/`
- 완화형 결과:
  `docs/FINAL_STAGE369_RELAXED_20260826.md`

## NPU 결과

- 최종 요약:
  `../npu/results/final_benchmark/full16_20260831_141825/summary.md`
- 분석용 CSV:
  `../npu/results/final_benchmark/full16_20260831_141825/summary.csv`
- 원본 전체 지표:
  `../npu/results/final_benchmark/full16_20260831_141825/benchmark.json`

## 주의사항

- PyTorch CUDA memory와 NPU memory는 allocator와 측정 방식이 다르므로 절대값을 직접
  비교하지 않는다. 각 장치 안에서 baseline 대비 감소율을 비교한다.
- GPU hook activation은 생성된 module output volume의 합이지 동시 생존 peak가 아니다.
- NPU memory는 500 ms telemetry sampling으로 관측한 할당량이며 매우 짧은 내부 peak를
  놓칠 수 있다.
- ONNX 기반 FLOPs는 주요 Conv/MatMul/Gemm만 센 이론적 비교값이다.
- NPU 최종 결과 폴더의 `summary.md`와 `benchmark.json`은 16/16 완료를 기록하지만,
  `run_metadata.json`의 status 필드는 실행 시작 당시의 `running` 값이 남아 있다. 발표에는
  완료된 summary와 benchmark 결과를 기준으로 사용한다.
