# ViT 압축부터 Mobilint NPU 실행까지: 최종 구현·중간 기록·전체 결과

작성일: 2026-10-21 UTC

서버 실험: `final_20261006` · 배포본: `release_final_20261006`

NPU 최종 평가: `final_20261015` · 평가 MXQ: `mxq_final_20261006`

## 0. 문서의 목적과 읽는 방법

이 문서는 **baseline 프로파일링 → 목표 구조의 메모리 예산 설정 → global L2 FFN pruning과 KD 학습 → physical reducing → Statistical Patch Slimming 검색·학습 → ONNX → calibration·MXQ 컴파일 → 실제 NPU 평가**를 하나의 흐름으로 설명한다. 방법론뿐 아니라 실제 설정, 중간 수치, 최종 정확도·메모리·속도, 파일의 연결 관계를 함께 기록한다.

여기서 “최종”은 **현재 보존된 제출용 실험의 버전**이라는 뜻이다. 모든 단계에서 정확도가 유지되었다는 의미는 아니다. 특히 Patch Slimming의 서버 FP32 정확도와 NPU 정확도 사이에는 큰 차이가 남아 있으며, 이를 감추거나 baseline/reduced 결과와 혼합하지 않는다.

### 0.1 근거와 범위

- 서버의 코드·설정과 [배포 결과](../output/release_final_20261006/reports/FINAL_RESULTS.md), 모델별 JSON/JSONL, 원본 실행 로그를 대조했다.
- NPU의 [최종 결과](../../npu/results/final_benchmark/final_20261015/summary.md), 모델별 평가 기록, 예측 배열, 독립 메모리 기록을 사용했다. 이 문서를 작성하면서 NPU를 새로 실행한 것은 아니다.
- 현재 서버 release의 ONNX 12개와 `npu/assets/onnx`의 ONNX 12개는 **파일 SHA-256이 모두 일치**한다. NPU 결과에 기록된 ONNX 해시와도 연결된다.
- 최종 서버 실험은 2026-10-08 UTC에 완료됐다. NPU 전체 평가는 2026-10-15에 다른 서버에서 수행됐다.
- 과거 progressive/Taylor/완화형 등 별도 실험의 수치를 최종 표에 섞지 않는다. 본 실험 중간 기록은 포함하되, 별도 실험을 이 결과의 ablation으로 간주하지 않는다.
- “확인된 결과”, “코드에서 확인한 동작”, “아직 검증되지 않은 원인 가설”을 구분한다. 보존되지 않은 calibration 원본 목록이나 MXQ 내부 양자화 세부 설정은 추정해서 채우지 않는다.

문서의 block 번호는 **1–12**다. 코드의 `block_index`와 `layer`는 **0–11**이다. 예를 들어 문서의 Block 11은 코드의 `blocks[10]`이다. 토큰 수에는 항상 CLS 1개가 포함된다. 정확도 차이의 단위 `%p`는 percentage point이며, 메모리 `MiB = bytes / 2²⁰`다. NPU 도구가 보고한 `MB`는 별도로 표기한다.

### 0.2 전체 목차

1. 최종 결과 요약
2. 시스템 분담과 산출물 연결
3. 원본 모델·데이터·실행 환경
4. 메모리 프로파일링과 M_max
5. Global L2 FFN pruning과 KD 학습
6. Physical reducing
7. Statistical Patch Slimming의 중요도·복원·검색
8. 실제 compact forward와 최종 fine-tuning
9. 서버 FP32 정확도
10. 서버 메모리·activation 측정
11. ONNX 생성·NPU 대응 최적화
12. Calibration과 MXQ 변환
13. NPU 평가 프로토콜과 최종 결과
14. 서버–NPU 비교 및 결과 해석
15. 재현 명령·W&B·파일 추적
16. 확인 범위와 후속 검증
17. 부록: 전체 epoch·검색 후보·batch probe·해시

## 1. 최종 결과 요약

네 압축 실험은 Tiny30, Tiny50, Small30, Small50이다. 각 실험에서 baseline/reduced/patch_slimming 3종을 내보내므로 ONNX와 MXQ는 각각 12개다. Tiny baseline 2개, Small baseline 2개는 실험 그룹별 비교를 위해 중복 배치한 것이며, 서로 다른 pretrained baseline 4종을 뜻하지 않는다.

다음 표의 서버 정확도는 **PyTorch FP32, ImageNet validation 50,000장**이고 NPU 정확도는 **MXQ, 동일 규모의 validation 평가**다. 서버와 NPU의 전처리 구현은 완전히 같다고 증명되지 않았으므로 서버–NPU 정확도 차이를 전부 양자화 손실이라고 단정하지 않는다.

| 실험 | Reduced/patch 학습 파라미터 | 서버 Reduced Top-1 | 서버 Patch Top-1 | NPU Reduced Top-1 | NPU Patch Top-1 | NPU Patch latency ms | NPU Patch peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | 4,003,396 | 72.212 | 67.902 | 70.880 | 53.528 | 2.408 | 7 |
| Tiny50 | 2,862,256 | 63.230 | 59.252 | 62.442 | 41.164 | 2.388 | 5 |
| Small30 | 15,434,188 | 79.624 | 77.970 | 79.324 | 70.094 | 4.585 | 23 |
| Small50 | 11,023,204 | 74.002 | 70.736 | 73.616 | 61.082 | 4.204 | 18 |

Baseline 직접 측정값은 서버에서 Tiny **74.660/92.494%**, Small **80.502/95.862%**(Top-1/Top-5), NPU에서 Tiny **74.226/92.270%**, Small **80.486/95.852%**다. 공식 모델 표를 가져온 수치가 아니다.

핵심 결론은 다음과 같다.

- L2 pruning 후 reducing으로 실제 파라미터 수가 목표 약 30%·50% 줄었고, pruned와 reduced의 FP32 정확도는 네 실험 모두 같았다.
- Patch Slimming은 토큰 길이와 이론 연산량, 모듈 출력 용량 합을 줄였다. 하지만 최종 서버 FP32 Top-1도 reduced보다 1.654–4.310%p 낮다.
- NPU에서는 Patch Slimming의 latency와 보고 메모리가 줄었지만 정확도 손실이 추가로 크게 나타났다. “실행 성공”과 “정확도 유지 성공”은 별개다.
- 서버 CUDA peak는 Patch Slimming에서 reduced보다 오히려 커졌다. 이를 NPU의 메모리 감소와 같은 지표로 취급하면 안 된다.
- 메모리 예산은 **추가 학습하지 않은 목표 구조**로 미리 설정했다. 30epoch 동안 네 실험의 메모리 초과량과 패널티는 모두 0이었다.

## 2. 시스템 분담과 산출물 연결

```text
[Server_Compression: PyTorch/CUDA 서버]
pretrained baseline
  ├─ baseline FP32 정확도·메모리 기준
  └─ 사본을 목표 30%/50%로 pruning + reducing (추가 학습 0 step)
       └─ FP32 추론 peak → 고정 M_max

baseline에서 새 student 시작 + frozen baseline teacher
  → global L2 mask + CE/KL KD + 측정 기반 메모리 항, 30epoch
  → terminal checkpoint의 EMA + raw pruning mask
  → physical reducing
  → reduced 모델을 teacher로 복사
  → 정적 패치 중요도 통계 / 후보별 3epoch block reconstruction
  → architecture.json + searched_student.pt
  → compact 모델 CE fine-tuning 30epoch, best checkpoint
  → baseline / reduced / patch_slimming FP32 ONNX (총 12개)
  → PyTorch–ORT 수치 검증, 서버 FP32 평가·메모리 측정
                           │ ONNX + 해시 + 전처리 계약 전달
                           ▼
[npu: 다른 CUDA 서버의 compiler Docker]
ImageNet calibration → HWC float32 .npy 1,000개
  → qbcompiler vision_transformer preset, global8, cpu_offload=True
  → MXQ 12개
                           │ MXQ + ONNX/메타데이터 전달
                           ▼
[npu: 실제 Mobilint Aries 장치가 연결된 서버]
qbruntime load / infer
  → ImageNet 50,000장 × 12개: Top-1/5, latency, FPS
  → host/NPU telemetry: 메모리·전력·온도·활용률
  → 모델별 별도 프로세스 random-input 메모리 재검증
```

컴파일 때의 CUDA GPU 번호와 실제 추론 때의 `--device-id 0`은 서로 다른 장치를 뜻한다. 이 후속 작업의 프로젝트는 **npu**이며 `Orin_Compression`이 아니다. NPU 호스트의 운영체제에 `tegra`가 표시되더라도 여기서 측정한 모델 추론 backend는 qbruntime/Mobilint NPU다.

### 2.1 모델 하나의 핵심 파일

아래 `<model>`은 `tiny30|tiny50|small30|small50`이다.

| 단계 | 서버 release 내부 경로 | 의미 |
|---|---|---|
| 원본 | `<model>/00_budget/baseline.pt` | 학습·예산·평가의 출발 checkpoint |
| 예산용 목표 구조 | `<model>/00_budget/target_untrained.pt` | 추가 학습 없이 목표만큼 물리적으로 압축한 사본 |
| 메모리 예산 | `<model>/00_budget/budget.json` | 목표/원본 해시, 측정 조건, M_max |
| Pruned | `<model>/01_structural/checkpoint_final.pt` | 정확히 30epoch 완료, dense shape + mask |
| Reduced | `<model>/02_reduced/reduced.pt` | 실제 줄어든 FFN 구조와 가중치 |
| 검색 구조 | `<model>/03_patch_slimming_paper/architecture.json` | layer별 고정 global patch ID |
| 검색 가중치 | `<model>/03_patch_slimming_paper/searched_student.pt` | 후보별 block reconstruction이 반영된 가중치 |
| 최종 compact 가중치 | `<model>/03_patch_slimming_paper/compact_finetuned_best.pt` | 30epoch 중 validation best |
| 배포 그래프 | `<model>/04_onnx/*.onnx`, `*.verified.json` | FP32 정적 그래프와 parity/hash |
| 서버 메모리 | `<model>/05_memory/*.json` | peak, latency, activation 정보 |
| 최종 서버 정확도 | `<model>/06_evaluation/*.json` | baseline/pruned/reduced/patch FP32 |
| 실제 장치 모델 | `npu/assets/mxq_final_20261006/*.mxq` | 다른 서버에서 생성한 최종 바이너리 |
| 실제 장치 결과 | `npu/results/final_benchmark/final_20261015/` | 전체 평가와 독립 메모리 기록 |

서버 release 루트는 [output/release_final_20261006](../output/release_final_20261006/)다. 로컬 원본 `output/final_20261006`에는 재개용 optimizer, stdout, W&B 로그 등이 추가로 존재한다.

## 3. 원본 모델·데이터·실행 환경

### 3.1 ViT 구조

| 항목 | ViT-Tiny | ViT-Small |
|---|---:|---:|
| timm 모델명 | vit_tiny_patch16_224 | vit_small_patch16_224 |
| pretrained 설정 | augreg_in21k_ft_in1k | augreg_in21k_ft_in1k |
| 입력 해상도 / patch | 224×224 / 16×16 | 224×224 / 16×16 |
| 영상 patch 수 | 196 | 196 |
| CLS 포함 초기 토큰 | 197 | 197 |
| Transformer block | 12 | 12 |
| Embedding D | 192 | 384 |
| Attention head H | 3 | 6 |
| Head dimension | 64 | 64 |
| 원본 FFN hidden width | 768 = 4D | 1,536 = 4D |
| 분류 class | 1,000 | 1,000 |
| 전체 학습 파라미터 | 5,717,416 | 22,050,664 |

FFN은 `D → 4D → D`다. 확장된 hidden 공간에서 GELU 비선형 변환을 수행하고 다시 D로 돌아오므로 residual 덧셈이 가능하다. 본 구조적 pruning은 이 중 **중간 hidden channel만** 줄이며 D, head 수, patch embedding, 분류 head의 출력 class 수는 유지한다.

### 3.2 데이터와 전처리

서버의 실제 데이터 수는 **train 1,280,297장**, **validation 50,000장/1,000 class**다. 널리 알려진 ImageNet train 총수를 대신 기입하지 않고 이번 실행 로그의 로딩 수를 사용했다. 데이터 루트는 `data/imagenet/{train,val}`이며 배포 저장소에는 포함하지 않는다.

| 용도 | 데이터 출처와 수 | 전처리/사용 |
|---|---|---|
| Structural CE/KD | train 전체 1,280,297장, 30epoch | timm train augmentation, shuffle, drop_last |
| Patch 중요도 통계 | train 중 2,048장 | deterministic validation transform, seed 2022 |
| Block reconstruction 학습 | train 중 4,096장 | deterministic transform, loader shuffle, seed 2023 |
| 검색 후보 채택 평가 | train 중 별도 2,048장 | deterministic transform, seed 2024 |
| Compact 최종 CE 학습 | train 전체, 30epoch | timm train augmentation |
| epoch validation / 최종 FP32 평가 | validation 50,000장 | deterministic validation transform |
| NPU calibration | 코드의 입력 계약: 이미지 1,000장 | HWC FP32, 원본 최종 이미지 목록은 미보존 |
| NPU accuracy | validation 1,000 class × 50장 | OpenCV `timm` profile |

패치 검색의 세 subset은 실제 ID 파일을 대조하여 **서로 겹치지 않음**을 확인했다. 합계 8,192개의 train 이미지를 용도별로 나눈 것이며 ImageNet validation에서 떼어 온 것이 아니다. 다만 후보 판정 subset은 여러 후보를 반복 비교하는 데 사용되므로, 완전히 새로운 최종 평가셋으로 볼 수는 없다.

서버 학습 augmentation은 `color_jitter=0.4`, `rand-m9-mstd0.5-inc1`, random erasing `prob=0.25, mode=pixel, count=1`이다. 학습 loader는 `RandomSampler`, `drop_last=True`, pinned memory, worker 8개를 사용한다. 따라서 epoch마다 마지막 불완전 batch는 학습에서 빠진다.

서버 validation의 data config는 `input_size=(3,224,224), interpolation=bicubic, crop_pct=0.9, crop_mode=center, mean=std=(0.5,0.5,0.5)`다. 정규화는 RGB를 [0,1]로 바꾸고 `(x−0.5)/0.5`를 적용한다. NPU 쪽 구현과의 픽셀 단위 차이는 12장에서 별도로 설명한다.

### 3.3 실제 환경과 병렬 실행

서버는 Python 3.10.18, PyTorch 2.6.0+cu118, torchvision 0.21.0+cu118, timm 1.0.25, NumPy 1.26.4, ONNX 1.21.0, ONNX Runtime 1.23.2, W&B 0.19.2였다. GPU는 RTX 2080 Ti, 기록된 드라이버는 520.61.05다. 근거는 [environment.json](../output/release_final_20261006/provenance/environment.json)이다.

네 GPU를 하나의 DDP 학습에 묶은 것이 아니라 **모델별 독립 작업 네 개**로 사용했다. 각 worker 안에서는 하나의 GPU만 노출하므로 `cuda:0`이 해당 physical GPU를 뜻한다.

| 실험 | Physical GPU | Structural batch | 검색 batch | Compact fine-tuning 실제 batch |
| --- | --- | --- | --- | --- |
| Tiny30 | 0 | 384 | 32 | 512 |
| Tiny50 | 1 | 384 | 32 | 640 |
| Small30 | 2 | 192 | 16 | 256 |
| Small50 | 3 | 192 | 16 | 256 |

Batch는 synthetic 짧은 학습 probe에서 처리량과 잔여 메모리 여유 **1,536 MiB**를 확인하여 선택했다. 단순히 가장 큰 batch를 무조건 사용한 것이 아니다. Tiny50의 patch batch는 설정 기본값 512가 아니라 **실제 probe 선택값 640**이다. Probe 전체 결과는 부록에 기록했다.

## 4. 메모리 프로파일링과 M_max

관련 코드: [prepare_budget.py](../scripts/profiling/prepare_budget.py), [profile_reduced_memory.py](../scripts/profiling/profile_reduced_memory.py), [memory_budget.py](../pruning/memory_budget.py).

### 4.1 완료된 reduced 모델 없이 예산을 만드는 방법

1. Pretrained baseline을 읽고 baseline checkpoint를 저장한다.
2. Baseline의 사본에 목표 전체 파라미터 감소율 30% 또는 50%의 L2 pruning을 적용한다.
3. 사본의 zero channel을 physical reducing으로 제거한다.
4. 이 target 구조를 **추가 학습 0 step** 상태로 저장한다.
5. Baseline과 target을 같은 조건에서 별도 프로세스로 FP32 추론 측정한다.
6. Target의 `torch.cuda.max_memory_allocated()`를 M_max로 고정한다.
7. 본 structural 학습은 target 가중치에서 시작하는 것이 아니라 기록된 baseline에서 시작한다.

여기서 “untrained target”은 무작위 초기화 모델이 아니다. **Pretrained 가중치를 압축했지만 복원 학습은 하지 않은 목표 구조**다. M_max를 만들기 위해 과거에 학습 완료한 reduced 모델을 준비할 필요는 없다.

예산을 파라미터 수만으로 계산하거나 baseline peak의 70%/50%로 곱한 것도 아니다. 실제 추론에서는 입력, 중간 텐서, 비-FFN 가중치 등도 필요하므로 전체 파라미터 30% 감소와 peak 30% 감소는 같지 않다.

| 실험 | 전체 파라미터 목표 감소 | Baseline CUDA peak MiB | Target/M_max MiB | Baseline 대비 peak 감소 | 예산용 추가 학습 step |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | 30% | 32.831055 | 26.152344 | 20.34% | 0 |
| Tiny50 | 50% | 32.831055 | 21.801758 | 33.59% | 0 |
| Small30 | 30% | 97.157715 | 71.555176 | 26.35% | 0 |
| Small50 | 50% | 97.157715 | 54.372070 | 44.04% | 0 |

### 4.2 측정 조건과 후보 갱신

예산과 학습 중 후보는 모두 **FP32, batch 1, 224×224, random input, warmup 10회, 측정 forward 50회** 조건이다. 측정 전 warmup을 마친 뒤 CUDA를 동기화하고 peak 통계를 초기화한다. 50회 후 동기화하여 peak를 읽는다. **50회 peak의 평균이 아니라, 전체 측정 구간에서 관측한 allocator 최대값**이다.

학습 모델은 dense mask 상태이므로 그대로 측정하면 reduced deployment 메모리를 나타내지 못한다. 따라서 현재 raw 모델과 EMA의 CPU snapshot을 만들고, 별도 프로세스에서 raw mask를 EMA에 적용한 뒤 reducing하여 후보 추론 메모리를 측정한다. 원본 학습 모델의 shape는 변경하지 않는다.

최초 마스킹 직후 epoch 0의 예산 비교를 수행한다. 이후 각 epoch에 최초로 갱신 가능한 batch의 optimizer/mask/EMA 처리 뒤 측정하며, 갱신한 scalar는 **다음 batch**부터 사용한다. “epoch 마지막에만 측정한다”는 구현은 아니다. 최종 로그는 모델별 `epoch_0000.json`부터 `epoch_0029.json`까지 30개다.

```text
M_candidate = max_memory_allocated(reduced snapshot) / 2²⁰
L_mem = λ × max(0, M_candidate − M_max)
λ = 1.0
L_total = L_task_KD + L_mem
```

이 L_mem은 외부 측정값에서 만든 **상수 scalar**다. `M_candidate`가 autograd에 연결되지 않으므로 weight에 대한 추가 gradient는 없고, 예산을 넘으면 자동으로 더 많은 channel을 자르는 제어기도 없다. 예산 초과를 loss/로그에 반영하는 기능과, 미분 가능한 메모리 정규화 또는 자동 구조 탐색은 구분해야 한다.

### 4.3 실제 측정 결과

각 모델의 30개 후보 메모리는 모두 해당 M_max와 같았다. 따라서 초과량과 L_mem은 전 epoch 0이다. 예산용 target의 layer별 FFN width와 최종 reduced width도 실제 파일에서 일치했다.

| 실험 | 후보 측정 횟수 | 후보 최소 MiB | 후보 최대 MiB | 초과 epoch 수 | 최대 L_mem |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | 30 | 26.152344 | 26.152344 | 0 | 0.000000 |
| Tiny50 | 30 | 21.801758 | 21.801758 | 0 | 0.000000 |
| Small30 | 30 | 71.555176 | 71.555176 | 0 | 0.000000 |
| Small50 | 30 | 54.372070 | 54.372070 | 0 | 0.000000 |

이 결과에서 “메모리 패널티가 정확도를 높였다”거나 “예산 위반을 gradient로 교정했다”고 결론 낼 수 없다. 패널티 유무만 바꾼 통제 실험도 본 최종 묶음에는 없다. 또한 M_max는 **structural reduced FP32 추론**의 예산이며, 이후 Patch Slimming implementation이나 NPU runtime의 메모리를 제한하는 예산은 아니다.

## 5. Global L2 FFN pruning과 KD 학습

관련 코드: [vit_pruning.py](../pruning/vit_pruning.py), [train.py](../pruning/train.py), [engine.py](../pruning/engine.py).

### 5.1 정확히 무엇을 자르는가

한 block의 FFN을 다음과 같이 표시한다.

```text
X: [B,N,D]
FC1: W1 [M,D], b1 [M]
GELU
FC2: W2 [D,M], b2 [D]
출력: [B,N,D]
```

Hidden channel j의 점수는 **FC1 weight의 j번째 행 L2 norm**이다.

```text
score(l,j) = sqrt(sum_d W1_l[j,d]²)
```

점수가 낮은 channel을 제거할 때 함께 마스킹하는 항목은 `W1[j,:]`, `b1[j]`, `W2[:,j]`다. FC2 bias `b2`는 D차원 출력에 붙기 때문에 제거하지 않는다. Channel 하나를 physical reducing하면 파라미터 **2D+1개**가 감소한다. Tiny는 385개, Small은 769개다.

주의할 점은 “channel”이 영상 RGB channel이나 patch token이 아니라 **FFN 내부 feature dimension**이라는 것이다. 이 단계에서 각 block의 토큰 수는 여전히 197이다.

### 5.2 순위와 제거량

- 모든 block의 FC1-row L2 점수를 하나로 모아 **global 순위**를 만든다. Layer마다 동일 비율을 자르는 방식이 아니다.
- Block별 최대 sparsity 0.95와 최소 생존 channel 4개 제한을 둔다. 제한을 넘어서 잘릴 후보는 global 선택에서 제외한다.
- 목표는 전체 모델 파라미터 수의 30%·50%다. FFN 파라미터만의 30%·50%도, 각 block hidden width의 30%·50%도 아니다.
- 전체 목표를 만족하는 총 channel 제거 수를 얻기 위해 64회 bisection으로 대응 sparsity를 정하고 정수 channel 개수로 반올림한다. 이 때문에 실제 전체 감소율은 정확히 30.000/50.000%가 아닐 수 있다.
- 실제 분배는 global 순위로 결정하므로 block별 생존 width가 서로 다르다.
- 점수 갱신 주기는 100 prune hook step이다. 학습 전 최초 mask를 적용하고, optimizer 업데이트 후에도 mask를 다시 적용한다.

이는 progressive Taylor 기법이 아니다. 목표 제거량은 처음부터 정해져 있고 Taylor gradient score나 점진적 목표 sparsity schedule을 사용하지 않는다. 구현에서 점수를 다시 계산하더라도 “반복할 때마다 목표 30%를 추가로 자른다”는 뜻은 아니다.

### 5.3 Dense mask 학습 순서

```text
초기 baseline student → global L2 mask 적용
batch forward → CE + teacher logit KD + 현재 메모리 scalar
  → backward / gradient clipping / AdamW step
  → pruning mask 재적용
  → EMA update
  → 해당 epoch 메모리 갱신이 필요하면 snapshot → reducing → profile
  → 다음 batch
```

학습 중 pruning은 **dense tensor 값에 0을 곱하는 방식**이다. FC1/FC2 shape 자체는 아직 줄이지 않으므로 이 단계만으로 dense 모델 파라미터 storage가 감소하지 않는다. Shape를 실제로 바꾸는 단계는 reducing이다.

### 5.4 KD의 종류와 손실

Teacher는 같은 모델 크기의 frozen pretrained baseline이다. Tiny 학생은 Tiny teacher, Small 학생은 Small teacher를 사용한다. Teacher는 eval/no_grad이고 학생과 같은 증강 이미지를 받는다.

```text
p_T = softmax(teacher_logits / T)
p_S = softmax(student_logits / T)
L_KD = T² × mean_batch KL(p_T || p_S)
L_task_KD = (1−α) × CE_label_smoothing + α × L_KD
L_total = L_task_KD + L_mem
T = 4.0, α = 0.5, label_smoothing = 0.1
```

구현은 `F.kl_div(log_softmax(student/T), softmax(teacher/T), reduction="batchmean") * T*T`다. **출력 logit 기반 KL KD**이며 샘플 사이 거리/각도 관계를 맞추는 관계형 KD는 아니다. Patch 검색의 feature reconstruction과도 다른 목적함수다.

### 5.5 실제 structural 학습 recipe

| 항목 | 최종 실행값 |
|---|---|
| Epoch 종료 / checkpoint 선택 | 30epoch / terminal epoch |
| Optimizer | AdamW |
| Base LR / 최소 LR | 5e-5 / 1e-6 |
| Weight decay / grad clipping | 0.05 / norm 1.0 |
| Warmup | 5epoch, LinearLR 시작 factor 0.01 |
| LR schedule horizon | 50epoch, warmup 이후 cosine 45epoch |
| AMP | 사용 |
| EMA | 사용, decay 0.9998 |
| KD | α=0.5, T=4 |
| 메모리 항 | λ=1, 매 epoch 갱신 |
| 초기 seed | 2022 |

**학습 종료 30epoch과 LR horizon 50epoch은 다른 설정**이다. 30epoch 종료 시 cosine이 최소 LR까지 모두 내려간 30epoch schedule을 완료한 것으로 설명하면 안 된다. Scheduler는 epoch 종료 후 갱신한다.

Structural 결과는 best checkpoint가 아니라 정확히 30epoch 완료 checkpoint다. 코드의 epoch 값 29는 0-based라서 30번째 epoch를 뜻한다. 이후 reducing은 이 checkpoint의 EMA에 raw pruning mask를 반영하여 수행한다.

### 5.6 중간 validation Top-1

다음은 학습 로그의 **AMP validation / EMA 모델** 값이다. 최종 FP32 재평가와 소수점이 조금 다를 수 있다. Epoch 1 값은 이미 1epoch 학습한 후이며, pruning 직후 학습 0회 정확도가 아니다.

| Epoch | Tiny30 | Tiny50 | Small30 | Small50 |
| --- | --- | --- | --- | --- |
| 1 | 6.056 | 0.478 | 21.436 | 0.970 |
| 5 | 63.646 | 39.254 | 74.628 | 59.740 |
| 10 | 69.804 | 55.934 | 77.850 | 68.896 |
| 15 | 70.964 | 59.856 | 78.636 | 71.358 |
| 20 | 71.634 | 61.510 | 79.088 | 72.624 |
| 25 | 71.974 | 62.602 | 79.340 | 73.514 |
| 30 | 72.216 | 63.238 | 79.640 | 74.008 |

전체 epoch별 train loss/accuracy와 validation Top-1/5는 부록 A에 있다.

## 6. Physical reducing

관련 코드: [vit_reducing.py](../pruning/vit_reducing.py), [reduce.py](../pruning/reduce.py).

FC1 row norm이 0이 아닌 생존 index 집합 I를 구하고, 새 Linear layer를 만들어 다음과 같이 복사한다.

```text
W1_reduced = W1[I, :]        # [M',D]
b1_reduced = b1[I]           # [M']
W2_reduced = W2[:, I]        # [D,M']
b2_reduced = b2              # [D], 그대로

D → M → D   ⇒   D → M' → D
```

정확히 0인 hidden channel은 FC1 bias와 FC2 연결까지 함께 0으로 마스킹되어 있으므로 삭제 전후 계산이 유지된다. EMA를 사용할 때는 raw 모델의 zero mask를 먼저 EMA에 옮겨, EMA에 남아 있을 수 있는 작은 잔여값 때문에 죽은 channel이 살아 있는 것으로 판정되지 않도록 한다.

최종 평가에서 pruned/reduced의 Top-1·Top-5뿐 아니라 기록된 loss도 모델별로 같았다. 이는 정확도 평가에서의 동등성을 확인한 결과이며, 모든 가능한 입력의 bitwise equivalence를 그 표 하나로 증명한다는 뜻은 아니다.

### 6.1 파라미터와 FP32 weight 용량

| 실험 | 원본 parameters | Reduced parameters | 실제 제거율 | FP32 weight MiB | 제거한 FFN channel 수 |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | 5,717,416 | 4,003,396 | 29.97893% | 15.2717 | 4452 |
| Tiny50 | 5,717,416 | 2,862,256 | 49.93794% | 10.9186 | 7416 |
| Small30 | 22,050,664 | 15,434,188 | 30.00579% | 58.8768 | 8604 |
| Small50 | 22,050,664 | 11,023,204 | 50.00965% | 42.0502 | 14340 |

FP32 weight는 `parameters × 4 bytes / 2²⁰`이며 checkpoint 파일 크기, ONNX 파일 크기, 추론 peak와는 다른 값이다. Pruned는 zero가 많더라도 dense parameter 수가 baseline과 같다. Patch Slimming은 token 흐름을 바꾸기 때문에 학습 parameter 수는 reduced와 같다.

### 6.2 최종 block별 FFN hidden width

이 width는 예산용 target, 최종 reduced, patch 모델에 공통이다. Attention의 embedding D는 줄지 않는다.

| Block | Tiny 원본 | Tiny30 | Tiny50 | Small 원본 | Small30 | Small50 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 768 | 462 | 298 | 1536 | 707 | 321 |
| 2 | 768 | 470 | 181 | 1536 | 740 | 240 |
| 3 | 768 | 354 | 80 | 1536 | 699 | 253 |
| 4 | 768 | 352 | 117 | 1536 | 708 | 218 |
| 5 | 768 | 357 | 121 | 1536 | 645 | 236 |
| 6 | 768 | 374 | 147 | 1536 | 694 | 256 |
| 7 | 768 | 359 | 133 | 1536 | 665 | 229 |
| 8 | 768 | 337 | 109 | 1536 | 659 | 211 |
| 9 | 768 | 422 | 85 | 1536 | 806 | 336 |
| 10 | 768 | 414 | 104 | 1536 | 1054 | 491 |
| 11 | 768 | 354 | 175 | 1536 | 1348 | 1126 |
| 12 | 768 | 509 | 250 | 1536 | 1103 | 175 |

## 7. Statistical Patch Slimming: 중요도 통계·복원·검색

참고한 연구는 Tang 등의 *Patch Slimming for Efficient Vision Transformers* (CVPR 2022)다. 최종 출력에 대한 영향이 작은 patch를 통계적으로 식별하고 뒤쪽 layer에서 앞쪽으로 선택을 진행한다는 접근을 따른다. 본 프로젝트는 이를 **이미 FFN이 reduced된 모델**, 독립 subset, relative reconstruction metric, NPU용 compact graph에 맞게 구현했다. 따라서 폴더명의 `paper`는 논문 기반 탐색을 가리키며, 논문의 모든 실험 조건을 그대로 재현했다는 뜻은 아니다. [공식 논문 페이지](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Patch_Slimming_for_Efficient_Vision_Transformers_CVPR_2022_paper.html)

관련 코드: [scoring.py](../patch_slimming/psvit/scoring.py), [reconstruct.py](../patch_slimming/psvit/reconstruct.py), [search.py](../patch_slimming/psvit/search.py), [data.py](../patch_slimming/psvit/data.py), [architecture.py](../patch_slimming/psvit/architecture.py).

### 7.1 무엇이 정적인가

통계 단계에서 여러 이미지의 patch 중요도를 평균하여 **모델별·block별 유지 patch ID**를 한 번 정한다. 그 ID를 `architecture.json`에 저장하고 모든 추론 이미지에 동일하게 사용한다.

- Global ID 0은 CLS, 1–196은 원본 14×14 grid patch다.
- 각 block의 유지 ID는 정렬되며 CLS가 항상 포함된다.
- 다음 block이 요구하는 ID는 현재 block의 출력 ID의 부분집합이다.
- Runtime에 이미지마다 importance scoring, TopK, thresholding을 하지 않는다.
- 입력마다 token 길이가 달라지는 동적 pruning과 다르다. Block마다 길이는 다르지만 **각 block의 길이는 배포 시 고정**이다.

`architecture.json`은 가중치를 대체하지 않는다. 실제 모델을 복원하려면 줄어든 FFN 구조/가중치와 이 token ID 정보가 함께 필요하다. 최종 ONNX에는 ID에 대응하는 선택 연산이 이미 고정되어 들어 있으므로 ONNX/MXQ 추론 시 JSON을 다시 읽을 필요가 없다.

### 7.2 중요도 계산에 사용하는 hybrid network

탐색 대상 block을 t라고 하자. 아직 앞쪽 block의 patch 수가 결정되지 않았으므로 다음과 같이 계산한다.

1. t보다 앞선 block은 전체 토큰으로 forward한다.
2. t는 전체 output token과 attention을 계산한다.
3. t보다 뒤쪽 block은 이미 결정한 compact ID로 실행한다.
4. 뒤쪽 rectangular attention을 원본 global ID 좌표의 197×197 행렬로 scatter한다. 제외된 row/column은 0이다.

이것은 **통계 계산용 임시 network 표현**이다. 실제 배포 때 제거한 token을 197개로 다시 복원한다는 뜻은 아니다. Score 통계에서만 block 사이 patch ID를 같은 좌표로 맞추기 위한 표현이다.

### 7.3 Head-path energy DP

여기서 A_l^h는 head h의 attention 확률이며, 토큰 선택행렬 P와 혼동하지 않기 위해 기호를 분리한다. Score 계산 코드는 다음 연산을 수행한다.

```text
G = I_197
for downstream block l in reverse order:
    G = mean_h( transpose(A_l^h) @ G @ A_l^h )

downstream_energy[i] = max(G[i,i], 0)
U_t^h = A_t^h @ abs(Z_{t-1})
current_energy[i] = mean_h( sum_d U_t^h[i,d]² )

sample_score[i] = downstream_energy[i] × current_energy[i]
mean_score[i] = mean_over_2048_images(sample_score[i])
```

해석하면, **현재 patch가 담고 있는 feature energy**와 **그 patch가 뒤쪽 layer의 유지 출력에 전달되는 정도**를 함께 본다. 특정 한 head의 CLS attention만 보고 자르는 방법이 아니다.

여러 downstream head 조합을 직접 전부 열거하지 않고 위의 동적 계획법(DP)으로 집계한다. 이 DP는 계산 알고리즘의 이름이지 입력별 동적 token pruning을 뜻하지 않는다. 코드에는 작은 case에서 head-path를 완전 열거한 결과와 DP를 대조하는 함수가 있다.

Head 합 대신 평균을 사용한다. 한 layer 안에서 동일한 head 구성에 대한 공통 scale이므로 해당 layer 내부의 patch 순위에는 영향을 주지 않는 정규화이며, 다른 layer의 score 절댓값을 직접 비교해 전체 global patch 순위를 만드는 구현은 아니다. 이미지 간 score 누적은 float64로 하고 NaN/Inf를 검사한다.

### 7.4 뒤에서 앞으로 탐색하는 순서

마지막 Block 12의 출력은 분류에 필요한 **CLS 1개**로 먼저 고정한다. 그 다음 Block 11 → 10 → … → 1 순서로 검색한다.

```text
K_12 = {CLS}

for t = 11, 10, ..., 1:              # 문서의 1-based block 번호
    base = K_(t+1)
    candidates = importance 내림차순의 나머지 patch
    r = 0
    반복:
        K_trial = sorted(base ∪ candidates[:r])
        현재 block t를 3epoch reconstruction 학습
        다음 block t+1까지 forward한 독립 평가 오차 계산
        오차 ≤ 0.02이면 K_t 확정
        아니면 r을 10 증가
        전체 토큰도 조건 실패하면 전체 유지 fallback
```

후보 간 **block weight는 누적**한다. 실패한 후보마다 원래 weight로 되돌리지 않는다. 반면 후보별 optimizer는 새로 생성한다. 따라서 뒤에서 채택된 후보의 가중치는 이전 후보의 복원 학습 영향을 포함한다. 이 구현은 token 수만 바꾼 완전 독립 후보 비교가 아니다.

`K_(t+1) ⊆ K_t`를 유지하므로 실행 방향에서는 token 수가 증가하지 않는다. 그러나 매 block 반드시 감소해야 한다는 제약이나 “3/6/9에서만 줄인다”는 schedule은 없다.

### 7.5 Reconstruction은 어디의 오차인가

이 부분은 구현 해석에서 가장 중요하다. 현재 block t의 출력만 잘라 teacher와 비교하는 것이 아니라 **다음 block t+1을 통과한 뒤** 비교한다.

```text
Teacher: frozen reduced model, 전체 token
  이미지 → teacher prefix → Z_(t−1)
         → teacher block t → teacher block t+1 → target[K_(t+1)]

Student:
  같은 teacher Z_(t−1)
         → slim block t, K_trial
         → slim block t+1, K_(t+1) → prediction

loss: prediction과 target[K_(t+1)]의 MSE
```

예를 들어 Block 11을 검색할 때 target은 **Block 12를 통과한 CLS 출력**이다. Block 11 자체 출력의 CLS를 즉시 비교하는 것이 아니다. 삭제된 patch가 다음 attention에서 정보를 전달하지 못하는 효과가 판정에 포함된다.

다만 입력 prefix는 teacher의 full-token feature를 사용한다. 이미 압축된 앞쪽 모든 block을 거친 end-to-end student prefix까지 포함한 전체 오차는 아니다. 이 **국소 2-block 복원**의 한계는 최종 compact 모델 전체 fine-tuning과 validation에서 별도로 확인해야 한다.

Teacher는 이 단계에 들어올 때 reduced 모델을 복사하여 고정한 것이다. Structural KD에서 사용한 원본 baseline teacher와 구별한다. 학습 가능한 것은 현재 `student.blocks[t]` 파라미터뿐이며, 다음 block weight는 고정하지만 현재 block으로 gradient가 전달되어야 하므로 다음 block forward 전체를 `no_grad`로 막지는 않는다.

### 7.6 학습 loss와 후보 채택 metric은 다르다

Block reconstruction 학습은 element-wise MSE다. Candidate 하나마다 별도 4,096장으로 3epoch, AdamW LR=1e-5, weight decay=0, grad norm clipping=1을 사용한다. 이 **3epoch은 4,096장 subset 기준**이다. ImageNet train 전체를 3회 돈 것으로 해석하면 안 된다.

후보 채택은 학습에 쓰지 않은 2,048장에 대한 **샘플별 relative squared Frobenius의 평균**이다.

```text
각 이미지 b:
    e_b = sum((prediction_b − target_b)²)
          / max(sum(target_b²), 1e−12)

acceptance_error = mean_b(e_b)
채택 조건: acceptance_error ≤ ε = 0.02
```

이는 다음 값들과 다르다.

- Element-wise MSE: 전체 제곱오차를 B×N×D로 나눈 값.
- Raw Frobenius squared: 모든 제곱오차의 합.
- 전체 데이터 합의 relative error: Σ오차 / Σtarget energy.
- Top-1 2%p 손실 제한.

코드는 이들 보조 metric과 상대오차 p50/p95도 기록하지만 **채택 기준은 샘플별 비율의 평균**이다. `ε=0.02`를 논문의 다른 정규화 오차에서 같은 의미의 숫자라고 단정하지 않는다.

### 7.7 실제 탐색량

각 후보는 4,096장 × 3epoch을 학습한다. 다음의 누적 이미지 수는 반복 노출 횟수이며 서로 다른 이미지 개수가 아니다. 후보 소요 시간 합계는 `candidate_seconds`의 합으로 **복원 학습+후보 평가 구간**만 포함한다. 중요도 통계, 입출력, 전체 fine-tuning까지 포함한 search wall time이 아니다.

| 실험 | 검사 후보 수 | 후보별 epoch | 누적 학습 이미지 노출 | 후보 학습+평가 시간 합 s | 전체 유지 fallback block |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | 23 | 3 | 282,624 | 767.72 | 없음 |
| Tiny50 | 23 | 3 | 282,624 | 781.52 | 없음 |
| Small30 | 31 | 3 | 380,928 | 1160.58 | 1, 2, 3 |
| Small50 | 28 | 3 | 344,064 | 1060.34 | 없음 |

총 **105개 후보**, 각 3epoch의 reconstruction을 수행했다. 전체 후보별 3epoch loss, MSE, relative error, p95, 채택 여부는 부록 B에 있다.

### 7.8 최종 layer별 판정 오차

아래 값은 검색 완료 당시의 오차다. 이후 compact 전체 fine-tuning 뒤 다시 측정한 reconstruction error가 아니다.

| Block | Tiny30 relative error | Tiny50 relative error | Small30 relative error | Small50 relative error |
| --- | --- | --- | --- | --- |
| 1 | 0.014008 | 0.011517 | 0.030222 (전체 유지) | 0.018250 |
| 2 | 0.016802 | 0.011983 | 0.046311 (전체 유지) | 0.016477 |
| 3 | 0.018654 | 0.016825 | 0.026703 (전체 유지) | 0.019966 |
| 4 | 0.018714 | 0.015794 | 0.019313 | 0.019827 |
| 5 | 0.018138 | 0.017277 | 0.013556 | 0.010328 |
| 6 | 0.011394 | 0.011068 | 0.002287 | 0.001940 |
| 7 | 0.000287 | 0.000270 | 0.002220 | 0.001785 |
| 8 | 0.000211 | 0.000173 | 0.002201 | 0.001755 |
| 9 | 0.000175 | 0.000155 | 0.002465 | 0.001871 |
| 10 | 0.000141 | 0.000133 | 0.008050 | 0.007222 |
| 11 | 0.019094 | 0.018580 | 0.017659 | 0.015394 |

Block 12는 처음부터 CLS-only로 지정하며, architecture에 저장된 0은 별도 reconstruction을 측정하여 0오차가 나왔다는 의미가 아니다.

**Small30의 Block 1/2/3은 ε=0.02를 통과하지 않았다.** 각각 전체 197개를 유지해도 오차가 약 0.030222 / 0.046311 / 0.026703이어서 `all_tokens` fallback으로 확정했다. 따라서 “모든 layer가 0.02 이하를 보장한다”는 설명은 맞지 않는다.

전체 token을 유지해도 현재/다음 block weight가 이미 reconstruction 과정에서 달라졌을 수 있으므로 원래 teacher와 동일해진다고 보장되지 않는다. 로그는 fallback 사실을 입증하지만, 그 오차 중 weight drift와 다음 block 변경이 차지하는 기여를 별도 실험으로 분해하지는 않았다.

### 7.9 각 block의 실제 입력·출력 token 수

모든 수에는 CLS 1개가 포함된다. `입력→출력`으로 표기한다. Baseline과 reduced는 12개 block 모두 197→197이며 block stack 이후 CLS를 분류 head에 넣는다.

| Block | Baseline / Reduced | Tiny30 | Tiny50 | Small30 | Small50 |
| --- | --- | --- | --- | --- | --- |
| 1 | 197→197 | 197→121 | 197→121 | 197→197 | 197→171 |
| 2 | 197→197 | 121→121 | 121→121 | 197→197 | 171→151 |
| 3 | 197→197 | 121→121 | 121→121 | 197→197 | 151→131 |
| 4 | 197→197 | 121→121 | 121→121 | 197→141 | 131→121 |
| 5 | 197→197 | 121→121 | 121→121 | 141→111 | 121→111 |
| 6 | 197→197 | 121→111 | 121→111 | 111→111 | 111→111 |
| 7 | 197→197 | 111→111 | 111→111 | 111→111 | 111→111 |
| 8 | 197→197 | 111→111 | 111→111 | 111→111 | 111→111 |
| 9 | 197→197 | 111→111 | 111→111 | 111→111 | 111→111 |
| 10 | 197→197 | 111→111 | 111→111 | 111→111 | 111→111 |
| 11 | 197→197 | 111→111 | 111→111 | 111→111 | 111→111 |
| 12 | 197→197 | 111→1 | 111→1 | 111→1 | 111→1 |

예를 들어 Tiny의 Block 12 **입력은 111개이고 출력은 CLS 1개**다. “마지막 block이 처음부터 CLS 하나만 받는다”는 뜻이 아니다.

Tiny에서 후반 token 수가 111로 유지되는 이유는 다음 block이 필요한 token 집합을 현재 block의 최소값으로 가져오기 때문이다. 그 최소 집합만으로 현재의 다음-block 오차가 0.02 이하면 더 추가하지 않는다. 이는 top-down 탐색과 국소 오차 기준의 결과이며, layer마다 균등하게 줄어드는 pyramid를 강제하지 않은 결과다. Tiny30/Tiny50의 **개수**가 같아도 유지 ID와 학습 가중치가 같다는 뜻은 아니다.

## 8. 실제 compact forward와 최종 fine-tuning

관련 코드: [slim_block.py](../patch_slimming/psvit/slim_block.py), [compact.py](../patch_slimming/psvit/compact.py), [finetune.py](../patch_slimming/scripts/finetune.py).

### 8.1 Shape가 실제로 줄어드는 rectangular attention

한 block의 입력 token 수를 N_in, 출력으로 남길 수를 N_out이라고 하자.

```text
입력 X                         [B,N_in,D]
Norm1(X)                       [B,N_in,D]
Q: 선택한 token만 projection    [B,H,N_out,d_head]
K,V: 입력 token 전체 projection [B,H,N_in,d_head]
softmax(QKᵀ / sqrt(d_head))      [B,H,N_out,N_in]
attention @ V                  [B,H,N_out,d_head]
projection / 선택 residual     [B,N_out,D]
MLP hidden                     [B,N_out,M_reduced]
block 출력                     [B,N_out,D]
다음 block 입력                 [B,N_out,D]
```

이 block에서 출력으로 남기지 않는 patch도 **현재 block의 K/V에는 포함**된다. 선택한 Q/CLS가 마지막으로 그 정보를 참고할 수 있고, block 출력부터 해당 token row가 사라진다. 모든 Q/K/V를 block 시작부터 N_out으로 줄이는 방식과 다르다.

Q/K/V weight는 기존 fused qkv weight를 view로 논리 분할하여 재사용한다. 별도 attention weight 세트를 새로 학습하여 parameter를 늘리는 것이 아니다. D와 head 수는 유지되며 FFN width는 이미 reduced된 값이다.

### 8.2 학습의 index_select와 배포의 고정 선택행렬

GPU 검색·compact fine-tuning·최종 FP32 정확도 평가는 고정 ID로 `index_select`를 사용한다. ONNX export와 이번 서버 compact 메모리 프로파일은 `matmul` 모드를 사용한다.

N_in=5에서 local ID [0,2,4]를 남기는 예시는 다음과 같다.

```text
P = [[1,0,0,0,0],       X = [x0, x1, x2, x3, x4]ᵀ
     [0,0,1,0,0],
     [0,0,0,0,1]]

P @ X = [x0, x2, x4]ᵀ
P shape [3,5], X shape [5,D], result shape [3,D]
```

즉 **0을 곱한 row 5개를 그대로 보관하는 masking이 아니라, 결과 matrix의 row 수가 3개로 줄어드는 선택**이다. P는 학습 parameter가 아니며 모든 이미지에서 같은 상수다.

다만 이를 dense MatMul로 실행하면 선택 연산의 연산량/상수 저장량이 추가된다. 0/1이 수학적으로 단순하다는 사실만으로 compiler가 반드시 무료 인덱싱으로 최적화하거나, 양자화 경계에서도 완전히 동일한 결과를 낸다고 보장되지는 않는다.

### 8.3 연산량을 이해하는 근사식

Batch 1에서 주요 matrix 연산의 MAC은 다음과 같다. Norm/GELU/softmax/bias 및 runtime overhead는 제외한다.

```text
Q/K/V + attention output projection: 2 × (N_in + N_out) × D²
QKᵀ + attention@V:                 2 × N_in × N_out × D
FFN:                               2 × N_out × D × M_reduced
두 선택을 실제 dense MatMul로 실행: 최대 2 × N_in × N_out × D 추가
```

Identity 선택과 CLS-only 선택은 일반 선택 MatMul을 생략하므로 마지막 식을 모든 block에 무조건 더하면 안 된다. 실제 최종 보고서의 GFLOPs는 이 근사식을 손으로 합한 값이 아니라 **export된 ONNX의 Conv/MatMul/Gemm shape를 분석한 값**이다.

### 8.4 Patch fine-tuning recipe

검색 완료 가중치를 불러오고 **고정된 compact token 구조가 실제로 적용된 상태**로 전체 모델을 30epoch 학습한다. Dense 197-token 모델을 학습한 뒤 마지막에 mask만 붙이는 것이 아니다.

| 항목 | 최종 실행값 |
|---|---|
| 데이터 | ImageNet train 전체 |
| Epoch | 30 |
| Loss | Label-smoothed CE, smoothing 0.1 |
| Optimizer | AdamW |
| LR / weight decay | 2e-5 / 0.05 |
| Warmup / cosine | 3epoch / 이후 27epoch |
| 최소 LR / grad clip | 1e-6 / 1.0 |
| AMP | 사용 |
| 최종 checkpoint | validation Top-1 best |
| 추가 logit KD | 없음 |
| EMA | 없음 |
| 메모리 패널티 | 이 단계에서는 전달하지 않음 |

Block reconstruction은 feature MSE 학습이고, 마지막 compact 전체 학습은 CE다. “최종 학습까지 관계형 KD를 했다”거나 “patch fine-tuning에도 structural KD가 그대로 들어간다”는 설명은 맞지 않는다.

| 실험 | Epoch 1 val Top-1 | Epoch 30 val Top-1 | 선택된 best epoch | Best AMP val Top-1 | 최종 FP32 Top-1 |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | 66.242 | 67.900 | 28 | 67.910 | 67.902 |
| Tiny50 | 56.498 | 59.204 | 27 | 59.250 | 59.252 |
| Small30 | 74.370 | 77.938 | 29 | 77.948 | 77.970 |
| Small50 | 67.100 | 70.736 | 29 | 70.750 | 70.736 |

Epoch 1에서 best까지의 상승은 학습 중 회복을 보여주지만, **검색 직후/최종 fine-tuning 0epoch의 전체 validation 정확도는 이번 release에 독립 측정으로 보존되어 있지 않다.** 따라서 위 표로 pruning 직후 대비 정확히 몇 %p 회복했다고 계산하지 않는다.

## 9. 서버 FP32 최종 정확도

관련 코드: [evaluate.py](../scripts/evaluation/evaluate.py). Batch 128, float32, AMP off, ImageNet validation 50,000장이다. 아래 loss는 최종 평가 CE이며 train loss와 같지 않다.

| 실험 | 단계 | Parameters | Top-1 % | Top-5 % | 평가 loss |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 5,717,416 | 74.660 | 92.494 | 0.979368 |
| Tiny30 | pruned | 5,717,416 | 72.212 | 91.168 | 1.092819 |
| Tiny30 | reduced | 4,003,396 | 72.212 | 91.168 | 1.092819 |
| Tiny30 | patch_slimming | 4,003,396 | 67.902 | 88.130 | 1.399133 |
| Tiny50 | baseline | 5,717,416 | 74.660 | 92.494 | 0.979368 |
| Tiny50 | pruned | 5,717,416 | 63.230 | 85.832 | 1.499599 |
| Tiny50 | reduced | 2,862,256 | 63.230 | 85.832 | 1.499599 |
| Tiny50 | patch_slimming | 2,862,256 | 59.252 | 82.270 | 1.806516 |
| Small30 | baseline | 22,050,664 | 80.502 | 95.862 | 0.700559 |
| Small30 | pruned | 22,050,664 | 79.624 | 95.162 | 0.761775 |
| Small30 | reduced | 15,434,188 | 79.624 | 95.162 | 0.761775 |
| Small30 | patch_slimming | 15,434,188 | 77.970 | 94.190 | 0.934393 |
| Small50 | baseline | 22,050,664 | 80.502 | 95.862 | 0.700559 |
| Small50 | pruned | 22,050,664 | 74.002 | 92.234 | 1.005421 |
| Small50 | reduced | 11,023,204 | 74.002 | 92.234 | 1.005421 |
| Small50 | patch_slimming | 11,023,204 | 70.736 | 90.124 | 1.252564 |

단계별 Top-1 차이는 다음과 같다. 음수는 정확도 감소다.

| 실험 | Baseline→Reduced %p | Reduced→Patch %p | Baseline→Patch %p |
| --- | --- | --- | --- |
| Tiny30 | -2.448 | -4.310 | -6.758 |
| Tiny50 | -11.430 | -3.978 | -15.408 |
| Small30 | -0.878 | -1.654 | -2.532 |
| Small50 | -6.500 | -3.266 | -9.766 |

이 표는 **PyTorch 모델 정확도**다. ONNX 전체 validation 50,000장의 Top-1/5를 별도로 측정한 표라고 표시하면 안 된다. ONNX는 다음 장의 random-input parity로 검증했고, NPU는 MXQ로 별도 전체 평가했다.

## 10. 서버 메모리·activation 측정

관련 코드: [profile_reduced_memory.py](../scripts/profiling/profile_reduced_memory.py), [activation.py](../scripts/profiling/activation.py).

### 10.1 무엇을 어떻게 측정했는가

| 지표 | 계산/관측 방법 | 뜻하지 않는 것 |
|---|---|---|
| FP32 weight bytes | parameter별 numel × element_size | 전체 checkpoint/추론 메모리 아님 |
| Registered buffer bytes | 등록 buffer의 bytes 합 | 일반 attribute Tensor까지 모두 합한 값 아님 |
| Resident allocated | warmup 후 torch.cuda.memory_allocated | GPU 전체 사용량 아님 |
| Peak allocated | 측정 구간 max_memory_allocated | 50회 평균 아님 |
| Peak reserved | allocator가 확보한 최대 pool | 실사용 live tensor와 동일하지 않음 |
| Transient peak | peak allocated − warmup 후 resident | 순수 activation만 완벽히 분리한 값 아님 |
| Module output sum | 별도 1회 forward의 leaf module output bytes 합 | 동시에 살아 있는 activation peak 아님 |
| Block별 주요 tensor bytes | shape × dtype bytes로 계산 | allocator에서 tensor별 peak를 직접 관측한 값 아님 |
| Mean latency | 동기화된 50회 loop 총시간 / 50 | NPU latency와 동일 조건 속도 비교 아님 |

메모리/latency를 먼저 측정한 뒤 별도 forward에 hook을 붙인다. Hook은 실제 출력 tensor를 보관하지 않고 metadata만 모으므로 hook 때문에 peak 측정 구간이 변하는 것을 피한다.

`module_output_sum_bytes`는 norm/linear/activation 등 leaf module마다 생성한 출력들을 더한다. 같은 저장소나 연속 연산의 결과가 여러 번 계수될 수 있고, 함수형 attention MatMul 같은 연산은 leaf-module hook에 그대로 잡히지 않을 수 있다. 따라서 실제 activation peak, 총 메모리 traffic, 메모리 allocator peak 어느 것과도 같지 않다.

### 10.2 이번 compact 메모리 측정 경로의 세부사항

Baseline/reduced는 일반 timm 모델로, patch 모델은 **`CompactPSViT(select_mode="matmul")` + `bake_for_export()`**로 측정했다. Accuracy/학습의 `index_select` 경로와 구분해야 한다.

`bake_selection()`은 선택행렬을 `_sel_mat_q`와 `_sel_mat_r`에 같은 Tensor로 보관한다. 이 Tensor는 nn.Parameter도 registered buffer도 아니므로 parameter/buffer 합계에 완전히 드러나지 않지만, GPU에 존재하면 실제 allocated 메모리에는 포함된다. 현재 bake는 모든 block에 행렬을 준비하므로 forward에서 identity/CLS 선택을 생략해도 미리 만들어 둔 Tensor가 남을 수 있다. ONNX에서는 사용하지 않는 선택 연산을 남기지 않는 것과 구분해야 한다.

Baseline/reduced의 fused attention 가능 경로와 compact의 명시적 rectangular attention, 선택 tensor 및 임시 tensor도 다르다. 이것들이 peak/latency 증가에 기여할 수 있지만, 각 항목의 기여를 분리한 ablation은 이번 기록에 없다.

### 10.3 실제 FP32 batch-1 결과

모든 값은 같은 서버 모델별 측정 파일에서 가져왔다. Random input, warmup 10회, 측정 50회다.

| 실험 | 단계 | Weight MiB | Resident MiB | Peak allocated MiB | Peak reserved MiB | Transient MiB | Module output sum MiB | Mean ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 21.8102 | 30.5313 | 32.8311 | 48.0000 | 2.2998 | 56.2758 | 7.6311 |
| Tiny30 | reduced | 15.2717 | 23.9966 | 26.1523 | 40.0000 | 2.1558 | 42.8932 | 8.6098 |
| Tiny30 | patch_slimming | 15.2717 | 24.6250 | 27.3125 | 40.0000 | 2.6875 | 19.3636 | 23.7980 |
| Tiny50 | baseline | 21.8102 | 30.5313 | 32.8311 | 48.0000 | 2.2998 | 56.2758 | 11.0891 |
| Tiny50 | reduced | 10.9186 | 19.6460 | 21.8018 | 34.0000 | 2.1558 | 33.9834 | 7.7578 |
| Tiny50 | patch_slimming | 10.9186 | 20.2744 | 22.8862 | 36.0000 | 2.6118 | 14.5954 | 20.4542 |
| Small30 | baseline | 84.1166 | 93.1284 | 97.1577 | 112.0000 | 4.0293 | 112.5478 | 9.6904 |
| Small30 | reduced | 58.8768 | 68.0991 | 71.5552 | 80.0000 | 3.4561 | 86.6843 | 8.8645 |
| Small30 | patch_slimming | 58.8768 | 69.0264 | 74.0703 | 82.0000 | 5.0439 | 46.8540 | 22.1400 |
| Small50 | baseline | 84.1166 | 93.1284 | 97.1577 | 112.0000 | 4.0293 | 112.5478 | 8.7894 |
| Small50 | reduced | 42.0502 | 51.0635 | 54.3721 | 62.0000 | 3.3086 | 69.4420 | 9.9670 |
| Small50 | patch_slimming | 42.0502 | 51.7900 | 56.6802 | 64.0000 | 4.8901 | 32.6188 | 22.1218 |

Patch Slimming의 모듈 출력 합은 reduced 대비 크게 줄었지만 CUDA peak와 측정 latency는 증가했다. “토큰이 줄면 모든 backend에서 peak/속도가 반드시 개선된다”는 결론은 성립하지 않는다. 공유 서버에서 동시에 다른 작업을 수행했으므로 latency는 자원 경쟁 영향을 받을 수 있다.

CUDA allocator peak는 **해당 프로세스가 PyTorch로 할당한 메모리**이며 다른 프로세스 사용량을 단순 합산하지 않는다. 하지만 별도 프로세스 격리가 GPU 전체를 독점한다는 뜻은 아니므로 available memory, OOM 가능성, 속도에는 다른 작업이 영향을 줄 수 있다.

### 10.4 Patch/token와 activation의 용량식

- Block 입력 token tensor: `B × N_in × D × 4 bytes`. 영상 patch의 원본 RGB 픽셀 파일 크기가 아니라 embedding tensor다.
- FFN hidden activation: `B × N_out × M_l × 4 bytes`.
- Attention score 또는 probability 한 개: `B × H × N_out × N_in × 4 bytes`.
- 선택행렬 P 한 개: `N_out × N_in × 4 bytes`. 학습 weight와 다르다.

이 네 종류를 모두 “패치 메모리”라고 부르면 혼동된다. Table의 tensor bytes는 shape 기반 용량이고 allocator의 live peak를 합성한 값이 아니다. 다음은 patch 모델의 block별 대표 용량이다. Attention 열은 scores/probabilities 중 **한 tensor**의 용량이며 두 개가 동시에 언제 살아 있는지까지 합산한 값은 아니다.

#### Tiny30: block별 tensor 용량 (MiB)

| Block | N_in | N_out | FFN width | 입력 token | 출력 token | FFN hidden | Attention 한 개 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 197 | 121 | 462 | 0.144287 | 0.088623 | 0.213249 | 0.272793 |
| 2 | 121 | 121 | 470 | 0.088623 | 0.088623 | 0.216942 | 0.167553 |
| 3 | 121 | 121 | 354 | 0.088623 | 0.088623 | 0.163399 | 0.167553 |
| 4 | 121 | 121 | 352 | 0.088623 | 0.088623 | 0.162476 | 0.167553 |
| 5 | 121 | 121 | 357 | 0.088623 | 0.088623 | 0.164783 | 0.167553 |
| 6 | 121 | 111 | 374 | 0.088623 | 0.081299 | 0.158363 | 0.153706 |
| 7 | 111 | 111 | 359 | 0.081299 | 0.081299 | 0.152012 | 0.141003 |
| 8 | 111 | 111 | 337 | 0.081299 | 0.081299 | 0.142696 | 0.141003 |
| 9 | 111 | 111 | 422 | 0.081299 | 0.081299 | 0.178688 | 0.141003 |
| 10 | 111 | 111 | 414 | 0.081299 | 0.081299 | 0.175301 | 0.141003 |
| 11 | 111 | 111 | 354 | 0.081299 | 0.081299 | 0.149895 | 0.141003 |
| 12 | 111 | 1 | 509 | 0.081299 | 0.000732 | 0.001942 | 0.001270 |

#### Tiny50: block별 tensor 용량 (MiB)

| Block | N_in | N_out | FFN width | 입력 token | 출력 token | FFN hidden | Attention 한 개 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 197 | 121 | 298 | 0.144287 | 0.088623 | 0.137550 | 0.272793 |
| 2 | 121 | 121 | 181 | 0.088623 | 0.088623 | 0.083546 | 0.167553 |
| 3 | 121 | 121 | 80 | 0.088623 | 0.088623 | 0.036926 | 0.167553 |
| 4 | 121 | 121 | 117 | 0.088623 | 0.088623 | 0.054005 | 0.167553 |
| 5 | 121 | 121 | 121 | 0.088623 | 0.088623 | 0.055851 | 0.167553 |
| 6 | 121 | 111 | 147 | 0.088623 | 0.081299 | 0.062244 | 0.153706 |
| 7 | 111 | 111 | 133 | 0.081299 | 0.081299 | 0.056316 | 0.141003 |
| 8 | 111 | 111 | 109 | 0.081299 | 0.081299 | 0.046154 | 0.141003 |
| 9 | 111 | 111 | 85 | 0.081299 | 0.081299 | 0.035992 | 0.141003 |
| 10 | 111 | 111 | 104 | 0.081299 | 0.081299 | 0.044037 | 0.141003 |
| 11 | 111 | 111 | 175 | 0.081299 | 0.081299 | 0.074100 | 0.141003 |
| 12 | 111 | 1 | 250 | 0.081299 | 0.000732 | 0.000954 | 0.001270 |

#### Small30: block별 tensor 용량 (MiB)

| Block | N_in | N_out | FFN width | 입력 token | 출력 token | FFN hidden | Attention 한 개 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 197 | 197 | 707 | 0.288574 | 0.288574 | 0.531307 | 0.888268 |
| 2 | 197 | 197 | 740 | 0.288574 | 0.288574 | 0.556107 | 0.888268 |
| 3 | 197 | 197 | 699 | 0.288574 | 0.288574 | 0.525295 | 0.888268 |
| 4 | 197 | 141 | 708 | 0.288574 | 0.206543 | 0.380814 | 0.635765 |
| 5 | 141 | 111 | 645 | 0.206543 | 0.162598 | 0.273113 | 0.358223 |
| 6 | 111 | 111 | 694 | 0.162598 | 0.162598 | 0.293861 | 0.282005 |
| 7 | 111 | 111 | 665 | 0.162598 | 0.162598 | 0.281582 | 0.282005 |
| 8 | 111 | 111 | 659 | 0.162598 | 0.162598 | 0.279041 | 0.282005 |
| 9 | 111 | 111 | 806 | 0.162598 | 0.162598 | 0.341286 | 0.282005 |
| 10 | 111 | 111 | 1054 | 0.162598 | 0.162598 | 0.446297 | 0.282005 |
| 11 | 111 | 111 | 1348 | 0.162598 | 0.162598 | 0.570786 | 0.282005 |
| 12 | 111 | 1 | 1103 | 0.162598 | 0.001465 | 0.004208 | 0.002541 |

#### Small50: block별 tensor 용량 (MiB)

| Block | N_in | N_out | FFN width | 입력 token | 출력 token | FFN hidden | Attention 한 개 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 197 | 171 | 321 | 0.288574 | 0.250488 | 0.209393 | 0.771034 |
| 2 | 171 | 151 | 240 | 0.250488 | 0.221191 | 0.138245 | 0.590996 |
| 3 | 151 | 131 | 253 | 0.221191 | 0.191895 | 0.126431 | 0.452751 |
| 4 | 131 | 121 | 218 | 0.191895 | 0.177246 | 0.100624 | 0.362801 |
| 5 | 121 | 111 | 236 | 0.177246 | 0.162598 | 0.099930 | 0.307411 |
| 6 | 111 | 111 | 256 | 0.162598 | 0.162598 | 0.108398 | 0.282005 |
| 7 | 111 | 111 | 229 | 0.162598 | 0.162598 | 0.096966 | 0.282005 |
| 8 | 111 | 111 | 211 | 0.162598 | 0.162598 | 0.089344 | 0.282005 |
| 9 | 111 | 111 | 336 | 0.162598 | 0.162598 | 0.142273 | 0.282005 |
| 10 | 111 | 111 | 491 | 0.162598 | 0.162598 | 0.207905 | 0.282005 |
| 11 | 111 | 111 | 1126 | 0.162598 | 0.162598 | 0.476784 | 0.282005 |
| 12 | 111 | 1 | 175 | 0.162598 | 0.001465 | 0.000668 | 0.002541 |

## 11. ONNX 생성과 NPU 대응 최적화

관련 코드: [export_final_onnx.py](../scripts/export_final_onnx.py), [slim_block.py](../patch_slimming/psvit/slim_block.py), [compact.py](../patch_slimming/psvit/compact.py).

### 11.1 Export 계약

| 항목 | 값 |
|---|---|
| 모델 종류 | baseline / reduced / patch_slimming, 각 4개 |
| Weight/연산 기준 | FP32 |
| 입력 | `input: [1,224,224,3]`, NHWC |
| 출력 | `output: [1,1000]`, logits |
| Batch / 해상도 | 1 / 224×224 고정 |
| Opset | 17 |
| RGB 정규화 | 그래프 밖에서 mean/std 0.5 |
| 내부 이미지 layout | 필요 시 NHWC→NCHW transpose |
| Dynamic token 수 | 사용하지 않음 |
| 검증 | ONNX checker + PyTorch/ORT random-input parity |

### 11.2 일반 PyTorch 구현을 어떻게 정적으로 바꾸었는가

1. **고정 선택행렬:** JSON의 global ID를 해당 block의 local ID로 바꾸고, Q 입력과 residual 선택을 고정 0/1 P의 MatMul로 표현한다. GPU의 `index_select`를 그대로 export하여 Gather를 남기지 않는다.
2. **Identity 선택 제거:** 입력과 출력 ID가 동일하면 P=I를 곱하지 않고 tensor를 그대로 전달한다. 불필요한 연산과 양자화 경계 생성을 피한다.
3. **CLS-only 선택:** local ID 0 하나를 남길 때 `x[:, :1, :]`로 처리하여 static Slice를 사용한다.
4. **최종 CLS head:** CLS-only 출력은 `squeeze(1)`로 [1,D]로 바꾼다. Baseline/reduced의 CLS 추출도 정적 Slice/Squeeze 경로로 구성한다.
5. **고정 batch CLS 결합:** timm의 동적 CLS 확장 대신 batch 1 CLS tensor를 직접 concat하여 Equal/Where/Expand 같은 shape 분기를 피한다.
6. **명시적 attention:** Export에서 fused attention 대신 표현 가능한 matrix 연산/softmax 경로를 사용한다.
7. **금지 연산 검사:** Gather/GatherElements/GatherND/TopK/Equal/Where/Expand가 최종 그래프에 없는지 검사한다.

정적 구현이라고 원래부터 Gather가 없었던 것은 아니다. **고정 ID를 읽는 index_select도 ONNX에서는 Gather로 export될 수 있다.** TopK는 배포 시 필요 없지만, offline L2 pruning이나 importance ranking, 평가의 Top-k 계산에 사용하는 것까지 저장소에서 없앤다는 뜻은 아니다.

과거 변형의 성공/실패 수치를 최종 결과로 재사용하지 않고, 이번 최종 ONNX는 이 생성 경로에서 나온 12개로 고정했다. `npusafe`는 이 구조 검사에 붙인 이름이며 모든 compiler 버전에서 실행/정확도를 보장하는 인증명이 아니다.

### 11.3 FP32 parity와 파일 크기

입력 3개씩으로 original/eager 모델과 NPU wrapper, wrapper와 ONNX Runtime CPU 출력을 비교했다. 모든 모델의 **eager–wrapper 최대 오차는 0**, PyTorch–ORT 최대 절대오차는 1e-4보다 작았다. 이것은 FP32 graph conversion이 작은 수치 오차로 유지된다는 검사이며, INT8/MXQ의 정확도 검사가 아니다.

| 실험 | 단계 | ONNX MiB | 검증 입력 수 | Eager↔wrapper max abs | PyTorch↔ORT max abs | 서버→npu 해시 |
| --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 21.8890 | 3 | 0 | 1.168251e-5 | 일치 |
| Tiny30 | reduced | 15.3505 | 3 | 0 | 8.583069e-6 | 일치 |
| Tiny30 | patch_slimming | 15.6211 | 3 | 0 | 8.106232e-6 | 일치 |
| Tiny50 | baseline | 21.8890 | 3 | 0 | 1.168251e-5 | 일치 |
| Tiny50 | reduced | 10.9974 | 3 | 0 | 5.900860e-6 | 일치 |
| Tiny50 | patch_slimming | 11.2680 | 3 | 0 | 7.152557e-6 | 일치 |
| Small30 | baseline | 84.1954 | 3 | 0 | 6.437302e-6 | 일치 |
| Small30 | reduced | 58.9555 | 3 | 0 | 3.099442e-6 | 일치 |
| Small30 | patch_slimming | 59.2731 | 3 | 0 | 3.486872e-6 | 일치 |
| Small50 | baseline | 84.1954 | 3 | 0 | 6.437302e-6 | 일치 |
| Small50 | reduced | 42.1290 | 3 | 0 | 4.291534e-6 | 일치 |
| Small50 | patch_slimming | 42.9447 | 3 | 0 | 4.455447e-6 | 일치 |

Patch ONNX는 선택 상수·graph metadata가 추가되어 reduced ONNX보다 조금 크다. 이를 “학습 파라미터 수가 늘었다”고 해석하면 안 된다. 현재 parity 결과는 random input 3개이므로 **전체 ImageNet ONNX accuracy를 측정한 것으로 대체할 수 없다.**

## 12. Calibration과 MXQ 변환

관련 코드: [generate_hwc.py](../../npu/compile/calibration/generate_hwc.py), [compile_hwc.py](../../npu/compile/mxq/compile_hwc.py). 실행 역할은 [NPU RUNBOOK](../../npu/docs/RUNBOOK.md)에 분리되어 있다.

### 12.1 Calibration의 목적과 실제 입력 계약

ONNX FP32 activation의 범위를 대표 이미지로 관측하여 compiler가 저정밀 표현을 구성할 수 있도록 입력 tensor를 제공한다. 이 과정은 모델 weight를 CE로 다시 학습하는 fine-tuning/QAT와 다르다.

현재 calibration 생성기는 다음 순서다.

```text
이미지 경로 정렬 → 읽을 수 있는 앞 1,000장
  → cv2.imread (BGR) → RGB
  → short side 249, bicubic resize
  → 224×224 center crop
  → float32 / 255
  → (x − [0.5,0.5,0.5]) / [0.5,0.5,0.5]
  → HWC [224,224,3] .npy
  → 파일 경로 목록 calib_hwc.txt
```

개별 .npy에는 batch 축이 없다. 실제 ONNX input은 batch를 붙인 [1,224,224,3]이다. 파일을 uint8 RGB 그대로 저장하거나 CHW로 저장하면 계약이 달라진다.

현재 제출용 코드에서는 `preprocess_profile.json`, tensor 개수 1,000, shape, float32, finite 여부를 검사한다. 출처를 알 수 없는 예전 tensor에 profile 파일만 붙여 사용하는 방식은 허용하지 않는다. 이 검사는 향후 재생성 시 안전장치이며, **최종 MXQ 생성 당시의 calibration 이미지 경로·hash 전체가 보존되었다는 증거는 아니다.**

최종 MXQ의 원본 calibration 개별 이미지 목록/전체 compile 환경 lock/resolved quantizer config는 충분히 보존되지 않았다. 따라서 같은 1,000장 규모로 다시 컴파일해도 이번 MXQ와 bitwise 동일한 바이너리가 나온다고 보장할 수 없다.

### 12.2 “timm 전처리”의 구현 차이

서버 PyTorch는 timm/PIL 계열 transform과 `crop_pct=0.9`를 사용한다. 여기서는 보통 `int(224/0.9)=248`의 resize 기준이 적용된다. NPU calibration/runtime의 profile 이름도 `timm`이지만, 실제 코드는 **OpenCV bicubic, short side 249**, 비율 기반 크기 반올림을 사용한다.

평균/표준편차와 crop 크기는 같아도 resize 길이, 보간 구현, rounding이 다르다. 그러므로 서버–NPU 정확도 차이를 비교할 때 “입력 pixel이 완전히 같다”는 전제를 둘 수 없다. 최종 NPU끼리의 baseline/reduced/patch 비교는 같은 NPU 전처리와 샘플 목록을 사용했다.

### 12.3 확인된 compiler 호출

핵심 인자는 다음과 같다.

```python
mxq_compile(
    model=onnx_path,
    calib_data_path=calib_txt,
    save_path=mxq_path,
    backend="onnx",
    inference_scheme="global8",
    cpu_offload=True,
    config_preset="vision_transformer",
)
```

- `vision_transformer`: ViT용 compiler preset을 사용한다.
- `global8`: 최종 compile/runtime에서 맞춘 inference scheme이다. 이름을 보고 “모든 연산을 INT8로 강제한다”는 뜻으로 해석하면 안 된다.
- `cpu_offload=True`: compiler/runtime가 지원하지 않는 부분에 host 처리를 허용하는 옵션이다. 측정한 inference API latency를 순수 NPU kernel 시간으로만 설명하면 안 된다.
- Compile CUDA GPU는 calibration/양자화 계산용이다. 실행할 NPU 장치의 번호와 무관하다.

현재 코드에서 사용하지 않는 수동 histogram/KL 옵션을 과거 파일에서 발견하더라도, 그것이 최종 모델의 실제 quantizer 설정이라고 기록하지 않는다.

### 12.4 양자화 scale에 대해 확정할 수 있는 것과 없는 것

일반적인 affine 양자화를 설명하면 아래와 같다.

```text
q = clip(round(x / scale) + zero_point, q_min, q_max)
x_hat = scale × (q − zero_point)
```

이는 개념 설명이며 qbcompiler 내부의 모든 layer가 이 정확한 방법을 사용했다는 reverse-engineering 결과가 아니다. 최종 기록만으로는 다음을 확정할 수 없다.

- Activation scale의 실제 최적화 방식이 min/max인지, percentile/entropy/KL 등인지.
- Weight/activation별 symmetric/asymmetric, per-tensor/per-channel 적용 범위.
- 연산별 effective bit width와 mixed precision, scale 재사용/fusion 정책.
- 각 선택 MatMul 또는 rectangular attention 전후의 requantization 경계.
- 최종 컴파일에 사용한 정확한 compiler build와 resolved preset 전체.

따라서 “전부 uniform INT8, min/max scale로 실행했다”라고 쓰지 않는다. 정확한 답은 compiler가 내보낸 quantizer config/중간 tensor 또는 공급사 설명을 추가로 확보해야 한다.

### 12.5 고정 0/1 선택행렬도 왜 점검해야 하는가

P@X는 FP32 수학에서는 정확한 row 선택이지만, compiler가 P와 X를 어떤 dtype/scale로 취급하고 출력에 어떤 scale을 적용하는지는 별도 문제다. 특히 선택 연산의 입출력 activation scale, attention/LayerNorm/softmax 주변 범위, 연산 fusion 및 CPU/NPU 경계에 따라 오차가 달라질 수 있다.

반대로 최종 정확도 하락을 “0/1 행렬이 양자화되어서”라고 이미 확정할 근거도 없다. 최종 FP32 parity는 좋고 NPU accuracy 하락은 관측되지만, 어느 연산에서 오차가 시작되는지를 layer-wise로 분리한 최종 실험은 보존 결과에 없다. 장별 결과 비교와 가설을 분리한다.

## 13. 실제 Mobilint NPU 평가 프로토콜과 최종 결과

관련 코드: [eval_final_models.py](../../npu/benchmark/final/eval_final_models.py), [profile_memory.py](../../npu/benchmark/memory/profile_memory.py). 최종 원본은 [final_20261015](../../npu/results/final_benchmark/final_20261015/)에 있다.

### 13.1 실제 실행 환경

| 항목 | 기록된 값 |
|---|---|
| 실행 시작 | 2026-10-15 17:22:26 +09:00 |
| Host | Linux 5.15.185-tegra, aarch64 |
| Python | 3.10.12 |
| NumPy / OpenCV | 2.2.6 / 4.13.0 |
| qbruntime | v1.1.0 |
| mobilint-cli | v1.2.0 |
| Device | Aries(aries0), device-id 0 |
| Driver / firmware | Aries 1.11.0 / 1.2.5 |
| Inference scheme | global8 |
| Validation root | /media/airlab_compression/nvme_storage/imagenet_val |
| 최종 MXQ root | assets/mxq_final_20261006 |
| 최종 결과 root | results/final_benchmark/final_20261015 |

실행 당시 Git HEAD는 `4575bb8f73b54710261c99705634afa4b87daa3b`이고 worktree dirty가 기록되어 있다. 따라서 해당 commit만 checkout하면 실행 당시 코드를 100% 복원한다고 주장하지 않는다. 결과를 보존한 후의 정리 검증에서는 원본 모델·결과 68개 파일의 hash가 유지됨을 확인했다. **현재 제출용 안전장치 추가와 실제 측정 당시 코드 버전은 구분**한다.

### 13.2 정확도 평가

- ImageNet validation에서 1,000 class × class당 50장 = 50,000장.
- Label은 정렬된 class directory 순서와 정답 index를 맞춘다.
- Seed 42로 만든 같은 sample manifest를 모든 모델에 사용한다.
- 12개 × 50,000장 = **600,000회**, 성공 600,000회, 오류 0회.
- 모델마다 warmup 10회 후 측정한다.
- qbruntime output logits의 Top-1/Top-5를 정답과 비교한다.
- Sample manifest SHA-256: `a461792c70ea36081d54a238c95d595ca6c87f8e8c49a42238d3db5ea2029703`.

저장된 prediction 배열과 정답으로 Top-1을 재계산한 검증 결과는 12개 모두 집계와 일치한다. Top-5 전체 prediction은 보존되지 않아 Top-5는 측정 JSON/CSV 집계 일치까지 확인한 값이다. [기존 제출물 검증 기록](../../npu/docs/VALIDATION.md)

### 13.3 Latency/FPS 구간

`latency_ms`는 **`model.infer([tensor])` 호출 구간**이다. 이미지 파일을 읽고 전처리하는 시간은 제외하지만, runtime/전송/허용된 CPU fallback 등의 비용은 포함될 수 있다. NPU device kernel만 따로 계측한 수치는 아니다.

`inference_fps = 성공 건수 / inference 구간 총시간`이고, `end_to_end_fps`는 이미지 로딩·전처리를 포함한 평가 처리량이다. 두 FPS를 혼용하지 않는다. batch 1 순차 추론이며 높은 batch의 최대 throughput 벤치마크가 아니다.

### 13.4 전체 정확도·속도·NPU 메모리

NPU peak 열의 단위는 telemetry가 보고한 **MB**다. 아래 MXQ MiB는 실제 파일 bytes를 2²⁰으로 나눈 것이며 런타임 메모리가 아니다.

| 실험 | 단계 | Top-1 % | Top-5 % | Infer 평균 ms | Infer FPS | E2E FPS | NPU peak MB | MXQ MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 74.226 | 92.270 | 3.192 | 313.29 | 99.06 | 12 | 7.5621 |
| Tiny30 | reduced | 70.880 | 90.744 | 3.065 | 326.32 | 105.01 | 11 | 6.0055 |
| Tiny30 | patch_slimming | 53.528 | 78.384 | 2.408 | 415.24 | 120.83 | 7 | 5.7438 |
| Tiny50 | baseline | 74.226 | 92.270 | 3.170 | 315.41 | 102.79 | 12 | 7.5621 |
| Tiny50 | reduced | 62.442 | 85.314 | 2.976 | 336.03 | 106.31 | 10 | 4.8642 |
| Tiny50 | patch_slimming | 41.164 | 66.934 | 2.388 | 418.83 | 120.66 | 5 | 4.6027 |
| Small30 | baseline | 80.486 | 95.852 | 5.884 | 169.96 | 69.75 | 35 | 24.4799 |
| Small30 | reduced | 79.324 | 95.102 | 5.569 | 179.58 | 73.09 | 28 | 18.3080 |
| Small30 | patch_slimming | 70.094 | 90.184 | 4.585 | 218.09 | 83.62 | 23 | 18.1462 |
| Small50 | baseline | 80.486 | 95.852 | 5.821 | 171.79 | 71.59 | 35 | 24.4799 |
| Small50 | reduced | 73.616 | 92.124 | 5.310 | 188.31 | 76.95 | 24 | 13.9043 |
| Small50 | patch_slimming | 61.082 | 84.134 | 4.204 | 237.87 | 87.12 | 18 | 13.8297 |

### 13.5 Latency 분포

평균만으로 긴 tail을 가릴 수 있으므로 주요 percentile을 함께 기록한다. 단위 ms다.

| 실험 | 단계 | 평균 | 표준편차 | 최소 | p50 | p90 | p95 | p99 | 최대 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 3.1919 | 0.1698 | 2.9400 | 3.1736 | 3.3194 | 3.3751 | 3.5588 | 16.4075 |
| Tiny30 | reduced | 3.0645 | 0.1422 | 2.8122 | 3.0470 | 3.1889 | 3.2436 | 3.4126 | 15.5370 |
| Tiny30 | patch_slimming | 2.4083 | 0.1259 | 2.2143 | 2.3891 | 2.5184 | 2.5698 | 2.7317 | 9.4028 |
| Tiny50 | baseline | 3.1704 | 0.1500 | 2.9173 | 3.1533 | 3.2950 | 3.3462 | 3.5100 | 17.7645 |
| Tiny50 | reduced | 2.9759 | 0.1245 | 2.7382 | 2.9597 | 3.0953 | 3.1455 | 3.3108 | 11.5599 |
| Tiny50 | patch_slimming | 2.3876 | 0.1238 | 2.1670 | 2.3702 | 2.5044 | 2.5547 | 2.7269 | 9.6837 |
| Small30 | baseline | 5.8837 | 0.1742 | 5.5010 | 5.8735 | 6.0705 | 6.1326 | 6.3141 | 15.6329 |
| Small30 | reduced | 5.5686 | 0.1653 | 5.2012 | 5.5563 | 5.7515 | 5.8167 | 6.0071 | 12.2833 |
| Small30 | patch_slimming | 4.5853 | 0.1797 | 4.2941 | 4.5686 | 4.7435 | 4.8066 | 4.9785 | 25.3317 |
| Small50 | baseline | 5.8212 | 0.1616 | 5.4778 | 5.8092 | 6.0086 | 6.0737 | 6.2535 | 12.7326 |
| Small50 | reduced | 5.3104 | 0.1839 | 4.9916 | 5.2952 | 5.4674 | 5.5311 | 5.7075 | 17.0543 |
| Small50 | patch_slimming | 4.2040 | 0.1594 | 3.9002 | 4.1860 | 4.3590 | 4.4186 | 4.5866 | 11.7332 |

### 13.6 Host 메모리·전력·온도·활용률

Host memory는 psutil 기반 20 ms sampling, NPU telemetry는 500 ms sampling이다. NPU telemetry는 전체 평가 중 수집하므로 전처리와 추론 사이의 idle 구간도 영향을 준다. 별도 100% 연속 추론 시 전력과 같다고 해석하지 않는다.

Host `rss_peak_mb` 등의 원본 field 이름은 mb지만 코드의 환산은 2²⁰이므로 이 표에서는 MiB로 바로잡아 표시한다. RSS는 프로세스 resident memory, USS는 해당 프로세스의 고유 page 용량이다.

| 실험 | 단계 | Host RSS peak MiB | Host USS peak MiB | Board 평균 W | NPU 평균 W | Board 최대 W | NPU 평균 util % | 평균 온도 °C | NPU sample 수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 332.50 | 321.83 | 16.880 | 6.992 | 17.260 | 24.03 | 45.06 | 1011 |
| Tiny30 | reduced | 377.88 | 374.63 | 16.795 | 6.938 | 16.920 | 24.28 | 43.53 | 954 |
| Tiny30 | patch_slimming | 335.49 | 332.34 | 16.612 | 6.817 | 16.730 | 21.00 | 43.44 | 829 |
| Tiny50 | baseline | 351.92 | 339.55 | 16.858 | 6.986 | 17.050 | 24.96 | 43.52 | 974 |
| Tiny50 | reduced | 334.88 | 323.74 | 16.738 | 6.898 | 16.870 | 23.84 | 43.50 | 942 |
| Tiny50 | patch_slimming | 378.76 | 375.59 | 16.567 | 6.788 | 16.950 | 20.35 | 43.46 | 830 |
| Small30 | baseline | 337.38 | 341.86 | 17.432 | 7.384 | 17.970 | 34.69 | 44.38 | 1435 |
| Small30 | reduced | 346.36 | 338.89 | 17.181 | 7.193 | 17.570 | 34.46 | 43.85 | 1370 |
| Small30 | patch_slimming | 349.11 | 342.13 | 17.014 | 7.037 | 17.710 | 31.75 | 43.94 | 1197 |
| Small50 | baseline | 384.78 | 377.37 | 17.429 | 7.379 | 17.840 | 35.57 | 43.47 | 1398 |
| Small50 | reduced | 341.38 | 332.53 | 17.069 | 7.108 | 17.390 | 34.72 | 42.95 | 1301 |
| Small50 | patch_slimming | 338.34 | 327.03 | 16.876 | 6.948 | 17.420 | 29.81 | 42.68 | 1149 |

Board total과 NPU core power는 다른 범위다. DDR 등 세부 rail 값이 null인 항목은 측정된 0 W로 처리하지 않는다. Board 전력에서 서버 GPU 에너지나 시스템 전체 에너지를 유추하지 않는다. 측정 구간이 다른 평균 전력×inference latency를 곱해 정확한 image당 에너지라고 제시하지도 않는다.

### 13.7 독립 프로세스 메모리 재검증

전체 평가 프로세스의 이전 모델 상태/host allocator 누적을 구분하기 위해 모델마다 **새 프로세스**에서 별도 측정을 수행했다.

- 입력: seed 2022의 동일 FP32 NHWC [1,224,224,3] random tensor.
- 공통 입력 hash: `7b892cd12ad9069c25c421f7377f1b74b9f2a68b43ab3546097ba394ac6708c7`.
- Warmup 10회, latency 측정 100회.
- 이어서 memory sampling을 위해 약 10초 동안 반복 추론.
- Idle NPU memory와 active NPU memory를 따로 읽고 차이를 기록.
- 아래 반복수는 100회 latency loop가 아니라 **10초 telemetry loop의 추론 횟수**다.

이 측정은 ImageNet 정확도 평가가 아니다. 서버 profiler의 random 입력도 seed 2022지만 서버 최종 JSON의 input_sha256은 null이다. 서로 다른 generator/layout일 수 있으므로 **서버와 NPU가 바이트까지 같은 random 입력을 사용했다고 주장하지 않는다.**

| 실험 | 단계 | Idle peak MB | Active peak MB | Delta MB | Host RSS MiB | Host USS MiB | 100회 평균 ms | Telemetry 반복수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 0 | 12 | 12 | 141.617 | 122.215 | 3.1552 | 3186 |
| Tiny30 | reduced | 0 | 11 | 11 | 141.852 | 122.875 | 3.0427 | 3288 |
| Tiny30 | patch_slimming | 0 | 7 | 7 | 138.344 | 118.969 | 2.4152 | 4074 |
| Tiny50 | baseline | 0 | 12 | 12 | 137.859 | 118.875 | 3.1411 | 3185 |
| Tiny50 | reduced | 0 | 10 | 10 | 139.602 | 120.227 | 2.9659 | 3407 |
| Tiny50 | patch_slimming | 0 | 5 | 5 | 138.219 | 118.859 | 2.4657 | 4229 |
| Small30 | baseline | 0 | 35 | 35 | 139.727 | 120.352 | 5.6641 | 1752 |
| Small30 | reduced | 0 | 28 | 28 | 143.996 | 125.121 | 5.4344 | 1827 |
| Small30 | patch_slimming | 0 | 23 | 23 | 139.051 | 120.387 | 4.5845 | 2197 |
| Small50 | baseline | 0 | 35 | 35 | 137.723 | 118.363 | 5.7358 | 1756 |
| Small50 | reduced | 0 | 24 | 24 | 140.012 | 121.133 | 5.1973 | 1940 |
| Small50 | patch_slimming | 0 | 18 | 18 | 141.488 | 122.430 | 4.1801 | 2453 |

12개 모두 idle은 0 MB였고 active peak는 전체 ImageNet 평가에서의 NPU peak와 같았다. 이는 결과의 일관성을 뒷받침한다. 다만 NPU telemetry는 **장치 전체의 sampling된 사용량**이므로 다른 프로세스가 동시에 NPU를 쓰면 영향이 포함될 수 있다. 500 ms sampling은 짧은 transient peak를 놓칠 수 있다.

NPU 레코드의 `npu_activation_bytes`는 null이다. 이 메모리 결과를 “NPU 각 layer의 activation을 직접 측정했다”라고 표현하면 안 된다. Server의 shape 기반 activation 분석과 NPU 장치 전체 메모리 보고는 서로 보완하는 다른 지표다.

### 13.8 정적 ONNX 연산량

ONNX의 Conv/MatMul/Gemm에 대해 shape로 계산한 이론값이며 **1 MAC = 2 FLOPs**로 환산했다. Normalization, GELU, softmax, elementwise, compiler fusion/tiling 및 실제 하드웨어 실행 횟수는 포함하지 않는다. Selection MatMul이 남아 있으면 해당 주요 연산에도 계수된다.

| 실험 | 단계 | 주요 GFLOPs/image | GMAC/image | 같은 그룹 baseline 대비 GFLOPs 감소 |
| --- | --- | --- | --- | --- |
| Tiny30 | baseline | 2.507366 | 1.253683 | 0.00% |
| Tiny30 | reduced | 1.833797 | 0.916898 | 26.86% |
| Tiny30 | patch_slimming | 0.990278 | 0.495139 | 60.51% |
| Tiny50 | baseline | 2.507366 | 1.253683 | 0.00% |
| Tiny50 | reduced | 1.385355 | 0.692678 | 44.75% |
| Tiny50 | patch_slimming | 0.750282 | 0.375141 | 70.08% |
| Small30 | baseline | 9.197765 | 4.598882 | 0.00% |
| Small30 | reduced | 6.594263 | 3.297132 | 28.31% |
| Small30 | patch_slimming | 4.244585 | 2.122293 | 53.85% |
| Small50 | baseline | 9.197765 | 4.598882 | 0.00% |
| Small50 | reduced | 4.858595 | 2.429298 | 47.18% |
| Small50 | patch_slimming | 2.997665 | 1.498833 | 67.41% |

## 14. 서버–NPU 비교와 결과 해석

### 14.1 Cross-backend accuracy

“차이”는 NPU Top-1에서 서버 PyTorch FP32 Top-1을 뺀 값이다.

| 실험 | 단계 | PyTorch FP32 Top-1 | MXQ/NPU Top-1 | 차이 %p | PyTorch FP32 Top-5 | MXQ/NPU Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | baseline | 74.660 | 74.226 | -0.434 | 92.494 | 92.270 |
| Tiny30 | reduced | 72.212 | 70.880 | -1.332 | 91.168 | 90.744 |
| Tiny30 | patch_slimming | 67.902 | 53.528 | -14.374 | 88.130 | 78.384 |
| Tiny50 | baseline | 74.660 | 74.226 | -0.434 | 92.494 | 92.270 |
| Tiny50 | reduced | 63.230 | 62.442 | -0.788 | 85.832 | 85.314 |
| Tiny50 | patch_slimming | 59.252 | 41.164 | -18.088 | 82.270 | 66.934 |
| Small30 | baseline | 80.502 | 80.486 | -0.016 | 95.862 | 95.852 |
| Small30 | reduced | 79.624 | 79.324 | -0.300 | 95.162 | 95.102 |
| Small30 | patch_slimming | 77.970 | 70.094 | -7.876 | 94.190 | 90.184 |
| Small50 | baseline | 80.502 | 80.486 | -0.016 | 95.862 | 95.852 |
| Small50 | reduced | 74.002 | 73.616 | -0.386 | 92.234 | 92.124 |
| Small50 | patch_slimming | 70.736 | 61.082 | -9.654 | 90.124 | 84.134 |

Baseline과 reduced는 상대적으로 작은 cross-backend 차이를 보이지만, patch는 **Tiny30 −14.374, Tiny50 −18.088, Small30 −7.876, Small50 −9.654%p**의 차이가 있다. 이 차이는 최종 기록에서 남아 있는 중요한 제한이다.

하지만 이 수치는 “같은 입력 tensor에서 compiler만 바꾼” 실험이 아니다. 서버/NPU 전처리 구현 차이, backend 수치 차이와 양자화·fusion·runtime 등이 함께 포함된다. **정확한 원인이 quantizer인지 compiler lowering인지 아직 분리되지 않았다.**

### 14.2 NPU에서 실제 얻은 성능 교환관계

아래는 NPU 내부에서 같은 protocol로 비교한 결과다. Latency 감소는 `1−patch/base`, memory 감소는 보고된 peak 비율이다.

| 실험 | Baseline→Patch latency 감소 | Reduced→Patch latency 감소 | Baseline→Patch NPU 메모리 감소 | Reduced→Patch NPU 메모리 감소 | NPU Reduced→Patch Top-1 %p |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | 24.55% | 21.41% | 41.67% | 36.36% | -17.352 |
| Tiny50 | 24.69% | 19.77% | 58.33% | 50.00% | -21.278 |
| Small30 | 22.07% | 17.66% | 34.29% | 17.86% | -9.230 |
| Small50 | 27.78% | 20.83% | 48.57% | 25.00% | -12.534 |

즉 실제 NPU의 자원 효율 개선은 관측되었지만, 이번 patch 설정은 정확도 손실을 동시에 수반한다. “최종 patch 모델이 모든 기준에서 더 우수하다”는 결론은 맞지 않는다. 정확도 우선의 선택과 메모리/latency 우선의 선택을 구분해야 한다.

### 14.3 서로 다른 메모리 지표를 한 숫자로 합치지 않는다

| 비교 항목 | 서버 | NPU |
|---|---|---|
| Weight 표현 | FP32 PyTorch | MXQ 내부 precision 정책 |
| Input 평가 | random batch 1 | ImageNet 전체 또는 고정 random 별도 측정 |
| Peak 원천 | 프로세스 PyTorch CUDA allocator | 장치 전체 telemetry |
| 단위 | MiB | 도구 보고 MB |
| Sampling | allocator peak 통계 | 500 ms polling |
| Activation 상세 | hook 출력 합 + shape 분석 | per-layer activation 직접 측정 없음 |
| 다른 프로세스 | allocator 합계에는 직접 포함 안 됨 | 같은 NPU 사용량이 섞일 수 있음 |

서버에서는 patch CUDA peak가 reduced보다 증가했고 NPU에서는 감소했다. Backend별 graph 실행 방식, precision, memory planning 및 측정 범위가 다르므로 모순된 결과는 아니다. 다만 정확한 원인별 메모리 기여는 해당 backend에서 추가 계측해야 한다.

## 15. 재현 명령·W&B·산출물 추적

아래는 **다시 실행할 때의 명령**이며 이 문서를 만들면서 재학습/재컴파일한 기록이 아니다. 최종 보관 폴더를 새 실행 출력으로 사용하지 않는다. 데이터/장치 경로는 해당 서버에 맞게 바꾼다.

### 15.1 Server_Compression에서 파일과 구현 검증

```bash
cd /workspace/etri_iitp/JS/Server_Compression
git lfs pull
python scripts/package_release.py --verify
python scripts/verify_release_models.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python patch_slimming/tests/run_all.py
```

호환되는 Python/PyTorch/timm 환경이 필요하다. 첫 두 검사는 모델/파일 및 random-input parity이며 전체 ImageNet 정확도 재측정은 아니다.

### 15.2 서버 처음부터 재실험

`configs/final.yaml`을 별도 설정으로 복사하고 `output_root`, `run_prefix`, `project`, `data_path`, GPU를 새 실험용으로 바꾼다. Structural 30epoch/LR horizon 50, patch 30epoch 등의 설정을 변경하면 본 결과와는 다른 실험이다.

```bash
cp configs/final.yaml configs/reproduction.yaml
# reproduction.yaml의 경로, 새 실험명, GPU를 편집한 후 실행
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/launch_final.py --matrix configs/reproduction.yaml --detach
```

Launcher가 GPU별 budget → structural/reducing → patch → export → memory → evaluate를 이어서 수행한다. Shared server의 다른 작업을 종료하거나 드라이버/CUDA를 업데이트하는 명령은 포함하지 않는다.

### 15.3 최종 patch ONNX만 새 위치로 재생성

예시는 Tiny30이다. 나머지는 모델 폴더명을 바꾼다.

```bash
python scripts/export_final_onnx.py --kind patch --npu-safe \
  --arch output/release_final_20261006/tiny30/03_patch_slimming_paper/architecture.json \
  --weights output/release_final_20261006/tiny30/03_patch_slimming_paper/compact_finetuned_best.pt \
  --output output/local_reexport/tiny30__patch_slimming__nhwc_b1_npusafe.onnx
```

기존 배포 ONNX를 덮어쓰지 않고, 생성된 verified JSON과 해시·parity를 다시 확인한다. 실행 환경/ONNX exporter 버전이 다르면 파일 해시가 달라질 수 있으므로 기존 NPU 결과에 새 파일을 대응시키지 않는다.

### 15.4 서버 개별 FP32 메모리 재측정

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profiling/profile_reduced_memory.py \
  --checkpoint output/release_final_20261006/tiny30/02_reduced/reduced.pt \
  --device cuda:0 --batch-size 1 --warmups 10 --runs 50 --details \
  --output output/local_profile/tiny30_reduced.json

CUDA_VISIBLE_DEVICES=0 python scripts/profiling/profile_reduced_memory.py \
  --arch output/release_final_20261006/tiny30/03_patch_slimming_paper/architecture.json \
  --weights output/release_final_20261006/tiny30/03_patch_slimming_paper/compact_finetuned_best.pt \
  --device cuda:0 --batch-size 1 --warmups 10 --runs 50 --details \
  --output output/local_profile/tiny30_patch.json
```

Baseline은 checkpoint를 `00_budget/baseline.pt`로 바꾼다. 명령의 GPU 0은 예시이며 점유 상태를 확인하고 배정한다.

### 15.5 npu 컴파일 Docker: calibration 및 CUDA GPU 1 사용

이하 명령은 **npu 저장소 루트**에서 실행한다. SDK/Python 의존성이 준비되어 있어야 한다.

```bash
python3 compile/calibration/generate_hwc.py \
  --profile timm \
  --src-dir /data/imagenet_calibration \
  --output-dir assets/calib_hwc_timm \
  --n-images 1000

CUDA_VISIBLE_DEVICES=1 \
NPU_CALIB_DIR="$PWD/assets/calib_hwc_timm" \
NPU_ONNX_DIR="$PWD/assets/onnx" \
NPU_MXQ_DIR="$PWD/assets/mxq/rebuild_final" \
NPU_ONNX_FILTER="__nhwc_b1_npusafe.onnx" \
NPU_INFERENCE_SCHEME=global8 \
python3 compile/mxq/compile_hwc.py
```

Calibration 출력은 새 폴더를 사용한다. Compiler는 이미 같은 출력 파일이 있으면 skip하므로 실패한 partial 파일을 그대로 둔 채 재시도하지 말고 **새 rebuild 폴더**로 구분한다. `CUDA_VISIBLE_DEVICES=1`은 Docker에 노출된 CUDA 번호이므로 host GPU UUID와 대응을 확인한다.

새 compile은 기존 최종 MXQ와 별도 실험이다. 실제 평가한 `assets/mxq_final_20261006`을 덮어쓰지 않는다.

### 15.6 실제 NPU: 보존 모델 검증과 전체 재평가

```bash
python3 scripts/verify_release.py --onnx --predictions
mobilint-cli status

python3 benchmark/final/eval_final_models.py \
  --models-dir assets/mxq_final_20261006 \
  --onnx-dir assets/onnx \
  --val-dir /data/imagenet_val \
  --classes 1000 --images-per-class 50 \
  --preprocess timm --core-mode global8 --device-id 0 \
  --warmup-runs 10 --progress-every 1000 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/reproduction_full \
  --fail-fast
```

새로 컴파일한 모델은 `--models-dir assets/mxq/rebuild_final`로 바꾼다. 먼저 `--dry-run`으로 경로/파일/데이터를 확인하고, 별도 출력에 `--limit 1` smoke test를 할 수 있다. Smoke 성공은 accuracy 검증이 아니다.

### 15.7 NPU 독립 메모리 재측정

```bash
python3 benchmark/memory/profile_memory.py \
  --models-dir assets/mxq_final_20261006 \
  --core-mode global8 --device-id 0 \
  --warmups 10 --runs 100 --memory-seconds 10 \
  --memory-sample-ms 20 --npu-sample-ms 500 \
  --output-dir results/reproduction_memory
```

측정 중 같은 장치에서 다른 inference를 병행하지 않는다. 정확도 평가가 아니며 출력의 idle/active/delta와 host RSS를 함께 해석한다.

### 15.8 W&B run 대응

서버 실행 프로젝트: [vit-final-target-memory-20261006](https://wandb.ai/junsu000306/vit-final-target-memory-20261006). 아래 이름/ID는 로컬 설정과 로그에 기록된 매핑이다. 이 문서 작성 과정에서 원격 W&B 페이지의 현재 보존 상태까지 재검증한 것은 아니다.

| 단계 | Run name 패턴 | Run ID 패턴 |
|---|---|---|
| Structural | `<model>__global_l2_kd__target_memory__structural30` | `final-20261006-<model>-structural` |
| Patch search | `<model>__target_memory__paper_rel002__search` | `final-20261006-<model>-patch-search` |
| Patch fine-tuning | `<model>__target_memory__paper_rel002__finetune30` | `final-20261006-<model>-patch-finetune` |
| 서버 FP32 평가 | `<model>__<stage>__fp32_eval` | `final-20261006-<model>-<stage>-eval` |

Train/validation loss·accuracy, 검색 오차/토큰 수, 메모리 scalar 등의 경과를 기록한다. NPU 최종 성능의 근거는 npu 결과 파일이며, 서버 W&B의 FP32 평가와 동일한 run/metric이라고 취급하지 않는다.

### 15.9 배포 checkpoint와 재개 checkpoint의 차이

Release에는 모델별 baseline, target, structural terminal, reduced, searched student, compact best의 .pt 6개, 총 **24개**와 ONNX **12개**가 있다. Release packaging에서 structural/patch checkpoint의 optimizer 등 재개용 상태를 제외했지만 model/model_ema tensor는 원본과 같은지 검사한다.

따라서 학습용 원본 checkpoint와 release checkpoint의 파일 hash는 다를 수 있다. 텐서를 바꿔 다시 학습했다는 뜻은 아니다. 추론/재내보내기는 release로 가능하지만 동일 optimizer/scheduler 상태에서 학습을 재개하려면 로컬 원본 checkpoint가 필요하다.

Server manifest: [MANIFEST.json](../output/release_final_20261006/MANIFEST.json).

NPU manifest: [RELEASE_MANIFEST.json](../../npu/assets/RELEASE_MANIFEST.json).

최종 파일 목록/내용 검증과 실제 모델 성능 검증을 구분한다.

### 15.10 핵심 코드 지도

| 역할 | 코드 |
|---|---|
| 모델/GPU/학습 설정 | [configs/final.yaml](../configs/final.yaml) |
| 전체 pipeline / supervisor | [run_pipeline.py](../scripts/run_pipeline.py), [launch_final.py](../scripts/launch_final.py) |
| 사전 target / M_max | [prepare_budget.py](../scripts/profiling/prepare_budget.py) |
| L2 ranking / mask | [vit_pruning.py](../pruning/vit_pruning.py) |
| CE/KL/optimizer/mask/EMA 순서 | [engine.py](../pruning/engine.py), [train.py](../pruning/train.py) |
| 측정 기반 memory scalar | [memory_budget.py](../pruning/memory_budget.py) |
| FFN physical slicing | [vit_reducing.py](../pruning/vit_reducing.py) |
| Patch significance DP | [scoring.py](../patch_slimming/psvit/scoring.py) |
| 두 block reconstruction | [reconstruct.py](../patch_slimming/psvit/reconstruct.py) |
| Top-down token 검색 | [search.py](../patch_slimming/psvit/search.py) |
| Rectangular attention / 고정 선택 | [slim_block.py](../patch_slimming/psvit/slim_block.py) |
| 전체 compact graph | [compact.py](../patch_slimming/psvit/compact.py) |
| Compact 전체 CE 학습 | [finetune.py](../patch_slimming/scripts/finetune.py) |
| ONNX export/parity | [export_final_onnx.py](../scripts/export_final_onnx.py) |
| 서버 FP32 accuracy | [evaluate.py](../scripts/evaluation/evaluate.py) |
| 서버 메모리/activation | [profile_reduced_memory.py](../scripts/profiling/profile_reduced_memory.py), [activation.py](../scripts/profiling/activation.py) |
| HWC calibration | [generate_hwc.py](../../npu/compile/calibration/generate_hwc.py) |
| MXQ compiler 호출 | [compile_hwc.py](../../npu/compile/mxq/compile_hwc.py) |
| NPU 전체 평가 | [eval_final_models.py](../../npu/benchmark/final/eval_final_models.py) |
| NPU 별도 memory worker | [profile_memory.py](../../npu/benchmark/memory/profile_memory.py) |
| NPU 제출물 해시/예측 검증 | [verify_release.py](../../npu/scripts/verify_release.py) |

## 16. 확인 범위와 후속 검증

### 16.1 현재 기록으로 말할 수 있는 것

- 목표 구조를 사전 생성하여 M_max를 정했고, 완료된 reduced 모델을 사전 준비물로 요구하지 않았다.
- L2 기준으로 FFN channel을 global 선택했고 CE+logit KL KD로 30epoch 학습했다.
- Reducing이 실제 matrix 크기를 줄였고 pruned/reduced의 최종 FP32 정확도가 일치한다.
- Patch는 별도 통계·block reconstruction·30epoch compact 학습을 거쳤으며 token 길이가 실제로 줄었다.
- ONNX 12개의 FP32 parity, 정적 입출력, 제한 연산 부재, repo 간 파일 전달 일치가 확인됐다.
- MXQ 12개는 실제 NPU에서 50,000장씩 오류 없이 평가됐다.
- NPU에서 patch latency/보고 메모리는 줄었지만 정확도 손실이 컸다.
- 독립 memory 재검증 결과가 통합 평가의 NPU peak와 일치한다.

### 16.2 현재 결과로 주장하면 안 되는 것

- 메모리 penalty가 gradient로 구조를 자동 최적화했거나, 정확도 개선의 원인이라는 주장.
- 모든 layer가 ε 이하를 만족했다는 주장(Small30 앞 3개 fallback).
- 논문의 dataset/metric/학습 조건을 모두 동일하게 재현했다는 주장.
- 서버 FP32 accuracy를 곧바로 ONNX 또는 NPU accuracy로 표기하는 것.
- 최종 MXQ의 모든 layer가 동일 INT8/min-max quantization이라는 단정.
- Patch NPU 손실의 원인이 선택 MatMul/특정 core mode라고 확정하는 것.
- 서버 CUDA peak와 NPU telemetry peak를 같은 단위/동일 의미로 직접 비교하는 것.
- Module output sum을 실제 동시에 살아 있는 activation peak라고 부르는 것.
- 다른 서버에서 calibration/NPU를 수행하지 않았다는 이유로 측정값을 추정치로 바꾸거나, 반대로 기록이 없는 실험을 수행했다고 만드는 것.

### 16.3 추가로 원인을 분리하려면

아래는 **미수행 후속 검증 제안**이며 이 문서의 최종 실험 결과에 포함되지 않는다.

1. 서버와 NPU에 동일한 전처리 결과 tensor를 전달하여 FP32 PyTorch → ORT → MXQ의 logits를 같은 이미지에서 비교한다.
2. 동일 tensor/이미지 순서로 ORT 전체 validation을 수행해 export 자체와 preprocessing 영향을 분리한다.
3. Compiler의 resolved quantizer config, precision/scale 및 가능하면 intermediate tensor를 확보한다.
4. 선택 MatMul, LayerNorm, rectangular attention 주변의 layer-wise 상대오차/cosine similarity를 비교하여 오차가 처음 커지는 지점을 찾는다.
5. Calibration subset을 명시적으로 저장하고 원본 이미지·생성 tensor hash를 보존하여 재컴파일한다.
6. 특정 연산 precision 변경이나 완화형 token schedule은 새 실험으로 구분하고, 같은 FP32/MXQ 평가 조건에서 개선 여부를 확인한다.

결론적으로 이 프로젝트는 **구조적 FFN 압축과 정적 token 압축을 실제 NPU 실행 가능한 산출물까지 연결하고, 정확도/자원 사용의 교환관계를 측정한 파이프라인**이다. 최종 모델의 정확도와 자원 효율을 따로 보고하며, 관측된 손실의 원인 미확정 부분도 함께 남긴다.

## 17. 상세 기록 부록

### 부록 A. 30epoch 전체 학습 기록

원본 epoch index는 0-based지만 표는 완료 epoch 1–30으로 변환했다. Structural train loss는 CE/KD/메모리 항의 총 loss이며, patch train loss는 CE다. 두 loss의 수치를 같은 목적함수로 직접 비교하지 않는다. Validation은 학습 중 AMP 평가이고, 최종 FP32 수치는 9장의 별도 평가표를 사용한다.

표의 `S`는 structural, `P`는 compact patch fine-tuning이다. Structural memory loss는 30epoch 전부 0이고, P에서는 메모리 항을 사용하지 않는다. Train accuracy는 증강 이미지의 training-mode 측정이라 deterministic validation accuracy보다 낮게 나올 수 있다.

#### Tiny30: epoch별 loss/accuracy

근거: [structural metrics](../output/release_final_20261006/tiny30/01_structural/metrics.jsonl), [patch metrics](../output/release_final_20261006/tiny30/03_patch_slimming_paper/metrics.jsonl).

| Epoch | S train loss | S train Top-1 | S val Top-1 | S val Top-5 | P train loss | P train Top-1 | P val Top-1 | P val Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.08934 | 5.226 | 6.056 | 17.312 | 3.34262 | 47.857 | 66.242 | 87.256 |
| 2 | 3.16751 | 27.870 | 28.484 | 55.194 | 3.19590 | 49.712 | 66.666 | 87.530 |
| 3 | 2.54306 | 39.684 | 49.462 | 76.098 | 3.13877 | 50.254 | 66.790 | 87.570 |
| 4 | 2.31275 | 44.656 | 59.102 | 83.342 | 3.11781 | 50.439 | 66.548 | 87.480 |
| 5 | 2.19538 | 47.389 | 63.646 | 86.270 | 3.09688 | 50.673 | 66.670 | 87.484 |
| 6 | 2.12546 | 49.115 | 66.038 | 87.782 | 3.07840 | 50.994 | 66.830 | 87.496 |
| 7 | 2.07509 | 50.404 | 67.618 | 88.698 | 3.06797 | 51.184 | 66.856 | 87.580 |
| 8 | 2.04058 | 51.364 | 68.568 | 89.258 | 3.05709 | 51.300 | 66.804 | 87.538 |
| 9 | 2.01473 | 52.070 | 69.346 | 89.606 | 3.04889 | 51.495 | 66.806 | 87.634 |
| 10 | 1.99688 | 52.573 | 69.804 | 89.774 | 3.03937 | 51.614 | 66.802 | 87.734 |
| 11 | 1.98124 | 53.037 | 70.186 | 89.986 | 3.02806 | 51.792 | 67.058 | 87.750 |
| 12 | 1.96618 | 53.535 | 70.412 | 90.170 | 3.02269 | 51.940 | 67.098 | 87.758 |
| 13 | 1.95612 | 53.762 | 70.640 | 90.294 | 3.01597 | 52.055 | 67.232 | 87.856 |
| 14 | 1.94475 | 54.120 | 70.786 | 90.424 | 3.01048 | 52.168 | 67.238 | 87.732 |
| 15 | 1.93664 | 54.460 | 70.964 | 90.488 | 3.00477 | 52.243 | 67.194 | 87.882 |
| 16 | 1.92913 | 54.635 | 71.154 | 90.578 | 2.99787 | 52.401 | 67.344 | 87.912 |
| 17 | 1.92292 | 54.840 | 71.242 | 90.636 | 2.99354 | 52.506 | 67.526 | 87.924 |
| 18 | 1.91628 | 55.067 | 71.432 | 90.662 | 2.98936 | 52.560 | 67.486 | 87.936 |
| 19 | 1.91014 | 55.227 | 71.528 | 90.686 | 2.98240 | 52.722 | 67.466 | 87.828 |
| 20 | 1.90570 | 55.380 | 71.634 | 90.728 | 2.97997 | 52.762 | 67.636 | 87.982 |
| 21 | 1.89926 | 55.581 | 71.758 | 90.834 | 2.97620 | 52.888 | 67.708 | 88.048 |
| 22 | 1.89327 | 55.816 | 71.814 | 90.852 | 2.97029 | 52.977 | 67.862 | 88.032 |
| 23 | 1.88829 | 55.991 | 71.836 | 90.950 | 2.96814 | 52.970 | 67.690 | 88.078 |
| 24 | 1.88420 | 56.105 | 71.882 | 91.004 | 2.96457 | 53.107 | 67.664 | 88.120 |
| 25 | 1.87937 | 56.197 | 71.974 | 91.014 | 2.96140 | 53.159 | 67.830 | 88.170 |
| 26 | 1.87556 | 56.342 | 72.080 | 91.098 | 2.95906 | 53.219 | 67.798 | 88.130 |
| 27 | 1.87125 | 56.508 | 72.074 | 91.104 | 2.95752 | 53.243 | 67.792 | 88.130 |
| 28 | 1.86735 | 56.644 | 72.124 | 91.136 | 2.95657 | 53.211 | 67.910 | 88.136 |
| 29 | 1.86295 | 56.730 | 72.146 | 91.160 | 2.95357 | 53.314 | 67.876 | 88.130 |
| 30 | 1.86028 | 56.842 | 72.216 | 91.170 | 2.95483 | 53.260 | 67.900 | 88.164 |

#### Tiny50: epoch별 loss/accuracy

근거: [structural metrics](../output/release_final_20261006/tiny50/01_structural/metrics.jsonl), [patch metrics](../output/release_final_20261006/tiny50/03_patch_slimming_paper/metrics.jsonl).

| Epoch | S train loss | S train Top-1 | S val Top-1 | S val Top-5 | P train loss | P train Top-1 | P val Top-1 | P val Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.18155 | 0.584 | 0.478 | 2.224 | 3.78160 | 38.152 | 56.498 | 80.530 |
| 2 | 4.56938 | 6.525 | 2.912 | 10.158 | 3.64264 | 39.978 | 57.312 | 81.142 |
| 3 | 3.82622 | 16.428 | 14.806 | 35.408 | 3.59081 | 40.614 | 57.410 | 81.180 |
| 4 | 3.39782 | 23.764 | 29.620 | 55.478 | 3.56564 | 40.926 | 57.466 | 81.260 |
| 5 | 3.12607 | 28.906 | 39.254 | 66.196 | 3.54482 | 41.253 | 57.564 | 81.390 |
| 6 | 2.94088 | 32.571 | 45.808 | 72.090 | 3.53313 | 41.410 | 57.808 | 81.306 |
| 7 | 2.80939 | 35.255 | 49.944 | 75.752 | 3.51815 | 41.691 | 57.816 | 81.470 |
| 8 | 2.72079 | 37.116 | 52.802 | 78.044 | 3.50912 | 41.834 | 58.110 | 81.476 |
| 9 | 2.65622 | 38.504 | 54.680 | 79.482 | 3.49652 | 42.068 | 58.070 | 81.488 |
| 10 | 2.60609 | 39.571 | 55.934 | 80.634 | 3.48616 | 42.293 | 58.260 | 81.670 |
| 11 | 2.56524 | 40.542 | 57.102 | 81.382 | 3.48048 | 42.317 | 58.248 | 81.746 |
| 12 | 2.52995 | 41.344 | 57.928 | 82.030 | 3.47172 | 42.502 | 58.366 | 81.780 |
| 13 | 2.50160 | 41.978 | 58.660 | 82.614 | 3.46712 | 42.604 | 58.500 | 81.770 |
| 14 | 2.47557 | 42.589 | 59.262 | 83.134 | 3.46103 | 42.712 | 58.608 | 81.842 |
| 15 | 2.45532 | 43.112 | 59.856 | 83.502 | 3.45592 | 42.842 | 58.504 | 81.880 |
| 16 | 2.43586 | 43.521 | 60.260 | 83.816 | 3.44960 | 42.942 | 58.562 | 81.926 |
| 17 | 2.41889 | 43.938 | 60.688 | 84.092 | 3.44476 | 43.008 | 58.698 | 82.020 |
| 18 | 2.40296 | 44.347 | 60.986 | 84.362 | 3.44191 | 43.051 | 58.738 | 81.946 |
| 19 | 2.38861 | 44.692 | 61.332 | 84.574 | 3.43443 | 43.188 | 58.816 | 82.070 |
| 20 | 2.37741 | 44.919 | 61.510 | 84.772 | 3.42920 | 43.352 | 58.976 | 82.066 |
| 21 | 2.36488 | 45.247 | 61.800 | 84.964 | 3.42581 | 43.355 | 58.916 | 82.072 |
| 22 | 2.35297 | 45.546 | 62.026 | 85.124 | 3.42211 | 43.443 | 59.036 | 82.166 |
| 23 | 2.34233 | 45.817 | 62.136 | 85.188 | 3.42088 | 43.477 | 59.076 | 82.188 |
| 24 | 2.33253 | 46.035 | 62.350 | 85.360 | 3.41740 | 43.518 | 59.194 | 82.274 |
| 25 | 2.32472 | 46.208 | 62.602 | 85.470 | 3.41268 | 43.678 | 59.174 | 82.252 |
| 26 | 2.31558 | 46.507 | 62.742 | 85.540 | 3.41390 | 43.599 | 59.142 | 82.232 |
| 27 | 2.30813 | 46.663 | 62.920 | 85.600 | 3.41392 | 43.606 | 59.250 | 82.256 |
| 28 | 2.30093 | 46.871 | 63.066 | 85.714 | 3.40828 | 43.773 | 59.184 | 82.310 |
| 29 | 2.29401 | 47.020 | 63.170 | 85.776 | 3.40722 | 43.732 | 59.218 | 82.272 |
| 30 | 2.28908 | 47.146 | 63.238 | 85.844 | 3.40628 | 43.779 | 59.204 | 82.338 |

#### Small30: epoch별 loss/accuracy

근거: [structural metrics](../output/release_final_20261006/small30/01_structural/metrics.jsonl), [patch metrics](../output/release_final_20261006/small30/03_patch_slimming_paper/metrics.jsonl).

| Epoch | S train loss | S train Top-1 | S val Top-1 | S val Top-5 | P train loss | P train Top-1 | P val Top-1 | P val Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.92448 | 13.559 | 21.436 | 44.798 | 2.44599 | 66.066 | 74.370 | 92.522 |
| 2 | 2.37141 | 43.675 | 57.906 | 82.238 | 2.30548 | 67.461 | 75.408 | 92.970 |
| 3 | 1.90097 | 55.480 | 68.610 | 89.488 | 2.26795 | 68.055 | 75.766 | 93.264 |
| 4 | 1.73607 | 59.943 | 72.628 | 91.792 | 2.26032 | 68.060 | 75.716 | 93.228 |
| 5 | 1.65678 | 62.148 | 74.628 | 92.838 | 2.24039 | 68.481 | 76.068 | 93.208 |
| 6 | 1.61638 | 63.392 | 75.820 | 93.440 | 2.22710 | 68.723 | 76.160 | 93.474 |
| 7 | 1.57673 | 64.562 | 76.598 | 93.780 | 2.21512 | 68.939 | 76.170 | 93.368 |
| 8 | 1.54914 | 65.482 | 77.044 | 93.988 | 2.20381 | 69.197 | 76.458 | 93.530 |
| 9 | 1.52922 | 66.141 | 77.448 | 94.150 | 2.19158 | 69.545 | 76.520 | 93.536 |
| 10 | 1.51375 | 66.675 | 77.850 | 94.338 | 2.18216 | 69.683 | 76.496 | 93.626 |
| 11 | 1.50181 | 67.058 | 78.064 | 94.412 | 2.17530 | 69.892 | 76.618 | 93.620 |
| 12 | 1.49071 | 67.459 | 78.226 | 94.518 | 2.16513 | 70.156 | 76.732 | 93.626 |
| 13 | 1.48261 | 67.768 | 78.392 | 94.612 | 2.15641 | 70.313 | 76.882 | 93.674 |
| 14 | 1.47451 | 68.066 | 78.522 | 94.684 | 2.14820 | 70.496 | 76.958 | 93.720 |
| 15 | 1.46759 | 68.325 | 78.636 | 94.752 | 2.13757 | 70.772 | 77.130 | 93.848 |
| 16 | 1.45982 | 68.648 | 78.810 | 94.818 | 2.12825 | 71.012 | 77.048 | 93.776 |
| 17 | 1.45263 | 68.852 | 78.902 | 94.830 | 2.12261 | 71.146 | 77.284 | 93.908 |
| 18 | 1.44639 | 69.141 | 78.872 | 94.850 | 2.11185 | 71.403 | 77.410 | 93.838 |
| 19 | 1.44171 | 69.304 | 79.044 | 94.874 | 2.10501 | 71.578 | 77.414 | 93.950 |
| 20 | 1.43551 | 69.541 | 79.088 | 94.940 | 2.09828 | 71.743 | 77.624 | 93.972 |
| 21 | 1.42988 | 69.761 | 79.096 | 94.936 | 2.09152 | 71.905 | 77.522 | 93.978 |
| 22 | 1.42351 | 70.023 | 79.302 | 95.000 | 2.08454 | 72.073 | 77.686 | 94.026 |
| 23 | 1.41838 | 70.231 | 79.338 | 95.028 | 2.07835 | 72.231 | 77.614 | 94.144 |
| 24 | 1.41303 | 70.414 | 79.316 | 95.056 | 2.07408 | 72.368 | 77.794 | 94.164 |
| 25 | 1.40762 | 70.623 | 79.340 | 95.086 | 2.07028 | 72.429 | 77.820 | 94.114 |
| 26 | 1.40168 | 70.835 | 79.466 | 95.092 | 2.06581 | 72.605 | 77.838 | 94.126 |
| 27 | 1.39727 | 71.002 | 79.596 | 95.180 | 2.06024 | 72.661 | 77.838 | 94.142 |
| 28 | 1.39051 | 71.267 | 79.578 | 95.118 | 2.05631 | 72.816 | 77.900 | 94.176 |
| 29 | 1.38637 | 71.385 | 79.558 | 95.166 | 2.05630 | 72.805 | 77.948 | 94.192 |
| 30 | 1.38035 | 71.647 | 79.640 | 95.164 | 2.05548 | 72.838 | 77.938 | 94.166 |

#### Small50: epoch별 loss/accuracy

근거: [structural metrics](../output/release_final_20261006/small50/01_structural/metrics.jsonl), [patch metrics](../output/release_final_20261006/small50/03_patch_slimming_paper/metrics.jsonl).

| Epoch | S train loss | S train Top-1 | S val Top-1 | S val Top-5 | P train loss | P train Top-1 | P val Top-1 | P val Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.98962 | 0.991 | 0.970 | 4.088 | 2.87941 | 55.790 | 67.100 | 87.932 |
| 2 | 3.67088 | 16.061 | 20.424 | 43.406 | 2.75641 | 57.029 | 68.052 | 88.654 |
| 3 | 2.90457 | 31.277 | 44.292 | 70.212 | 2.72125 | 57.438 | 68.172 | 88.718 |
| 4 | 2.54086 | 39.666 | 54.416 | 78.984 | 2.71643 | 57.422 | 68.032 | 88.680 |
| 5 | 2.33508 | 44.701 | 59.740 | 83.274 | 2.70099 | 57.667 | 68.278 | 88.784 |
| 6 | 2.20754 | 47.979 | 63.302 | 85.644 | 2.68484 | 57.981 | 68.496 | 88.900 |
| 7 | 2.10772 | 50.470 | 65.424 | 87.312 | 2.67386 | 58.179 | 68.626 | 88.950 |
| 8 | 2.03735 | 52.382 | 66.990 | 88.178 | 2.66211 | 58.428 | 68.794 | 88.988 |
| 9 | 1.98699 | 53.700 | 68.092 | 88.842 | 2.65178 | 58.630 | 69.112 | 89.094 |
| 10 | 1.94817 | 54.785 | 68.896 | 89.384 | 2.64086 | 58.867 | 69.246 | 89.142 |
| 11 | 1.91663 | 55.677 | 69.640 | 89.784 | 2.63226 | 59.048 | 69.172 | 89.138 |
| 12 | 1.88974 | 56.397 | 70.284 | 90.086 | 2.62297 | 59.253 | 69.214 | 89.212 |
| 13 | 1.86775 | 57.082 | 70.590 | 90.372 | 2.61487 | 59.440 | 69.632 | 89.340 |
| 14 | 1.84715 | 57.643 | 71.056 | 90.650 | 2.60582 | 59.594 | 69.630 | 89.416 |
| 15 | 1.83088 | 58.150 | 71.358 | 90.750 | 2.59652 | 59.846 | 69.700 | 89.550 |
| 16 | 1.81425 | 58.652 | 71.796 | 90.914 | 2.58876 | 60.044 | 69.838 | 89.610 |
| 17 | 1.80000 | 59.034 | 72.028 | 91.096 | 2.58048 | 60.225 | 70.060 | 89.760 |
| 18 | 1.78613 | 59.453 | 72.224 | 91.248 | 2.57164 | 60.386 | 70.012 | 89.650 |
| 19 | 1.77529 | 59.802 | 72.382 | 91.324 | 2.56426 | 60.602 | 70.120 | 89.830 |
| 20 | 1.76305 | 60.196 | 72.624 | 91.512 | 2.55580 | 60.756 | 70.134 | 89.848 |
| 21 | 1.75173 | 60.448 | 72.834 | 91.544 | 2.54974 | 60.904 | 70.348 | 89.898 |
| 22 | 1.74067 | 60.848 | 73.032 | 91.736 | 2.54447 | 61.057 | 70.350 | 90.004 |
| 23 | 1.73181 | 61.127 | 73.124 | 91.784 | 2.53774 | 61.155 | 70.394 | 90.000 |
| 24 | 1.72272 | 61.417 | 73.200 | 91.880 | 2.53366 | 61.272 | 70.490 | 89.988 |
| 25 | 1.71279 | 61.669 | 73.514 | 91.966 | 2.52913 | 61.348 | 70.580 | 90.026 |
| 26 | 1.70395 | 62.045 | 73.578 | 92.034 | 2.52331 | 61.496 | 70.634 | 90.052 |
| 27 | 1.69656 | 62.242 | 73.720 | 92.116 | 2.52081 | 61.563 | 70.662 | 90.168 |
| 28 | 1.68679 | 62.565 | 73.866 | 92.196 | 2.51719 | 61.622 | 70.674 | 90.116 |
| 29 | 1.67991 | 62.715 | 73.934 | 92.176 | 2.51478 | 61.690 | 70.750 | 90.124 |
| 30 | 1.67224 | 62.990 | 74.008 | 92.230 | 2.51523 | 61.694 | 70.736 | 90.156 |

### 부록 B. 전체 105개 검색 후보의 복원 결과

각 행은 후보 한 개다. Block은 1-based, n_keep는 CLS 포함 출력 token 수다. `r`은 다음 block의 base ID 외에 추가한 후보 수다. Train1/2/3은 해당 후보의 3epoch MSE이고, Eval MSE/relative/p95는 독립 평가 subset 결과다.

`통과`는 mean relative ≤ 0.02, `계속`은 다음 후보로 진행, `전체 유지`는 ε 미달성 상태의 full-token fallback이다. p95는 채택 기준이 아니므로 통과한 후보도 p95가 0.02보다 클 수 있다. 원본은 선택 ID와 score 통계도 함께 담고 있다.

#### Tiny30: 검색 순서별 후보

근거: [search_records.jsonl](../output/release_final_20261006/tiny30/03_patch_slimming_paper/search_records.jsonl), [최종 architecture.json](../output/release_final_20261006/tiny30/03_patch_slimming_paper/architecture.json).

| Block | r | n_keep | Train1 MSE | Train2 MSE | Train3 MSE | Eval MSE | Eval relative | Eval p95 | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0 | 1 | 0.349243 | 0.318639 | 0.298026 | 0.293824 | 0.670221 | 0.966859 | 계속 |
| 11 | 10 | 11 | 0.093888 | 0.079619 | 0.071831 | 0.071199 | 0.163920 | 0.350519 | 계속 |
| 11 | 20 | 21 | 0.048710 | 0.046527 | 0.045137 | 0.047236 | 0.110077 | 0.262480 | 계속 |
| 11 | 30 | 31 | 0.036665 | 0.035747 | 0.035133 | 0.036956 | 0.086739 | 0.224778 | 계속 |
| 11 | 40 | 41 | 0.028526 | 0.028019 | 0.027644 | 0.029589 | 0.069886 | 0.187286 | 계속 |
| 11 | 50 | 51 | 0.022312 | 0.021981 | 0.021731 | 0.023671 | 0.056193 | 0.153178 | 계속 |
| 11 | 60 | 61 | 0.018645 | 0.018397 | 0.018210 | 0.019880 | 0.047419 | 0.132831 | 계속 |
| 11 | 70 | 71 | 0.015244 | 0.015050 | 0.014906 | 0.016387 | 0.039428 | 0.112945 | 계속 |
| 11 | 80 | 81 | 0.012469 | 0.012324 | 0.012213 | 0.013572 | 0.032774 | 0.095984 | 계속 |
| 11 | 90 | 91 | 0.010807 | 0.010681 | 0.010590 | 0.011779 | 0.028520 | 0.085884 | 계속 |
| 11 | 100 | 101 | 0.008910 | 0.008805 | 0.008727 | 0.009996 | 0.024259 | 0.073687 | 계속 |
| 11 | 110 | 111 | 0.006967 | 0.006879 | 0.006815 | 0.007874 | 0.019094 | 0.059332 | 통과 |
| 10 | 0 | 111 | 0.003465 | 0.002840 | 0.002638 | 0.002637 | 0.000141 | 0.000298 | 통과 |
| 9 | 0 | 111 | 0.003848 | 0.003324 | 0.003160 | 0.003227 | 0.000175 | 0.000412 | 통과 |
| 8 | 0 | 111 | 0.004519 | 0.003995 | 0.003813 | 0.003809 | 0.000211 | 0.000460 | 통과 |
| 7 | 0 | 111 | 0.005903 | 0.005336 | 0.005102 | 0.005092 | 0.000287 | 0.000558 | 통과 |
| 6 | 0 | 111 | 0.004234 | 0.003519 | 0.003277 | 0.003212 | 0.011394 | 0.017738 | 통과 |
| 5 | 0 | 111 | 0.005251 | 0.004627 | 0.004405 | 0.004369 | 0.023729 | 0.035489 | 계속 |
| 5 | 10 | 121 | 0.003465 | 0.003372 | 0.003311 | 0.003341 | 0.018138 | 0.027513 | 통과 |
| 4 | 0 | 121 | 0.003161 | 0.002624 | 0.002430 | 0.002356 | 0.018714 | 0.025885 | 통과 |
| 3 | 0 | 121 | 0.002947 | 0.002382 | 0.002224 | 0.002172 | 0.018654 | 0.025520 | 통과 |
| 2 | 0 | 121 | 0.002535 | 0.002010 | 0.001882 | 0.001850 | 0.016802 | 0.020878 | 통과 |
| 1 | 0 | 121 | 0.004072 | 0.003145 | 0.002839 | 0.002735 | 0.014008 | 0.017506 | 통과 |

#### Tiny50: 검색 순서별 후보

근거: [search_records.jsonl](../output/release_final_20261006/tiny50/03_patch_slimming_paper/search_records.jsonl), [최종 architecture.json](../output/release_final_20261006/tiny50/03_patch_slimming_paper/architecture.json).

| Block | r | n_keep | Train1 MSE | Train2 MSE | Train3 MSE | Eval MSE | Eval relative | Eval p95 | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0 | 1 | 0.307130 | 0.269988 | 0.245408 | 0.238137 | 0.706175 | 1.052296 | 계속 |
| 11 | 10 | 11 | 0.091346 | 0.072937 | 0.062538 | 0.059833 | 0.181219 | 0.364821 | 계속 |
| 11 | 20 | 21 | 0.038027 | 0.035505 | 0.033992 | 0.034234 | 0.103956 | 0.243600 | 계속 |
| 11 | 30 | 31 | 0.027597 | 0.026710 | 0.026180 | 0.026945 | 0.082088 | 0.212418 | 계속 |
| 11 | 40 | 41 | 0.021459 | 0.021013 | 0.020723 | 0.021555 | 0.066088 | 0.177135 | 계속 |
| 11 | 50 | 51 | 0.017392 | 0.017115 | 0.016922 | 0.017672 | 0.054535 | 0.147819 | 계속 |
| 11 | 60 | 61 | 0.014446 | 0.014236 | 0.014096 | 0.014657 | 0.045614 | 0.130225 | 계속 |
| 11 | 70 | 71 | 0.011843 | 0.011691 | 0.011589 | 0.012292 | 0.038469 | 0.103512 | 계속 |
| 11 | 80 | 81 | 0.009510 | 0.009396 | 0.009316 | 0.009925 | 0.031275 | 0.084094 | 계속 |
| 11 | 90 | 91 | 0.008244 | 0.008151 | 0.008087 | 0.008615 | 0.027376 | 0.074586 | 계속 |
| 11 | 100 | 101 | 0.006867 | 0.006788 | 0.006737 | 0.007235 | 0.023104 | 0.064591 | 계속 |
| 11 | 110 | 111 | 0.005358 | 0.005291 | 0.005247 | 0.005765 | 0.018580 | 0.053502 | 통과 |
| 10 | 0 | 111 | 0.003570 | 0.003004 | 0.002792 | 0.002765 | 0.000133 | 0.000290 | 통과 |
| 9 | 0 | 111 | 0.003848 | 0.003285 | 0.003097 | 0.003211 | 0.000155 | 0.000364 | 통과 |
| 8 | 0 | 111 | 0.004259 | 0.003726 | 0.003547 | 0.003532 | 0.000173 | 0.000351 | 통과 |
| 7 | 0 | 111 | 0.006411 | 0.005786 | 0.005525 | 0.005490 | 0.000270 | 0.000494 | 통과 |
| 6 | 0 | 111 | 0.004781 | 0.003975 | 0.003694 | 0.003637 | 0.011068 | 0.017027 | 통과 |
| 5 | 0 | 111 | 0.006605 | 0.005855 | 0.005581 | 0.005501 | 0.023453 | 0.035211 | 계속 |
| 5 | 10 | 121 | 0.004252 | 0.004133 | 0.004059 | 0.004055 | 0.017277 | 0.026025 | 통과 |
| 4 | 0 | 121 | 0.003379 | 0.002761 | 0.002532 | 0.002454 | 0.015794 | 0.021500 | 통과 |
| 3 | 0 | 121 | 0.003312 | 0.002780 | 0.002605 | 0.002538 | 0.016825 | 0.022662 | 통과 |
| 2 | 0 | 121 | 0.002026 | 0.001673 | 0.001570 | 0.001541 | 0.011983 | 0.014966 | 통과 |
| 1 | 0 | 121 | 0.003431 | 0.002588 | 0.002353 | 0.002286 | 0.011517 | 0.014277 | 통과 |

#### Small30: 검색 순서별 후보

근거: [search_records.jsonl](../output/release_final_20261006/small30/03_patch_slimming_paper/search_records.jsonl), [최종 architecture.json](../output/release_final_20261006/small30/03_patch_slimming_paper/architecture.json).

| Block | r | n_keep | Train1 MSE | Train2 MSE | Train3 MSE | Eval MSE | Eval relative | Eval p95 | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0 | 1 | 0.633870 | 0.517514 | 0.440835 | 0.427965 | 0.442173 | 0.743333 | 계속 |
| 11 | 10 | 11 | 0.299280 | 0.262119 | 0.243438 | 0.252337 | 0.266575 | 0.625468 | 계속 |
| 11 | 20 | 21 | 0.169434 | 0.159734 | 0.152566 | 0.166272 | 0.176203 | 0.458097 | 계속 |
| 11 | 30 | 31 | 0.115962 | 0.110866 | 0.106696 | 0.124331 | 0.131614 | 0.373032 | 계속 |
| 11 | 40 | 41 | 0.070047 | 0.066659 | 0.064070 | 0.082240 | 0.087660 | 0.244095 | 계속 |
| 11 | 50 | 51 | 0.051033 | 0.048734 | 0.046940 | 0.065479 | 0.069758 | 0.197707 | 계속 |
| 11 | 60 | 61 | 0.041229 | 0.039538 | 0.038181 | 0.058104 | 0.061829 | 0.176774 | 계속 |
| 11 | 70 | 71 | 0.030084 | 0.028790 | 0.027810 | 0.046015 | 0.049027 | 0.143585 | 계속 |
| 11 | 80 | 81 | 0.024009 | 0.023032 | 0.022264 | 0.039496 | 0.042335 | 0.124726 | 계속 |
| 11 | 90 | 91 | 0.015272 | 0.014218 | 0.013564 | 0.025028 | 0.026952 | 0.073309 | 계속 |
| 11 | 100 | 101 | 0.011732 | 0.010935 | 0.010403 | 0.020596 | 0.022238 | 0.060050 | 계속 |
| 11 | 110 | 111 | 0.008946 | 0.008308 | 0.007883 | 0.016326 | 0.017659 | 0.046413 | 통과 |
| 10 | 0 | 111 | 0.071333 | 0.058798 | 0.054068 | 0.055165 | 0.008050 | 0.017520 | 통과 |
| 9 | 0 | 111 | 0.013555 | 0.012136 | 0.011590 | 0.011893 | 0.002465 | 0.005237 | 통과 |
| 8 | 0 | 111 | 0.009933 | 0.008930 | 0.008596 | 0.008739 | 0.002201 | 0.004713 | 통과 |
| 7 | 0 | 111 | 0.007667 | 0.007008 | 0.006791 | 0.006866 | 0.002220 | 0.004445 | 통과 |
| 6 | 0 | 111 | 0.007595 | 0.006560 | 0.006233 | 0.006222 | 0.002287 | 0.004401 | 통과 |
| 5 | 0 | 111 | 0.007706 | 0.005690 | 0.005170 | 0.005073 | 0.013556 | 0.027280 | 통과 |
| 4 | 0 | 111 | 0.008924 | 0.006080 | 0.005435 | 0.005316 | 0.031749 | 0.043960 | 계속 |
| 4 | 10 | 121 | 0.004562 | 0.004362 | 0.004237 | 0.004274 | 0.025568 | 0.036592 | 계속 |
| 4 | 20 | 131 | 0.003715 | 0.003637 | 0.003580 | 0.003667 | 0.021930 | 0.030849 | 계속 |
| 4 | 30 | 141 | 0.003211 | 0.003163 | 0.003128 | 0.003228 | 0.019313 | 0.026707 | 통과 |
| 3 | 0 | 141 | 0.010559 | 0.006352 | 0.005533 | 0.005329 | 0.044290 | 0.048886 | 계속 |
| 3 | 10 | 151 | 0.004923 | 0.004640 | 0.004475 | 0.004440 | 0.036907 | 0.040823 | 계속 |
| 3 | 20 | 161 | 0.004204 | 0.004104 | 0.004029 | 0.004032 | 0.033520 | 0.036993 | 계속 |
| 3 | 30 | 171 | 0.003832 | 0.003774 | 0.003728 | 0.003740 | 0.031101 | 0.034421 | 계속 |
| 3 | 40 | 181 | 0.003577 | 0.003533 | 0.003499 | 0.003515 | 0.029232 | 0.032206 | 계속 |
| 3 | 50 | 191 | 0.003377 | 0.003343 | 0.003315 | 0.003333 | 0.027722 | 0.030498 | 계속 |
| 3 | 56 | 197 | 0.003242 | 0.003215 | 0.003193 | 0.003211 | 0.026703 | 0.029359 | 전체 유지 |
| 2 | 0 | 197 | 0.011310 | 0.006159 | 0.005200 | 0.004947 | 0.046311 | 0.054766 | 전체 유지 |
| 1 | 0 | 197 | 0.007939 | 0.004103 | 0.003363 | 0.003167 | 0.030222 | 0.036569 | 전체 유지 |

#### Small50: 검색 순서별 후보

근거: [search_records.jsonl](../output/release_final_20261006/small50/03_patch_slimming_paper/search_records.jsonl), [최종 architecture.json](../output/release_final_20261006/small50/03_patch_slimming_paper/architecture.json).

| Block | r | n_keep | Train1 MSE | Train2 MSE | Train3 MSE | Eval MSE | Eval relative | Eval p95 | 판정 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 0 | 1 | 0.967740 | 0.677416 | 0.533698 | 0.514270 | 0.508154 | 0.846833 | 계속 |
| 11 | 10 | 11 | 0.405224 | 0.331999 | 0.302426 | 0.308704 | 0.310288 | 0.702885 | 계속 |
| 11 | 20 | 21 | 0.169007 | 0.156231 | 0.147985 | 0.161055 | 0.166208 | 0.472339 | 계속 |
| 11 | 30 | 31 | 0.118317 | 0.112405 | 0.107686 | 0.129520 | 0.133711 | 0.413229 | 계속 |
| 11 | 40 | 41 | 0.072026 | 0.068020 | 0.065149 | 0.083241 | 0.087657 | 0.253695 | 계속 |
| 11 | 50 | 51 | 0.046817 | 0.044499 | 0.042716 | 0.061899 | 0.065360 | 0.186099 | 계속 |
| 11 | 60 | 61 | 0.033786 | 0.032130 | 0.030868 | 0.048718 | 0.051474 | 0.143387 | 계속 |
| 11 | 70 | 71 | 0.024899 | 0.023616 | 0.022658 | 0.039016 | 0.041565 | 0.118061 | 계속 |
| 11 | 80 | 81 | 0.017998 | 0.016912 | 0.016196 | 0.029383 | 0.031709 | 0.086734 | 계속 |
| 11 | 90 | 91 | 0.013396 | 0.012573 | 0.012021 | 0.022951 | 0.024893 | 0.072236 | 계속 |
| 11 | 100 | 101 | 0.010174 | 0.009525 | 0.009087 | 0.018712 | 0.020325 | 0.057664 | 계속 |
| 11 | 110 | 111 | 0.007738 | 0.007064 | 0.006667 | 0.014232 | 0.015394 | 0.042098 | 통과 |
| 10 | 0 | 111 | 0.072750 | 0.056954 | 0.052085 | 0.052444 | 0.007222 | 0.015228 | 통과 |
| 9 | 0 | 111 | 0.011181 | 0.009871 | 0.009404 | 0.009765 | 0.001871 | 0.003849 | 통과 |
| 8 | 0 | 111 | 0.009211 | 0.008084 | 0.007773 | 0.007987 | 0.001755 | 0.003537 | 통과 |
| 7 | 0 | 111 | 0.007749 | 0.006938 | 0.006690 | 0.006782 | 0.001785 | 0.003489 | 통과 |
| 6 | 0 | 111 | 0.008430 | 0.007196 | 0.006807 | 0.006798 | 0.001940 | 0.003606 | 통과 |
| 5 | 0 | 111 | 0.007689 | 0.005538 | 0.005002 | 0.004990 | 0.010328 | 0.019535 | 통과 |
| 4 | 0 | 111 | 0.008691 | 0.006037 | 0.005315 | 0.005196 | 0.026192 | 0.035367 | 계속 |
| 4 | 10 | 121 | 0.004273 | 0.004027 | 0.003884 | 0.003928 | 0.019827 | 0.027282 | 통과 |
| 3 | 0 | 121 | 0.007888 | 0.004289 | 0.003552 | 0.003389 | 0.025302 | 0.029338 | 계속 |
| 3 | 10 | 131 | 0.002981 | 0.002783 | 0.002679 | 0.002675 | 0.019966 | 0.023365 | 통과 |
| 2 | 0 | 131 | 0.007814 | 0.004154 | 0.003405 | 0.003234 | 0.028795 | 0.033312 | 계속 |
| 2 | 10 | 141 | 0.002568 | 0.002388 | 0.002303 | 0.002286 | 0.020362 | 0.023887 | 계속 |
| 2 | 20 | 151 | 0.001919 | 0.001879 | 0.001851 | 0.001849 | 0.016477 | 0.019668 | 통과 |
| 1 | 0 | 151 | 0.009838 | 0.005364 | 0.004399 | 0.004141 | 0.036505 | 0.040837 | 계속 |
| 1 | 10 | 161 | 0.003245 | 0.002856 | 0.002667 | 0.002618 | 0.023081 | 0.026627 | 계속 |
| 1 | 20 | 171 | 0.002227 | 0.002139 | 0.002080 | 0.002068 | 0.018250 | 0.021309 | 통과 |

### 부록 C. Batch probe 전체 기록

본 학습 전에 synthetic 입력으로 실행한 짧은 probe다. Images/s는 probe 구간 값이며 ImageNet 실제 학습 처리량이나 NPU inference FPS가 아니다. `insufficient_headroom`은 잔여 메모리가 요구 여유 1,536 MiB에 미달했음을 뜻한다.

선택은 통과 후보 중 처리량을 고려하며, 메모리 사용량이 가장 크거나 batch가 가장 큰 후보를 항상 고르는 규칙이 아니다.

| 실험 | 단계 | Batch | 상태 | Probe images/s | Peak allocated MiB | Free after MiB | 실제 선택 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny30 | structural | 256 | ok | 1441.53 | 4732.32 | 5705.69 | — |
| Tiny30 | structural | 384 | ok | 1490.35 | 7038.28 | 3333.69 | 선택 |
| Tiny30 | structural | 512 | insufficient_headroom | 1516.90 | 9293.60 | 929.69 | — |
| Tiny30 | patch | 384 | ok | 2263.89 | 4658.04 | 5027.69 | — |
| Tiny30 | patch | 512 | ok | 2525.94 | 6127.05 | 3129.69 | 선택 |
| Tiny30 | patch | 640 | insufficient_headroom | 2590.22 | 7628.31 | 1109.69 | — |
| Tiny50 | structural | 256 | ok | 1407.39 | 4732.32 | 5705.69 | — |
| Tiny50 | structural | 384 | ok | 1467.33 | 7038.28 | 3331.69 | 선택 |
| Tiny50 | structural | 512 | insufficient_headroom | 1503.18 | 9293.60 | 929.69 | — |
| Tiny50 | patch | 384 | ok | 2574.52 | 4179.09 | 5557.69 | — |
| Tiny50 | patch | 512 | ok | 2666.80 | 5499.43 | 3841.69 | — |
| Tiny50 | patch | 640 | ok | 2702.23 | 6842.06 | 2095.69 | 선택 |
| Small30 | structural | 128 | ok | 637.09 | 5035.15 | 5647.69 | — |
| Small30 | structural | 192 | ok | 672.39 | 7279.20 | 3415.69 | 선택 |
| Small30 | structural | 256 | insufficient_headroom | 674.03 | 9479.43 | 1067.69 | — |
| Small30 | patch | 128 | ok | 895.11 | 3888.65 | 6103.69 | — |
| Small30 | patch | 192 | ok | 961.33 | 5720.66 | 4017.69 | — |
| Small30 | patch | 256 | ok | 964.86 | 7489.61 | 1853.69 | 선택 |
| Small30 | patch | 320 | insufficient_headroom | 968.23 | 9285.09 | 529.69 | — |
| Small50 | structural | 128 | ok | 634.41 | 5035.15 | 5647.69 | — |
| Small50 | structural | 192 | ok | 672.25 | 7279.20 | 3415.69 | 선택 |
| Small50 | structural | 256 | insufficient_headroom | 687.27 | 9479.43 | 1067.69 | — |
| Small50 | patch | 128 | ok | 1007.15 | 3071.26 | 6677.69 | — |
| Small50 | patch | 192 | ok | 1202.33 | 4517.20 | 4849.69 | — |
| Small50 | patch | 256 | ok | 1255.98 | 5918.54 | 2869.69 | 선택 |
| Small50 | patch | 320 | insufficient_headroom | 1268.99 | 7338.73 | 915.69 | — |

### 부록 D. 최종 ONNX/MXQ 파일과 SHA-256

ONNX 이름은 `<model>__<stage>__nhwc_b1_npusafe.onnx`, MXQ 이름은 `<model>__<stage>__nhwc_b1_npusafe_vt_global8.mxq`다. 표는 공통 suffix를 생략해 식별자를 짧게 표기했다. 같은 baseline ONNX의 중복은 정상이며 네 실험 그룹의 3종 비교를 위한 구성이다.

이 해시는 파일 내용 추적용이다. Hash 일치가 새 하드웨어/새 runtime에서 동일 accuracy를 자동 보장하지는 않는다.

| 식별자 | ONNX SHA-256 | 평가한 MXQ SHA-256 |
| --- | --- | --- |
| tiny30__baseline | `0d8655a8e7641e9023c6464f7c97ca172b447709f5c2b3abb6c91d83a19b156b` | `96de47b4f6aa3698ba76049f3208acdb5201b48e2221b0ecaa73cb5314b2aa54` |
| tiny30__reduced | `bc628a54be653609a895d279eb43826827eeaaddc3cf8ea23c3855108551cff6` | `0c651c7c4b202aa400671924fa5ef7be51c21900df82d1606653c7c834f03f0c` |
| tiny30__patch_slimming | `08d4014de88f6f156dd88e312e240bf71e3aff0af473b2fd33aba03d7e5a5fa1` | `90a9b4ddd7eaf6492539eab8f4af02ea8420e46453d92ef2df3f5721ad1d1113` |
| tiny50__baseline | `0d8655a8e7641e9023c6464f7c97ca172b447709f5c2b3abb6c91d83a19b156b` | `2c740f6034989d5512d3e77a1df693f15c3c2076346565f018e3b77cbbff73ff` |
| tiny50__reduced | `aa122558ccb0ea5b3cd081d6b8ed1129948edb95bcbcb6e1a63753859b9fe169` | `52be73fb93398a53f8e0c6c9235ffa1c9320e36616b003cda71e7031064a6cb1` |
| tiny50__patch_slimming | `90086de3e74482a4fdbc25b0065e6d2dce9895cb1e417aa9af63d3f77302f65c` | `8626d858b2fb43e3023c226be212173d0d358e4e4bf88c9827b072cabfc116f4` |
| small30__baseline | `9812da2a85262ee515dfb7947f5ced1186a0f447005e7a7988c6bb996726cdc0` | `f6d45ea2ce23104d184bbb42b09cd8515e6244eb25485367485715784f58fd48` |
| small30__reduced | `1bf7f5fade8b32de735151f01f714e63d98e815ef9204fc2562ff3e978bad5cd` | `7411c8c7441d1244fdfb8b88c15294f419a7a3cc1e2234dd827a2ed4dd39d0ab` |
| small30__patch_slimming | `0146dde675d79f36d95529ddda67ace59dca3a7a5f718fb02f4039716ab3b6a1` | `bbd41a00058593de61acce35bd1121882551b861d17866db0c7e10ffa36b879a` |
| small50__baseline | `9812da2a85262ee515dfb7947f5ced1186a0f447005e7a7988c6bb996726cdc0` | `590a31cfb6dd8dcbfab5ba30d455d7eaa2f7c0d690552435f7e66b0afb31034f` |
| small50__reduced | `0caf8659b7232bedff19e42a2f798b8b508338a80c8381047630d422671183fc` | `3719c948c08f3e79a6bdbfc9a4917d334781c46dffda0483eaf5e9a4a25617a1` |
| small50__patch_slimming | `ec63a02ee86783a268a4cfb0f001aa16fdbfb549128a9a2c8d0d794952789ce3` | `668741deb2709383a9caa521f3bf06d3d1a9393f999eec47558173b662d3ef19` |

### 부록 E. 핵심 원본 증거 링크

이 문서의 표는 아래 숫자를 옮기거나 명시된 식으로 계산한 것이다. 원본 파일이 최종 상세값의 기준이며, 문서 표시용 반올림 때문에 마지막 자리에는 차이가 있을 수 있다.

| 실험 | 예산 | 학습 설정 | 검색 설정 | 메모리 | FP32 평가 |
| --- | --- | --- | --- | --- | --- |
| Tiny30 | [budget](../output/release_final_20261006/tiny30/00_budget/budget.json) | [structural](../output/release_final_20261006/tiny30/configs/structural.yaml) | [patch](../output/release_final_20261006/tiny30/configs/patch_slimming_paper.yaml) | [05_memory](../output/release_final_20261006/tiny30/05_memory/) | [06_evaluation](../output/release_final_20261006/tiny30/06_evaluation/) |
| Tiny50 | [budget](../output/release_final_20261006/tiny50/00_budget/budget.json) | [structural](../output/release_final_20261006/tiny50/configs/structural.yaml) | [patch](../output/release_final_20261006/tiny50/configs/patch_slimming_paper.yaml) | [05_memory](../output/release_final_20261006/tiny50/05_memory/) | [06_evaluation](../output/release_final_20261006/tiny50/06_evaluation/) |
| Small30 | [budget](../output/release_final_20261006/small30/00_budget/budget.json) | [structural](../output/release_final_20261006/small30/configs/structural.yaml) | [patch](../output/release_final_20261006/small30/configs/patch_slimming_paper.yaml) | [05_memory](../output/release_final_20261006/small30/05_memory/) | [06_evaluation](../output/release_final_20261006/small30/06_evaluation/) |
| Small50 | [budget](../output/release_final_20261006/small50/00_budget/budget.json) | [structural](../output/release_final_20261006/small50/configs/structural.yaml) | [patch](../output/release_final_20261006/small50/configs/patch_slimming_paper.yaml) | [05_memory](../output/release_final_20261006/small50/05_memory/) | [06_evaluation](../output/release_final_20261006/small50/06_evaluation/) |

- 서버 전체 설정: [final.yaml](../configs/final.yaml)
- 서버 환경: [environment.json](../output/release_final_20261006/provenance/environment.json)
- 서버 source 추적: [source_manifest.json](../output/release_final_20261006/provenance/source_manifest.json)
- 서버 결과 요약: [FINAL_RESULTS.md](../output/release_final_20261006/reports/FINAL_RESULTS.md)
- 서버 배포 manifest: [MANIFEST.json](../output/release_final_20261006/MANIFEST.json)
- NPU 실행 환경/명령: [run_metadata.json](../../npu/results/final_benchmark/final_20261015/run_metadata.json)
- NPU 전체 결과 JSON: [benchmark.json](../../npu/results/final_benchmark/final_20261015/benchmark.json)
- NPU CSV: [summary.csv](../../npu/results/final_benchmark/final_20261015/summary.csv)
- NPU 개별 모델 기록: [models](../../npu/results/final_benchmark/final_20261015/models/)
- NPU 샘플 목록: [samples.tsv](../../npu/results/final_benchmark/final_20261015/samples.tsv)
- NPU Top-1 예측 배열: [predictions](../../npu/results/final_benchmark/final_20261015/predictions/)
- NPU 독립 메모리: [isolated_memory/summary.json](../../npu/results/final_benchmark/final_20261015/isolated_memory/summary.json)
- NPU 보존 자산 manifest: [RELEASE_MANIFEST.json](../../npu/assets/RELEASE_MANIFEST.json)
- NPU 제출물 검증: [VALIDATION.md](../../npu/docs/VALIDATION.md)

GitHub에서 두 저장소를 별도로 열면 `../../npu/` 상대링크는 작동하지 않을 수 있다. 해당 경로는 로컬 sibling 저장소 구성을 기준으로 했으며, 외부에서는 [npu 저장소](https://github.com/junsu0306/npu)의 동일 상대경로를 사용한다.
