# Patch Slimming 전체 구현 및 실험 결과

> 2026-08-06 업데이트: 논문 epsilon과 element-wise MSE를 직접 대응한 문제를 수정하고,
> reconstruction train/eval을 분리한 relative-error pilot 결과는
> [`experiments/08_ERROR_METRIC_AND_DATA_SPLIT_CORRECTION.md`](experiments/08_ERROR_METRIC_AND_DATA_SPLIT_CORRECTION.md)에 기록했다.
> Structural Tiny-30에서 42.59% body MAC 감소, Pre-FT Top-1 68.48%를 얻어 기존 MSE
> epsilon=.02 결과의 과도한 81.40% 절단을 바로잡았다.
>
> 2026-08-06 progressive update: layer마다 중요도 상위 token을 최소 3개씩 복원하는
> deployment prior를 결합해 `[151,148,145,142,139,136,133,130,127,124,121,1]` schedule을
> 얻었다. Structural Tiny-30 대비 body MAC 35.77% 감소, Pre-FT Top-1 69.63%이며 자세한
> 결과는 [`experiments/09_PROGRESSIVE_TOKEN_SCHEDULE_EXPERIMENT.md`](experiments/09_PROGRESSIVE_TOKEN_SCHEDULE_EXPERIMENT.md)에 기록했다.

## 1. 목적과 실제 적용 대상

이 프로젝트의 최종 목적은 이미지마다 동적으로 토큰을 고르는 EViT 대신, 데이터셋
통계로 미리 정한 정적 토큰 mask를 사용해 NPU에서 컴파일 가능한 Patch Slimming을
구현하는 것이다.

실제 적용 대상은 원본 DeiT가 아니라 Stage 1 structural pruning을 거쳐 FFN channel이
물리적으로 제거된 다음 네 모델이다.

- `output/vit_tiny_30_final/reduced.pt`
- `output/vit_tiny_50_final/reduced.pt`
- `output/vit_small_30_final/reduced.pt`
- `output/vit_small_50_final/reduced.pt`

원본 DeiT-Tiny 실험은 알고리즘과 export 경로를 검증하기 위한 reference 실험이며,
production token schedule은 structural 모델에서 다시 산출했다.

## 2. 문서와 코드 기준

- 최초 알고리즘·구현 계획: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- 논문 원문: [`../references/patch_slimming_cvpr2022.pdf`](../references/patch_slimming_cvpr2022.pdf)
- 핵심 라이브러리: `patch_slimming/psvit/`
- 실행 workflow: `patch_slimming/scripts/`
- 설정: `patch_slimming/configs/reference/`, `patch_slimming/configs/structural/`
- 테스트: `patch_slimming/tests/`

## 3. 실제 실험 진행 순서

### 3.1 NPU 연산 사전 검증

전체 모델 구현 전에 rectangular attention과 정적 토큰 선택을 toy ONNX로 검증했다.
Aries2에서는 activation `Gather`가 지원되지 않았지만, 고정 선택 행렬 `P @ x`는
컴파일됐다. 따라서 export 경로는 다음으로 고정했다.

- 입력: static batch-one NHWC `[1, 224, 224, 3]`
- 토큰 선택: 상수 selection-matrix `MatMul`
- attention: `Q: N_out`, `K/V: N_in` rectangular attention
- 금지 연산: `Gather`, `GatherElements`, `ScatterElements`, `TopK`, `Equal`

구현은 `scripts/toy_npu_validation.py`, 검증 ONNX는 `artifacts/`에 있다.
JSON mask가 실제 compact forward와 NPU 상수 선택행렬로 변환되는 과정, 그리고 Gather
우회 MatMul의 추가 비용과 필요한 최적화는
[`experiments/05_STATIC_TOKEN_SELECTION_AND_NPU_COST.md`](experiments/05_STATIC_TOKEN_SELECTION_AND_NPU_COST.md)에 정리했다.

### 3.2 핵심 Patch Slimming 구현

다음 기능을 구현했다.

- global token ID와 nested mask 검증
- timm ViT block instrumentation
- 논문 path-energy significance score의 dynamic programming 계산
- top-down candidate search와 block reconstruction
- layer checkpoint 저장·재개
- `SlimBlock`의 rectangular attention
- 실제 sequence 길이가 감소하는 `CompactPSViT`
- index-select 학습 경로와 selection-matrix export 경로의 수치 동등성
- reduced FFN topology를 검색·학습·평가·export 전체에서 보존하는 checkpoint loader
- W&B candidate/finetuning 로깅

최종 회귀 테스트는 33/33 통과했다.
Patch 중요도 수식의 tensor별 구현, head-path DP, production 3·6·9 ID 선택, 논문식
reconstruction search, GPU `index_select` compact forward의 전체 연결은
[`experiments/06_PATCH_SIGNIFICANCE_AND_GPU_PRUNING_IMPLEMENTATION.md`](experiments/06_PATCH_SIGNIFICANCE_AND_GPU_PRUNING_IMPLEMENTATION.md)에 정리했다.

### 3.3 Smoke 실험

32장 calibration으로 전체 pipeline을 빠르게 검증했다.

- schedule: `[197, 81, ..., 81, 1]`
- 파인튜닝 전 Top-1: 59.28%
- transformer MAC 감소: 64.04%
- PyTorch/ORT 최대 오차: `1.97e-06`
- 최종 smoke ONNX에서 금지 연산 없음

이 결과는 correctness 검증용이며 최종 architecture 판단에는 사용하지 않았다. 상세:
[`experiments/01_SMOKE_EXPERIMENT.md`](experiments/01_SMOKE_EXPERIMENT.md).

### 3.4 원본 DeiT-Tiny 메인 실험

ImageNet train 2,048장으로 full top-down search를 수행하고 30 epoch fine-tuning했다.

- schedule: `[197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 91, 41, 1]`
- pretrained baseline: Top-1 71.36%, Top-5 90.71%
- search 직후: Top-1 61.19%, Top-5 84.48%
- epoch 27 best: Top-1 72.82%, Top-5 91.26%
- baseline 대비: Top-1 `+1.46%p`, Top-5 `+0.55%p`
- transformer MAC 감소: 18.34%
- ONNX Runtime 최대 오차: `3.10e-06`
- NPU 금지 연산 없음

Block 8과 7은 epsilon을 만족하지 못해 all-token fallback이 발생했다. 이 결과는
파인튜닝이 큰 초기 정확도 하락을 회복할 수 있음을 보였지만, structural-pruned 모델의
최종 mask로 직접 재사용하지 않았다. 상세:
[`experiments/02_DEIT_TINY_MAIN_EXPERIMENT.md`](experiments/02_DEIT_TINY_MAIN_EXPERIMENT.md).

### 3.5 Production structural 모델 재산출

실제 배포 모델에서는 1-based layer 3·6·9에서만 토큰 수를 변경하는 stage 구조를
사용했다. ImageNet train 1,024장으로 모델별 significance score와 token ID를 다시
산출하고 ImageNet validation 50,000장으로 모든 후보를 평가했다.

| Policy | Layer 3 | Layer 6 | Layer 9 |
|---|---:|---:|---:|
| mild | 177 | 157 | 137 |
| medium | 157 | 118 | 79 |
| aggressive | 138 | 89 | 40 |
| extra-aggressive | 118 | 69 | 30 |

파인튜닝 전 결과:

| 모델 | Structural baseline | Mild Top-1 / MAC 감소 | Medium Top-1 / MAC 감소 | Aggressive Top-1 / MAC 감소 |
|---|---:|---:|---:|---:|
| Tiny 30 | 73.53% | 70.91% / 19.97% | 66.08% / 34.75% | 59.42% / 45.29% |
| Tiny 50 | 66.14% | 63.37% / 21.63% | 57.72% / 36.42% | 49.41% / 46.69% |
| Small 30 | 81.05% | 79.16% / 19.82% | 72.44% / 35.51% | 57.35% / 46.85% |
| Small 50 | 77.41% | 75.26% / 20.53% | 65.10% / 35.59% | 54.31% / 46.37% |

### 3.6 Medium 정책 1-epoch 회복 검증

파인튜닝 전 하락을 넉넉히 허용한다는 실험 방향에 따라 medium 후보를 ImageNet 전체
train으로 1 epoch 학습했다. LR은 `2e-5`, global batch는 512, warmup은 사용하지 않았다.

| 모델 | 파인튜닝 전 | 1 epoch 후 | 회복량 | Structural baseline까지 차이 |
|---|---:|---:|---:|---:|
| Tiny 30 | 66.08% | 68.42% | +2.34%p | -5.11%p |
| Tiny 50 | 57.72% | 61.19% | +3.47%p | -4.95%p |
| Small 30 | 72.44% | 75.34% | +2.90%p | -5.71%p |
| Small 50 | 65.10% | 70.94% | +5.84%p | -6.47%p |

네 모델 모두 한 epoch 만에 뚜렷하게 회복했다. 상세 후보 ID, W&B run ID, checkpoint
경로는 [`experiments/03_STRUCTURAL_MODELS_EXPERIMENT.md`](experiments/03_STRUCTURAL_MODELS_EXPERIMENT.md)에 있다.

### 3.7 논문식 epsilon 0.05 stress test

token 수까지 reconstruction MSE로 결정하는 논문식 top-down search를 의도적으로
느슨한 epsilon 0.05에 실행했다. 1 epoch 후에도 Tiny30/50은 24.53/20.18%,
Small30/50은 64.71/50.55%에 머물러 production에는 지나치게 공격적이었다. 상세 결과는
[`experiments/04_PAPER_EPSILON_STRUCTURAL_EXPERIMENT.md`](experiments/04_PAPER_EPSILON_STRUCTURAL_EXPERIMENT.md)에 있다.

### 3.8 논문식 epsilon 0.02 재실험

epsilon을 논문의 큰 설정인 0.02로 낮춰 동일한 네 structural 모델에 search, ImageNet
val 50k, 1-epoch recovery를 다시 수행했다.

| 모델 | 검색 schedule | Pre-FT | 1 epoch 후 | Medium 1 epoch |
|---|---|---:|---:|---:|
| Tiny 30 | `[197, 41 x11, 1]` | 41.86% | 52.44% | 68.42% |
| Tiny 50 | `[197, 31 x11, 1]` | 20.23% | 39.69% | 61.19% |
| Small 30 | `[197, 111 x10, 81, 1]` | 71.92% | 73.34% | 75.34% |
| Small 50 | `[197, 101 x10, 81, 1]` | 62.04% | 65.38% | 70.94% |

epsilon 0.05보다 네 모델 모두 크게 개선됐다. Small30은 medium과 2.00%p 차이에서
transformer body MAC을 추가 19.03% 줄여 장기 학습 비교 가치가 있다. Tiny에는
epsilon 0.02도 여전히 공격적이므로 현재 production 선택은 medium으로 유지한다.
selection-matrix 비용까지 포함한 분석과 모든 W&B run은
[`experiments/07_PAPER_EPSILON_002_STRUCTURAL_EXPERIMENT.md`](experiments/07_PAPER_EPSILON_002_STRUCTURAL_EXPERIMENT.md)에 있다.

## 4. 현재 선택

production 장기 학습 후보는 네 모델 모두 **medium `(157, 118, 79)`**으로 선택했다.

최종 schedule은 다음과 같다.

```text
[197, 197, 197, 157, 157, 157, 118, 118, 118, 79, 79, 79, 1]
```

원본 full-width ViT transformer 대비 structural pruning과 token slimming을 합친 MAC
감소는 다음과 같다.

| 모델 | Combined transformer MAC 감소 |
|---|---:|
| Tiny 30 | 52.70% |
| Tiny 50 | 65.55% |
| Small 30 | 54.00% |
| Small 50 | 66.37% |

Mild는 정확도 우선 fallback이다. Aggressive 계열은 medium 이후 추가 절감량에 비해
Small 모델의 초기 정확도 손실이 너무 커 메인 후보에서 제외했다.

## 5. W&B

Structural production 실험은 기존 프로젝트와 분리했다.

- Project: `vit-structural-patch-slimming`
- URL: <https://wandb.ai/junsu000306/vit-structural-patch-slimming>
- 후보 ablation과 네 medium recovery run을 모두 기록했다.
- checkpoint 전에 중단한 저LR warmup 시도는 결과 표에서 제외했다.

Paper-style 실험은 epsilon별 독립 프로젝트로 분리했다.

- Epsilon 0.05: <https://wandb.ai/junsu000306/vit-structural-patch-slimming-paper-search>
- Epsilon 0.02: <https://wandb.ai/junsu000306/vit-structural-patch-slimming-paper-eps002>
- Epsilon 0.02 프로젝트에는 네 search, 네 pre-FT full validation, 네 recovery run이 있다.

## 6. 산출물

```text
output/ps_structural/
├── vit_tiny_30_final/{mild,medium,aggressive,extra_aggressive}/
├── vit_tiny_50_final/{mild,medium,aggressive,extra_aggressive}/
├── vit_small_30_final/{mild,medium,aggressive,extra_aggressive}/
└── vit_small_50_final/{mild,medium,aggressive,extra_aggressive}/
```

각 후보 폴더에는 architecture, significance score, searched weights, 평가 수치가 있고,
`medium/`에는 1-epoch fine-tuned checkpoint도 있다. 모든 checkpoint가 원본
`reduced.pt`의 `mlp_dims`를 유지하고 strict load되는 것을 확인했다.

## 7. 남은 단계

현재 medium checkpoint는 후보 선택용 1-epoch 결과이며 최종 배포 모델은 아니다.
다음 순서로 마무리한다.

1. 네 medium 모델의 resume 가능한 30-epoch W&B fine-tuning
2. best checkpoint의 독립 ImageNet validation
3. combined MAC과 실제 GPU/NPU latency 측정
4. static NHWC ONNX export와 금지 연산 검사
5. 실제 NPU compiler 및 장치 검증
