# 최종 구현 방법론

## 1. 대상과 전체 흐름

대상은 timm `vit_tiny_patch16_224`, `vit_small_patch16_224`의 전체 파라미터 30%/50% 압축 조합 4개입니다. RGB 224×224 입력을 16×16 patch로 나누어 196개 patch와 CLS 1개, 총 197개 토큰으로 시작합니다. 두 모델 모두 12 block이며 embedding은 Tiny 192, Small 384입니다.

```text
Pretrained baseline 저장 및 FP32 추론 메모리 측정
    ↓
별도 사본: 목표 비율 global L2 pruning → reducing (추가 학습 0 step)
    ↓
별도 프로세스 CUDA peak 측정 → 고정 M_max
    ↓
저장한 baseline에서 본 pruning + logit KD 30 epoch
    ├─ optimizer update → mask 재적용 → EMA 갱신
    └─ 후보/EMA 사본 reducing → 추론 peak → 초과분 → 다음 batch loss
    ↓
30 epoch terminal checkpoint 물리적 reducing
    ↓
통계적 정적 Patch Slimming: 후보별 block reconstruction 3 epoch
    ↓
고정 ID compact 모델 CE fine-tuning 30 epoch
    ↓
3종 ONNX × 4실험 + FP32 parity + 정확도·메모리 평가 + 보고서
```

설정 원본은 `configs/final.yaml`, 실행기는 `scripts/run_pipeline.py`입니다.

## 2. 사전 예산 M_max

코드: `scripts/profiling/prepare_budget.py`의 `build_target()` 및 `main()`.

Pretrained baseline을 `00_budget/baseline.pt`에 저장합니다. 그 deep copy를 목표 비율로 global L2 pruning하고 작은 FC1/FC2로 바꿔 `target_untrained.pt`를 만듭니다. 이 과정에는 optimizer update, 데이터 학습, KD가 없습니다. `trained_steps=0`은 pretrained 가중치 자체가 미학습이라는 뜻이 아니라 **이번 압축 후 추가 학습이 0회**라는 뜻입니다.

예산용 target은 성능 비교용 최종 reduced 모델이 아닙니다. 목표 구조를 사전에 만들어 메모리를 측정하는 용도이며, 학습 완료된 reduced 모델을 입력으로 요구하지 않습니다.

`M_max = target_profile.peak_allocated_bytes / 2^20`.

`budget.json`에는 baseline/target 경로·SHA-256, 모델명, 목표 감소율, 측정 조건, 원본 profile과 M_max가 저장됩니다. `MemoryBudget`은 출처·설정·해시를 검증합니다. 본 학습 중 예산을 재산정하지 않습니다.

파라미터 감소율이 같아도 레이어별 채널 배치·activation·임시 버퍼에 따라 peak가 다를 수 있습니다. 따라서 30% 파라미터 감소를 baseline peak의 30% 감소로 치환하지 않습니다.

## 3. Global L2 FFN pruning

코드: `pruning/vit_pruning.py:ViTPruner`.

FFN은 `D → M → D` 구조입니다. Hidden channel j의 점수는 `s_j = ||W_fc1[j,:]||_2`입니다. 모든 block의 점수를 함께 비교해 낮은 채널부터 제거하며 block별 sparsity 상한 0.95, 최소 생존 채널 4개를 보장합니다.

한 채널은 FC1 weight의 한 행(D개), FC1 bias 한 값(1개), FC2 weight의 대응 열(D개)을 함께 마스킹합니다. 따라서 표준 ViT의 채널당 제거 파라미터는 `2D+1`개입니다. Attention, embedding 차원, classifier 출력 차원은 유지합니다.

목표 30%/50%는 FFN만의 비율이 아니라 **전체 모델 파라미터 수 대비 제거 비율**입니다. 이진 탐색과 정수 채널 반올림으로 실제 비율은 목표와 소폭 다를 수 있습니다.

본 학습 전에 마스크를 적용하고 EMA에도 같은 zero 패턴을 이식합니다. Optimizer update마다 마스크를 재적용하며 ranking은 100 step마다 갱신합니다. 이 단계에서는 dense tensor shape와 저장 파라미터 수는 유지됩니다.

## 4. KD와 메모리 항

코드: `pruning/engine.py:train_one_epoch`, `pruning/memory_budget.py:MemoryBudget`.

Teacher는 같은 아키텍처의 pretrained baseline이며 고정합니다. 관계형 KD가 아니라 logit 분포의 KL distillation입니다.

```text
L_task = (1−α) CE(z_student, y)
         + α T² KL(softmax(z_teacher/T) || softmax(z_student/T))
L_mem  = λ max(0, M_candidate − M_max)
L      = L_task + L_mem
```

α=0.5, T=4, λ=1.0, CE smoothing=0.1입니다. **학습 종료는 30 epoch**이며 AdamW, LR=5e-5, weight decay=0.05, warmup 5 epoch, AMP, EMA decay=0.9998을 사용합니다. Synthetic batch probe가 여유 메모리 조건을 만족하는 후보 중 빠른 batch를 선택하고 로그를 보존합니다.

학습률 스케줄 기준 길이 `lr_schedule_epochs=50`은 학습 종료 `epochs=30`과 분리합니다. Warmup 이후 cosine의 T_max는 45이며, 30 epoch에서 최소 LR까지 decay하는 recipe가 아닙니다. Resume 시 저장된 optimizer/scheduler 상태를 유지하고 horizon 불일치는 오류로 처리합니다.

메모리 측정은 본 학습 모델을 변경하지 않습니다. Raw/EMA state를 CPU snapshot으로 복사하여 자식 프로세스에 전달합니다. 자식은 raw mask를 EMA에 이식하고 reducing한 후보만 GPU에 올려 FP32 추론 peak를 측정합니다. 임시 snapshot만 측정 후 제거하고 profile JSON은 남깁니다.

초기 후보를 측정한 뒤 매 epoch 갱신하며 초과량은 다음 batch부터 적용합니다. 실패한 측정을 0으로 처리하지 않고 오류로 중단합니다.

**측정값 기반 상수 항이므로 weight gradient가 없으며 자동 추가 pruning 기능도 없습니다.** 초과분을 loss/로그에 반영하지만 예산 만족을 강제하는 최적화라고 주장하지 않습니다.

## 5. Reducing

코드: `pruning/reduce.py`, `pruning/vit_reducing.py`.

Raw FC1 row norm이 정확히 0인 채널을 제거합니다. EMA에는 raw zero 패턴을 먼저 이식하고 살아 있는 EMA weight를 보존합니다. 새로운 작은 `nn.Linear`로 FC1 행/bias, FC2 열을 물리적으로 줄입니다.

`reduced.pt`에 state_dict, model_name, 12개 mlp_dims, 전후 파라미터 수, 실제 감소율을 저장합니다. Reload는 baseline 구조에 mlp_dims를 적용한 뒤 state_dict를 읽습니다. CPU 테스트는 masked/reduced 출력의 수치 동등성을 확인합니다.

후속 reducing에는 30번째 epoch의 학습과 validation이 끝난 `checkpoint_last.pt`를 `checkpoint_final.pt`로 고정해 사용합니다. 체크포인트의 zero-based `epoch`는 29여야 합니다. Best epoch나 31epoch 이후의 모델로 대신하지 않습니다. SHA-256과 선택 근거를 `terminal_checkpoint.json`에 보존하며 pruned FP32 평가도 같은 final checkpoint를 사용합니다.

## 6. Statistical Patch Slimming

### 통계 데이터

ImageNet train에서 겹치지 않는 부분집합 3개를 뽑고 sample index/경로를 저장합니다. 현재 사용 가능한 train은 1,280,297장, validation은 50,000장입니다.

| 용도 | 수량 | seed |
|---|---:|---:|
| Significance score | 2,048장 | 2022 |
| Reconstruction train | 4,096장 | 2023 |
| Reconstruction acceptance eval | 2,048장 | 2024 |

코드: `patch_slimming/psvit/data.py`, `scoring.py`, `scripts/search.py`.

Score는 현재 block attention/feature energy와 downstream attention 경로의 영향을 결합합니다. Head-path 집계는 `G ← mean_h(P_hᵀ G P_h)`의 역방향 동적 계획법으로 계산하고 작은 예제의 모든 경로 열거와 일치하는지 검증합니다. 이미지별 score의 평균으로 고정 patch ID ranking을 얻습니다. 서로 다른 layer의 score 절대값은 비교하지 않습니다.

### Top-down 탐색과 복원

마지막 block 출력은 CLS 하나로 고정하고 index 10→0 순서로 탐색합니다. 현재 mask는 다음 layer의 mask를 포함합니다. 이 집합에서 시작해 중요도가 높은 patch를 10개씩 추가합니다. CLS를 항상 유지하며 nested 관계를 검사합니다. 매 layer의 강제 감소량은 없으므로 plateau가 허용됩니다.

후보마다 현재 block만 AdamW LR=1e-5로 3 epoch 학습합니다. 다음 block의 weight는 고정하되 gradient는 통과시킵니다. Teacher는 **Patch Slimming 전 reduced 모델의 고정 사본**입니다. Target과 prediction은 모두 다음 block 출력의 대응 토큰입니다.

학습 loss는 element-wise MSE입니다. 후보 채택은 독립 eval subset에서 다음을 사용합니다.

`E = mean_b( ||pred_b−target_b||_F² / max(||target_b||_F², 1e-12) )`.

E≤0.02이면 채택합니다. 전체 토큰에서도 초과하면 `all_tokens` fallback으로 기록하며 ε 통과로 표현하지 않습니다. 후보 간 block weight는 누적하고 optimizer는 후보마다 새로 만듭니다. 3 epoch은 전체 ImageNet이 아니라 지정한 reconstruction 4,096장의 epoch입니다.

Top-down 구조와 다음 block 복원 판정은 논문 기반입니다. Relative 정규화, 부분집합 규모, 구체적 학습 recipe는 프로젝트의 명시적인 구현 선택입니다. 논문의 raw Frobenius ε와 숫자가 직접 동등하다고 주장하지 않습니다.

### Compact forward와 최종 학습

코드: `psvit/slim_block.py`, `compact.py`, `scripts/finetune.py`.

각 block의 K/V는 입력 N_in개에서, Q와 FFN은 선택된 N_out개에서 계산합니다. Attention은 N_out×N_in이며 residual도 대응 행을 선택합니다. 단순 zero mask가 아니라 실제 tensor shape가 `[B,N_in,D] → [B,N_out,D]`로 줄어듭니다.

구조는 `architecture.json`, 검색 가중치는 `searched_student.pt`에 저장합니다. 고정 token ID의 compact 모델을 CE로 30 epoch fine-tuning합니다. LR=2e-5, AdamW, weight decay=0.05, warmup=3, smoothing=0.1이며 추가 KD는 없습니다.

## 7. ONNX

코드: `scripts/export_final_onnx.py`.

- FP32 NHWC `[1,224,224,3]` 입력, `[1,1000]` 출력, opset 17.
- GPU 학습은 고정 ID의 index_select, ONNX는 고정 0/1 선택행렬 P의 MatMul로 행 선택.
- 동일 token 집합은 identity selection 생략, CLS-only 선택은 static Slice.
- Batch 1 CLS를 직접 연결하고 최종 CLS는 Slice/Squeeze로 추출.
- Gather/GatherElements/GatherND/TopK/Equal/Where/Expand가 남으면 검증 실패.
- ONNX checker와 random input 3개에서 eager/변환 wrapper/ORT FP32 출력 최대 절대오차 <1e-4 검증.

선택행렬은 학습 파라미터는 아니지만 ONNX 상수 용량에는 영향을 줍니다. 4실험×3종=12개를 생성합니다. FP32 parity와 MXQ 양자화/NPU 정확도 검증은 별개입니다.

## 8. 측정

실행기: `scripts/profiling/profile_reduced_memory.py`. 새 프로세스, FP32 eval/inference_mode, batch 1, 224×224 random input(seed 2022), warmup 10회, 측정 forward 50회입니다.

Warmup 후 peak를 reset하고 `torch.cuda.max_memory_allocated()`를 읽습니다. 이는 50회 중 최대 allocator 사용량이지 평균이 아닙니다. 이전 forward의 logits를 다음 반복까지 유지하지 않습니다. Parameter/buffer bytes, resident, transient peak, reserved peak와 프로세스 RSS/USS/PSS를 구분합니다. 다른 프로세스의 allocator 메모리는 더해지지 않지만 GPU 경쟁은 성능과 측정 안정성에 영향을 줄 수 있습니다.

Peak 측정 후 hook을 연결한 별도 forward 1회로 leaf module 출력 shape/bytes를 측정합니다. 출력 합은 live peak가 아닙니다. Functional attention 텐서는 block shape로 별도 분석합니다. FFN hidden activation은 `B × N_out × M × 4 bytes`입니다. Baseline/reduced의 fused attention과 compact의 explicit rectangular attention 차이 때문에 token 감소율이 peak 감소율과 일치하지 않을 수 있습니다.

정확도는 `scripts/evaluation/evaluate.py`로 validation 50,000장, timm 권장 전처리, FP32/AMP off에서 평가합니다. Baseline, masked pruned(EMA+mask), reduced, patch-slimmed를 포함합니다. 학습 중 AMP best 지표와 최종 FP32 재평가는 구분합니다.

## 9. 검증과 증적

예산 출처/해시, 상수 loss의 gradient 불변, 사본 격리, masked/reduced 등가성, nested ID, rectangular attention, DP/열거 일치, 다음 block 복원 위치, ONNX wrapper를 테스트합니다. 실행 시 코드/설정을 `provenance/source/`에 복사하고 SHA-256 목록을 남깁니다. 원본 JSON이 있는 측정만 보고서에 반영하고 미완료를 완료로 표시하지 않습니다.
