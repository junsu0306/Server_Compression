# 08. Reconstruction error metric and dataset split correction

## 목적

논문 Algorithm 1 재현에서 다음 세 항목을 검증하고 수정한다.

1. layer `t`의 mask가 실제로 layer `t+1` 출력에서 판정되는지 확인
2. 논문의 epsilon을 element-wise MSE에 그대로 적용한 문제 수정
3. block reconstruction 학습과 acceptance 평가 데이터 분리

## 2026-08-06 코드 감사 결과

- `teacher_features()`와 `_student_prediction()`은 모두 block `t`와 `t+1`을 실행한다.
  기존 구현은 이미 다음 layer 출력을 target으로 사용했다.
- 이 동작을 고정하는 `test_teacher_target_is_next_block_output()` 회귀 테스트를 추가했다.
- 기존 acceptance metric은 `sum squared error / (B*N*D)`인 element-wise MSE였다.
- 논문의 squared Frobenius epsilon과 동일한 수치라는 근거가 없으므로 per-sample relative
  Frobenius squared를 추가했다.

## 기존 결과의 metric audit

epsilon=0.02 검색의 Block 10, 41-token 후보 기록을 다시 읽은 결과:

| Model | element-wise MSE | global relative Frobenius squared | MSE acceptance |
|---|---:|---:|---|
| structural Tiny-30 | 0.0178836 | 0.0681576 | pass |
| structural Small-30 | 0.0756741 | 0.0996033 | fail |
| structural Small-50 | 0.0486938 | 0.0836793 | fail |

Tiny-30 후보는 MSE 기준으로만 0.02를 통과하며 relative error 기준으로는 통과하지 않는다.
따라서 MSE epsilon을 논문의 epsilon과 직접 대응시킨 것이 과도한 token 제거의 실제 원인
후보임을 확인했다.

## 구현 변경

- `score_data`, `reconstruction_train`, `reconstruction_eval` 설정을 독립 지원한다.
- 세 subset의 ID를 각각 저장하고 교집합이 없음을 실행 시 검증한다.
- 이전 YAML에는 하나의 calibration loader를 공유하는 하위 호환 경로를 유지한다.
- 다음 metric을 모두 기록한다.
  - element-wise MSE
  - raw squared Frobenius
  - global relative squared Frobenius
  - per-sample relative squared Frobenius mean / p50 / p95

## Split pilot 결과

- 모델: `output/vit_tiny_30_final/reduced.pt`
- GPU: 6
- score/train/eval: 1,024 / 1,024 / 1,024, 상호 배타적
- acceptance: mean per-sample relative Frobenius squared <= 0.02
- W&B project: `vit-patch-slimming-relative-error-split`
- Search run: `5egv4mlb`

검색 결과 token schedule:

```text
[197, 131, 131, 131, 121, 121, 121, 121, 121, 121, 121, 121, 1]
```

Block 10에서 41-token 후보의 mean per-sample relative error는 0.07510으로 탈락했고,
121 tokens에서 0.01827로 통과했다. Block 2에서는 121 tokens의 error가 0.02003으로
근소하게 탈락하여 131 tokens가 선택됐다. 따라서 기존 41-token plateau가 완화되고 얕은
layer에서 더 많은 token을 유지하는 단계형 구조가 생겼다.

| Schedule | Structural baseline 대비 body MAC 감소 | Pre-FT Top-1 | Baseline 대비 하락 |
|---|---:|---:|---:|
| 기존 MSE eps=.02, `[41 x11]` | 81.40% | 41.86% | -31.67%p |
| 기존 manual medium | 34.75% | 66.08% | -7.45%p |
| relative eps=.02 + split | **42.59%** | **68.48%** | **-5.05%p** |

새 방식은 manual medium보다 body MAC을 7.84%p 더 줄이면서 fine-tuning 전 Top-1이
2.40%p 높다. 논문의 약 45% FLOPs anchor에도 근접한다. Pre-FT evaluation W&B run은
`cm326eum`이다.

## 해석

- 다음-layer target 구현은 plateau 원인이 아니었다.
- MSE epsilon의 직접 이식은 명백히 지나치게 공격적인 schedule을 만들었다.
- relative metric과 독립 eval split을 함께 적용한 결과 정확도-연산량 trade-off가 크게
  개선됐다.
- schedule이 layer마다 매번 감소하지는 않는다. nested mask에서 현재 base mask가 epsilon을
  만족하면 같은 token 수가 유지되므로 plateau 자체는 Algorithm 1의 정상 가능한 결과다.
- 이 pilot만으로 metric 변경 효과와 data split 효과를 분리할 수는 없다. 기존 동일 subset을
  relative metric으로 재검색하거나 split train 규모 ablation이 필요하다.

## 1-epoch recovery

기존 비교와 동일하게 `lr=2e-5`, `warmup=0`, batch 256으로 전체 ImageNet train을
1 epoch fine-tuning했다.

| 상태 | Top-1 | Top-5 | Structural baseline과 차이 |
|---|---:|---:|---:|
| Pre-FT | 68.482% | 88.778% | -5.048%p |
| 1 epoch | 68.354% | 88.840% | -5.176%p |

W&B run: `1bcxwtay`. 1 epoch 후 Top-1은 0.128%p 낮아져 이 짧은 recipe에는 recovery
효과가 없었다. 기존 manual medium의 1-epoch 68.42%와 비교하면 0.07%p 낮지만,
structural baseline 대비 body MAC 감소는 34.75%에서 42.59%로 7.84%p 더 크다.

잘못된 기본 warmup이 적용된 최초 recovery 시도는 첫 step 직후 중단했으며 결과로 사용하지
않는다. 위 수치는 명시적으로 warmup을 0으로 고친 재실행 결과다.

## 후속 실험

- reconstruction train 1k/10k 규모 ablation
- reference DeiT에서 논문 accuracy-FLOPs anchor 재현
- 현재 1-epoch recipe 대신 더 긴 recovery 또는 더 작은 LR 비교
