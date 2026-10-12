# 최종 실험 결과

상태: `complete` · W&B: `vit-final-target-memory-20261006`

이 문서는 현재 실험의 원본 JSON에서 자동 생성됩니다. 미완료 항목은 결과로 제시하지 않습니다.

Structural 학습 종료: 30 epoch. 후속 단계는 정확히 해당 epoch의 checkpoint_final.pt를 사용합니다.
LR schedule horizon: 50 epoch (학습 종료 epoch와 별도).

## 메모리 예산

학습 없이 baseline을 global L2 pruning/reducing한 목표 구조의 CUDA peak를 M_max로 고정합니다.

## 사전 예산 및 후보 측정

| model | baseline_peak_mib | target_peak_mib | target_trained_steps | target_parameters | candidate_measurements | candidate_max_mib |
| --- | --- | --- | --- | --- | --- | --- |
| tiny30 | 32.8311 | 26.1523 | 0 | 4003396 | 30 | 26.1523 |
| tiny50 | 32.8311 | 21.8018 | 0 | 2862256 | 30 | 21.8018 |
| small30 | 97.1577 | 71.5552 | 0 | 15434188 | 30 | 71.5552 |
| small50 | 97.1577 | 54.3721 | 0 | 11023204 | 30 | 54.3721 |

## FP32 ImageNet 검증

| model | stage | parameters | weight_mib | top1 | top5 | samples |
| --- | --- | --- | --- | --- | --- | --- |
| tiny30 | baseline | 5717416 | 21.8102 | 74.6600 | 92.4940 | 50000 |
| tiny30 | pruned | 5717416 | 21.8102 | 72.2120 | 91.1680 | 50000 |
| tiny30 | reduced | 4003396 | 15.2717 | 72.2120 | 91.1680 | 50000 |
| tiny30 | patch_slimming | 4003396 | 15.2717 | 67.9020 | 88.1300 | 50000 |
| tiny50 | baseline | 5717416 | 21.8102 | 74.6600 | 92.4940 | 50000 |
| tiny50 | pruned | 5717416 | 21.8102 | 63.2300 | 85.8320 | 50000 |
| tiny50 | reduced | 2862256 | 10.9186 | 63.2300 | 85.8320 | 50000 |
| tiny50 | patch_slimming | 2862256 | 10.9186 | 59.2520 | 82.2700 | 50000 |
| small30 | baseline | 22050664 | 84.1166 | 80.5020 | 95.8620 | 50000 |
| small30 | pruned | 22050664 | 84.1166 | 79.6240 | 95.1620 | 50000 |
| small30 | reduced | 15434188 | 58.8768 | 79.6240 | 95.1620 | 50000 |
| small30 | patch_slimming | 15434188 | 58.8768 | 77.9700 | 94.1900 | 50000 |
| small50 | baseline | 22050664 | 84.1166 | 80.5020 | 95.8620 | 50000 |
| small50 | pruned | 22050664 | 84.1166 | 74.0020 | 92.2340 | 50000 |
| small50 | reduced | 11023204 | 42.0502 | 74.0020 | 92.2340 | 50000 |
| small50 | patch_slimming | 11023204 | 42.0502 | 70.7360 | 90.1240 | 50000 |

## 서버 메모리

| model | stage | weight_mib | peak_allocated_mib | transient_peak_mib | module_output_sum_mib | latency_mean_ms |
| --- | --- | --- | --- | --- | --- | --- |
| tiny30 | baseline | 21.8102 | 32.8311 | 2.2998 | 56.2758 | 7.6311 |
| tiny30 | reduced | 15.2717 | 26.1523 | 2.1558 | 42.8932 | 8.6098 |
| tiny30 | patch_slimming | 15.2717 | 27.3125 | 2.6875 | 19.3636 | 23.7980 |
| tiny50 | baseline | 21.8102 | 32.8311 | 2.2998 | 56.2758 | 11.0891 |
| tiny50 | reduced | 10.9186 | 21.8018 | 2.1558 | 33.9834 | 7.7578 |
| tiny50 | patch_slimming | 10.9186 | 22.8862 | 2.6118 | 14.5954 | 20.4542 |
| small30 | baseline | 84.1166 | 97.1577 | 4.0293 | 112.5478 | 9.6904 |
| small30 | reduced | 58.8768 | 71.5552 | 3.4561 | 86.6843 | 8.8645 |
| small30 | patch_slimming | 58.8768 | 74.0703 | 5.0439 | 46.8540 | 22.1400 |
| small50 | baseline | 84.1166 | 97.1577 | 4.0293 | 112.5478 | 8.7894 |
| small50 | reduced | 42.0502 | 54.3721 | 3.3086 | 69.4420 | 9.9670 |
| small50 | patch_slimming | 42.0502 | 56.6802 | 4.8901 | 32.6188 | 22.1218 |

## ONNX 산출물

| model | stage | file | size_mib | pytorch_ort_max_abs | sha256 |
| --- | --- | --- | --- | --- | --- |
| tiny30 | baseline | tiny30/04_onnx/tiny30__baseline__nhwc_b1_npusafe.onnx | 21.8890 | 0.0000 | 0d8655a8e7641e9023c6464f7c97ca172b447709f5c2b3abb6c91d83a19b156b |
| tiny30 | reduced | tiny30/04_onnx/tiny30__reduced__nhwc_b1_npusafe.onnx | 15.3505 | 0.0000 | bc628a54be653609a895d279eb43826827eeaaddc3cf8ea23c3855108551cff6 |
| tiny30 | patch_slimming | tiny30/04_onnx/tiny30__patch_slimming__nhwc_b1_npusafe.onnx | 15.6211 | 0.0000 | 08d4014de88f6f156dd88e312e240bf71e3aff0af473b2fd33aba03d7e5a5fa1 |
| tiny50 | baseline | tiny50/04_onnx/tiny50__baseline__nhwc_b1_npusafe.onnx | 21.8890 | 0.0000 | 0d8655a8e7641e9023c6464f7c97ca172b447709f5c2b3abb6c91d83a19b156b |
| tiny50 | reduced | tiny50/04_onnx/tiny50__reduced__nhwc_b1_npusafe.onnx | 10.9974 | 0.0000 | aa122558ccb0ea5b3cd081d6b8ed1129948edb95bcbcb6e1a63753859b9fe169 |
| tiny50 | patch_slimming | tiny50/04_onnx/tiny50__patch_slimming__nhwc_b1_npusafe.onnx | 11.2680 | 0.0000 | 90086de3e74482a4fdbc25b0065e6d2dce9895cb1e417aa9af63d3f77302f65c |
| small30 | baseline | small30/04_onnx/small30__baseline__nhwc_b1_npusafe.onnx | 84.1954 | 0.0000 | 9812da2a85262ee515dfb7947f5ced1186a0f447005e7a7988c6bb996726cdc0 |
| small30 | reduced | small30/04_onnx/small30__reduced__nhwc_b1_npusafe.onnx | 58.9555 | 0.0000 | 1bf7f5fade8b32de735151f01f714e63d98e815ef9204fc2562ff3e978bad5cd |
| small30 | patch_slimming | small30/04_onnx/small30__patch_slimming__nhwc_b1_npusafe.onnx | 59.2731 | 0.0000 | 0146dde675d79f36d95529ddda67ace59dca3a7a5f718fb02f4039716ab3b6a1 |
| small50 | baseline | small50/04_onnx/small50__baseline__nhwc_b1_npusafe.onnx | 84.1954 | 0.0000 | 9812da2a85262ee515dfb7947f5ced1186a0f447005e7a7988c6bb996726cdc0 |
| small50 | reduced | small50/04_onnx/small50__reduced__nhwc_b1_npusafe.onnx | 42.1290 | 0.0000 | 0caf8659b7232bedff19e42a2f798b8b508338a80c8381047630d422671183fc |
| small50 | patch_slimming | small50/04_onnx/small50__patch_slimming__nhwc_b1_npusafe.onnx | 42.9447 | 0.0000 | ec63a02ee86783a268a4cfb0f001aa16fdbfb549128a9a2c8d0d794952789ce3 |

## 측정 해석

- 정확도는 ImageNet validation 50,000장 FP32 재평가이며 학습 중 AMP/EMA best 지표와 구분합니다.
- Pruned는 dense shape에 zero mask를 적용한 모델, reduced는 해당 채널을 물리 제거한 모델입니다.
- 메모리는 batch 1, FP32, random input, warmup 10회 후 forward 50회 중 allocator 최대값입니다. 평균 peak가 아닙니다.
- Module output 합은 별도 1회 forward의 출력 텐서 용량 합이며 동시에 살아 있는 메모리의 peak가 아닙니다.
- 메모리 초과 패널티는 측정 기반 상수 항입니다. gradient나 자동 추가 pruning을 제공하지 않습니다.
- ONNX verification은 FP32 수치 동등성 확인입니다. MXQ/NPU 정확도는 별도 장비 검증 대상입니다.
