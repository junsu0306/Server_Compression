# 최종 결과 읽기

수치는 [최종 실험 결과표](../output/release_final_20261006/reports/FINAL_RESULTS.md)에서 확인합니다. 2026-10-08 UTC에 서버 전체 단계가 완료됐습니다. 이 문서는 지표 정의와 산출물 위치를 설명합니다. 새 실험의 Supervisor는 진행 중에도 완료된 측정만 반영한 보고서를 갱신합니다.

Git에는 `output/release_final_20261006/`의 최종 가중치 24개, ONNX 12개와 측정 증거만 포함합니다. 아래 원본 실행 구조 전체는 로컬에만 보존합니다. 배포본에는 optimizer/stdout/W&B/중간 search checkpoint가 없으며, 자세한 목록과 hash 대응은 [Git 배포 안내](GIT_RELEASE.md)와 [MANIFEST](../output/release_final_20261006/MANIFEST.json)에 있습니다.

## 모델별 저장 구조

```text
output/final_20261006/<tiny30|tiny50|small30|small50>/
├── 00_budget/
│   ├── baseline.pt                 # 본 학습 초기 pretrained 가중치
│   ├── target_untrained.pt         # 예산용, 추가 학습 0 step
│   ├── baseline_profile.json
│   ├── target_profile.json
│   └── budget.json                 # 고정 M_max와 출처/해시
├── 01_structural/
│   ├── checkpoint_{best,last}.pt
│   ├── checkpoint_final.pt         # 정확히 structural 30epoch 완료 모델
│   ├── terminal_checkpoint.json    # 선택 epoch와 SHA-256
│   ├── train.log
│   └── memory_budget/epoch_*.json  # 후보 사본의 reducing 후 peak
├── 02_reduced/reduced.pt
├── 03_patch_slimming_paper/
│   ├── architecture.json / searched_student.pt
│   ├── compact_finetuned_{best,last}.pt
│   ├── *_ids.txt                   # 통계·복원·판정 subset 식별자
│   └── search.log / finetune.log
├── 04_onnx/                        # 3종 ONNX + parity JSON
├── 05_memory/                      # 3종 서버 memory/activation JSON
├── 06_evaluation/                  # 4종 FP32 정확도 JSON
├── batch_probe/                    # 후보 batch, 처리량, 여유 메모리
└── configs/                        # 실제 단계별 입력 YAML
```

## 표의 의미

| 표 | 주요 값 | 해석 |
|---|---|---|
| budget.csv | baseline peak, target peak=M_max, candidate max | 전부 같은 FP32 추론 지표. 후보가 예산을 초과할 수 있음 |
| accuracy.csv | parameters, weight MiB, Top-1/5, samples | validation 50,000장 FP32 재평가 |
| memory.csv | weight, peak allocated, transient peak, module output sum | peak와 activation 출력 합을 구분 |
| onnx.csv | 파일 크기, 최대 절대오차, SHA-256 | FP32 그래프 parity이며 MXQ 성능 아님 |

Pruned 모델은 FFN tensor에 zero mask를 둔 dense 구조이므로 파라미터 저장량은 baseline과 같습니다. Reduced가 zero 채널을 실제 제거합니다. Patch Slimming은 토큰 흐름을 줄이는 것으로 FFN weight shape를 추가로 줄이는 것은 아닙니다.

Structural 결과는 best checkpoint가 아니라 **30epoch terminal checkpoint** 기준입니다. Pruned와 reduced는 이 동일한 weight/mask에서 파생되며 Patch Slimming은 해당 reduced 모델을 입력으로 사용합니다. LR horizon 50epoch, 실제 학습 종료 30epoch을 별도 설정으로 표시합니다.

Patch Slimming은 30epoch fine-tuning을 완료한 후 학습 중 AMP validation best를 선택해 FP32로 다시 평가합니다. 선택된 완료 epoch는 Tiny30=28, Tiny50=27, Small30=29, Small50=29입니다. 따라서 patch best와 마지막 30epoch 모델은 동일하지 않을 수 있습니다.

이번 결과에서 Patch Slimming Top-1은 reduced보다 1.654~4.310%p 낮습니다. 서버 모듈 출력 합은 감소했지만 CUDA peak와 latency는 reduced보다 증가했습니다. 출력 합을 live activation peak로, ONNX FP32 결과를 MXQ/NPU 성능으로 바꾸어 해석하지 않습니다. 메모리 패널티는 4모델 모두 예산 초과가 없어 0이었습니다.

`weight MiB = parameter_count × 4 / 2^20`입니다. 체크포인트 파일 크기에는 optimizer 등 부가 정보가 포함될 수 있고, ONNX 파일에는 상수 선택행렬이 포함될 수 있으므로 세 값은 서로 다릅니다.

메모리 peak는 warmup 10회 이후 50 forward 중 CUDA allocator 최대값입니다. 시간은 50회 평균입니다. Module output sum은 별도 forward 1회의 leaf module 출력 합이며 동일 텐서의 재사용/생존 시간을 반영하는 live peak가 아닙니다. Attention 중간값은 hook이 아니라 shape 기반 분석에 포함됩니다.

Memory JSON의 token_schedule은 초기 토큰 수와 12개 block **출력** 토큰 수를 포함한 길이 13 배열입니다. 12개 block 입력표는 이 배열의 마지막 원소를 제외한 12개입니다. 마지막 출력 1은 CLS이며 마지막 block의 K/V가 반드시 1개라는 뜻은 아닙니다.

측정/학습이 미완료면 수치가 없습니다. `in_progress` 표를 최종 완료 결과로 제출하지 않습니다. 모든 worker의 정상 종료와 보고서 `state=complete`, 12개 ONNX 검증 hash를 함께 확인해야 합니다.
