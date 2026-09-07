# Final Global-L2 Stage-3/6/9 Relaxed Patch Slimming

## 목적

최종 Global-L2 + KD reduced 모델 네 개를 그대로 사용하고, paper 자유검색 Patch
Slimming보다 완화된 정적 stage 정책을 비교한다. 1-based layer 3, 6, 9에서만 token을
줄여 NPU selection/quantization boundary를 줄이고 정확도를 더 보존한다.

## 고정 설정

- 시작 checkpoint: `output/final_global_kd_l2_paper_20260814/*/02_reduced/reduced.pt`
- 정책: layer 3/6/9 이후 `[171, 151, 131]` tokens
- block 출력 schedule: `[197,197,171,171,171,151,151,151,131,131,131,131,1]`
- token ID: 모델별 paper path-energy score로 독립 산출
- score sample: ImageNet train 2,048장
- final fine-tuning: ImageNet train 30 epochs, AdamW, LR `2e-5`, warmup 3 epochs
- W&B: `vit-final-global-kd-l2-stage369-relaxed-20260826`
- GPU 6 queue: Tiny30 → Tiny50
- GPU 7 queue: Small30 → Small50
- output: `output/final_global_kd_l2_stage369_relaxed_20260826/`

## NPU-safe export 변경

- identity selector는 MatMul을 만들지 않고 입력 tensor를 직접 반환한다.
- 마지막 CLS-only 선택은 one-hot MatMul 대신 static Slice로 export한다.
- 실제 scattered token 감소 세 지점만 constant selection MatMul을 사용한다.
- Gather/TopK는 사용하지 않는다.

## 파인튜닝 전 결과

| Model | Reduced Top-1 | Relaxed Top-1 | Drop | Top-5 | Transformer MAC 감소 |
|---|---:|---:|---:|---:|---:|
| Tiny30 | 72.800% | 68.576% | -4.220%p | 88.918% | 26.905% |
| Tiny50 | 64.954% | 60.702% | -4.252%p | 83.894% | 26.968% |
| Small30 | 80.218% | 78.186% | -2.032%p | 94.280% | 26.974% |
| Small50 | 75.448% | 72.268% | -3.180%p | 91.114% | 25.169% |

## 30-epoch 최종 결과

| Model | Best Top-1 | 해당 validation Top-5 | Reduced 대비 Top-1 | Paper Patch Top-1 |
|---|---:|---:|---:|---:|
| Tiny30 | 70.942% | 90.40% | -1.858%p | 69.386% |
| Tiny50 | 63.862% | 85.58% | -1.092%p | 60.384% |
| Small30 | 79.014% | 94.57% | -1.204%p | 78.678% |
| Small50 | 74.152% | 92.13% | -1.298%p | 73.330% |

네 모델 모두 epoch 29까지 완료했으며 `compact_finetuned_best.pt`,
`compact_finetuned_last.pt`와 NPU-safe ONNX를 저장했다. ONNX는 checker와 ORT 비교를
통과했고 Gather/TopK 및 identity selection MatMul이 없다. 최종 네 파일은
`/workspace/etri_iitp/JS/npu/assets/onnx/`에도 복사했다.

## 실행

```bash
python scripts/run_stage369_relaxed.py --phase materialize
python scripts/run_stage369_relaxed.py --phase all --gpu 6
python scripts/run_stage369_relaxed.py --phase all --gpu 7
```

실행기는 mask, last checkpoint 및 ONNX 존재 여부를 확인하며 중단된 final fine-tuning은
`compact_finetuned_last.pt`에서 재개한다.
