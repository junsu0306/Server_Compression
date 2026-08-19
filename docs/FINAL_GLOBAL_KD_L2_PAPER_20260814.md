# Final Global-KD L2 + Paper Patch Slimming Experiment

## 목적

Progressive Taylor 결과는 사용하지 않는다. ViT-Tiny/Small에 global L2 channel
pruning과 KD를 적용하여 각각 30%, 50% physical reducing을 만들고, 각 reduced 모델에
paper 정책 Patch Slimming을 적용한다. 마지막에는 baseline/reduced/Patch Slimming 세
단계 ONNX, 메모리·activation 및 정확도 결과를 비교한다.

## 고정 설정

- GPU 6: Tiny30 → Tiny50
- GPU 7: Small30 → Small50
- W&B: `vit-final-global-kd-l2-paper-20260814`
- Structural: 50 epochs, global ranking, L2 importance, no pruning ramp
- KD: alpha 0.5, temperature 4; 기존 global-KD 설정대로 동일 아키텍처 teacher
  (Tiny→Tiny, Small→Small). `kd_small`/`kd_base` 별도 확장 실험은 사용하지 않는다.
- Patch Slimming: paper path-energy, epsilon 0.02, sample-relative Frobenius
- Block fine-tuning: candidate마다 3 epochs
- Final Patch Slimming recovery: 30 epochs
- 출력: `output/final_global_kd_l2_paper_20260814/`

## 실행

```bash
python scripts/run_final_global_kd_l2.py --phase materialize

python scripts/run_final_global_kd_l2.py --phase structural --gpu 6
python scripts/run_final_global_kd_l2.py --phase structural --gpu 7

python scripts/run_final_global_kd_l2.py --phase patch --gpu 6
python scripts/run_final_global_kd_l2.py --phase patch --gpu 7

python scripts/run_final_global_kd_l2.py --phase export --gpu 6
python scripts/run_final_global_kd_l2.py --phase export --gpu 7

python scripts/run_final_global_kd_l2.py --phase memory --gpu 6
python scripts/run_final_global_kd_l2.py --phase memory --gpu 7
```

각 명령은 checkpoint를 검사하고 완료된 단계를 건너뛴다. `all`은 해당 GPU 큐의
structural과 Patch Slimming을 연속 수행한다.

ONNX export, 메모리 측정 및 최종 결과 표는 학습 완료 후 이 문서에 이어서 기록한다.

## ONNX 3종 export 규칙

세 단계 모두 fixed batch 1, NHWC `[1,224,224,3]`, opset 17로 통일하고 PyTorch와
ONNX Runtime 최대 절대오차가 `1e-4` 미만인지 검증한다.

```bash
python scripts/export_final_onnx.py --kind baseline \
  --model vit_tiny_patch16_224 --output baseline.onnx

python scripts/export_final_onnx.py --kind reduced \
  --reduced reduced.pt --output reduced.onnx

python scripts/export_final_onnx.py --kind patch \
  --arch architecture.json --weights compact_finetuned_best.pt \
  --output patch_slimming.onnx
```

모든 단계 완료 후 최종 표를 생성한다.

```bash
python scripts/build_final_global_kd_l2_report.py
```

결과는 실험 루트의 `FINAL_RESULTS.json`과 `FINAL_RESULTS.md`에 저장된다.
