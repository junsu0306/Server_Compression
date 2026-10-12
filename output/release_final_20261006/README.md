# 최종 결과 배포 패키지

원본 실행: `output/final_20261006` (서버 단계 완료: 2026-10-08 UTC).
이 폴더는 학습 재개용 디렉터리가 아니라 최종 추론·평가·ONNX 전달용입니다.

- 모델별 `00_budget`: pretrained baseline, 추가 학습 없는 예산용 target, 예산/측정 JSON.
- `01_structural`: 정확히 30epoch의 raw+EMA 가중치, 30회 후보 메모리 측정, 지표.
- `02_reduced`: 실제 FFN 채널을 제거한 최종 모델.
- `03_patch_slimming_paper`: 검색 종료 가중치, 고정 패치 ID, subset ID, 30epoch 학습 중 best 가중치/지표.
- `04_onnx`: baseline/reduced/patch-slimming ONNX와 원본 수치 검증 JSON. 전체 12개.
- `05_memory`, `06_evaluation`, `reports`: 실제 측정/평가 및 최종 표.
- `configs`, `batch_probe`, `provenance`: 실행 설정, 선택 batch, 환경/실행 코드 hash.

`checkpoint_final.pt`와 `compact_finetuned_best.pt`에서 optimizer/scheduler/scaler 등
학습 재개용 상태만 제거했습니다. 원본 model/model_ema 텐서는 모든 원소가 동일함을 검사했습니다.
Structural checkpoint의 raw mask와 EMA를 둘 다 보존하므로 기존 reducing 코드로 읽을 수 있습니다.
다른 `.pt`와 ONNX는 원본 바이트를 그대로 복사했습니다. 학습은 재실행하지 않았습니다.

`terminal_checkpoint.json`, 예산/평가 JSON, 보고서의 경로·hash는 **실험 당시 원본 출처**입니다.
원본과 포장된 파일의 대응/hash는 `MANIFEST.json`의 source_sha256 / sha256을 사용하세요.
절대 경로는 실행 기록이지 다른 서버에 필요한 경로가 아닙니다. 모델 로더에는 이 폴더 경로를 직접 넘깁니다.
Git clone 후 `git lfs pull`을 수행하고 저장소 루트에서 다음 명령으로 전체 파일을 검증합니다.

```bash
python scripts/package_release.py --verify
python scripts/verify_release_models.py
```

최종 정확도는 PyTorch FP32 ImageNet validation 50,000장 평가입니다.
ONNX는 FP32 동등성 검사이며 MXQ INT8/NPU 정확도를 측정한 것이 아닙니다.
MXQ 변환 및 실제 NPU 정확도·속도·메모리 측정은 별도 `npu` 프로젝트에서 수행합니다.

원본 로그·optimizer·중간 search checkpoint·W&B 디렉터리·데이터·past는 Git에 포함하지 않습니다.
학습 재개에는 서버에 보존된 원본 실험 폴더가 필요합니다.
