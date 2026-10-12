# Output 구분

- [release_final_20261006](release_final_20261006/README.md): Git에 포함되는 최종 가중치·ONNX·결과 증거. `.pt`와 `.onnx`는 Git LFS.
- `final_20261006/`: 서버 원본 실험. optimizer, 중간 checkpoint, stdout, W&B를 포함하므로 Git에서 제외.
- 나머지 output 폴더는 기본적으로 Git에서 제외됩니다. 새 파일이 실수로 업로드되지 않도록 release 내부도 파일별 허용 목록을 사용합니다.

원본은 로컬에 그대로 보존됩니다. 배포본은 추론/검증용이며 학습 중단 지점부터의 재개는 원본이 필요합니다.
실제 MXQ 변환 및 NPU 벤치마킹은 별도 `npu` 프로젝트에서 수행합니다.
