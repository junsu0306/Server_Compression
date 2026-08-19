# Patch Slimming documentation index

전체 학술적 배경, 실제 코드, 실험, checkpoint와 ONNX/NPU 변환을 한 문서에서 확인하려면
먼저 [`FINAL_IMPLEMENTATION_GUIDE.md`](FINAL_IMPLEMENTATION_GUIDE.md)를 읽는다.

문서는 목적에 따라 다음 세 층으로 구분한다.

## 1. 계획과 명세

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): 논문 알고리즘 해석, 구현 결정,
  tensor contract, 검색·학습·NPU export 요구사항을 기록한 최초 계획서

## 2. 전체 구현과 실험 결과

- [`EXPERIMENT_WORKFLOW_AND_RESULTS.md`](EXPERIMENT_WORKFLOW_AND_RESULTS.md): 프로젝트에서
  실제로 수행한 순서, 구현 변경, 전체 결과, 현재 선택과 다음 단계를 한 문서로 정리한
  주 문서

## 3. 단계별 상세 로그

- [`experiments/01_SMOKE_EXPERIMENT.md`](experiments/01_SMOKE_EXPERIMENT.md): 32장 smoke 검증
- [`experiments/02_DEIT_TINY_MAIN_EXPERIMENT.md`](experiments/02_DEIT_TINY_MAIN_EXPERIMENT.md):
  원본 pretrained DeiT-Tiny 2,048장 검색과 30 epoch 실험
- [`experiments/03_STRUCTURAL_MODELS_EXPERIMENT.md`](experiments/03_STRUCTURAL_MODELS_EXPERIMENT.md):
  실제 배포 대상 structural-pruned ViT 네 모델의 3·6·9 stage 실험
- [`experiments/04_PAPER_EPSILON_STRUCTURAL_EXPERIMENT.md`](experiments/04_PAPER_EPSILON_STRUCTURAL_EXPERIMENT.md):
  structural 모델에서 relaxed epsilon으로 수행하는 논문 Algorithm-1 비교 실험
- [`experiments/05_STATIC_TOKEN_SELECTION_AND_NPU_COST.md`](experiments/05_STATIC_TOKEN_SELECTION_AND_NPU_COST.md):
  static patch ID의 실제 forward, JSON-to-selection-matrix 변환, NPU 호환성과 추가 연산 비용 검토
- [`experiments/06_PATCH_SIGNIFICANCE_AND_GPU_PRUNING_IMPLEMENTATION.md`](experiments/06_PATCH_SIGNIFICANCE_AND_GPU_PRUNING_IMPLEMENTATION.md):
  path-energy 중요도 계산, production/paper mask 선택 차이, GPU compact pruning의 전체 코드 경로
- [`experiments/07_PAPER_EPSILON_002_STRUCTURAL_EXPERIMENT.md`](experiments/07_PAPER_EPSILON_002_STRUCTURAL_EXPERIMENT.md):
  structural 모델의 epsilon 0.02 논문식 search, validation, recovery 비교 실험
- [`experiments/08_ERROR_METRIC_AND_DATA_SPLIT_CORRECTION.md`](experiments/08_ERROR_METRIC_AND_DATA_SPLIT_CORRECTION.md):
  다음-layer target 회귀 검증, relative Frobenius metric, reconstruction train/eval 분리 및 재보정 실험
- [`experiments/09_PROGRESSIVE_TOKEN_SCHEDULE_EXPERIMENT.md`](experiments/09_PROGRESSIVE_TOKEN_SCHEDULE_EXPERIMENT.md):
  relative-error search에 점진적 token 축소 prior를 결합한 structural model 실험
- [`experiments/10_FINAL_12WAY_EXPERIMENT.md`](experiments/10_FINAL_12WAY_EXPERIMENT.md):
  reduced 네 모델과 paper/progressive-3/progressive-5 정책의 최종 12-way 검색·fine-tuning

`EXPERIMENT_WORKFLOW_AND_RESULTS.md`를 먼저 읽고, 재현이나 디버깅에 더 자세한 값이
필요할 때 단계별 로그를 참조한다.
