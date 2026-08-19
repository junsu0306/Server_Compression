# 10. Final 12-way structural Patch Slimming experiment

## 실험 행렬

배포 대상 reduced model 네 개와 정책 세 개의 Cartesian product 12개를 실행한다.

- Models: Tiny-30, Tiny-50, Small-30, Small-50
- Policies:
  - `paper`: top-down reconstruction search, progressive prior 없음
  - `progressive_g3`: shallower block마다 path-energy 상위 token 최소 3개 복원
  - `progressive_g5`: shallower block마다 path-energy 상위 token 최소 5개 복원

여기서 `paper`는 공개 논문의 미기재 error normalization을 임의로 raw epsilon과 동일시하지
않는다. 세 정책 모두 corrected mean per-sample relative Frobenius squared epsilon 0.02를
사용하며, `paper`는 그중 Algorithm-1 형태의 prior-free 대조군을 뜻한다.

## 공통 검색 조건

- significance score: ImageNet train 고정 2,048장
- block reconstruction train: 별도 4,096장
- acceptance eval: 별도 2,048장
- 세 subset은 상호 배타적이며 ID를 run output에 저장
- block fine-tuning: candidate마다 AdamW, lr 1e-5, 3 epochs
- candidate granularity: 10 tokens
- GPU: Tiny queue=6, Small queue=4
- search batch: Tiny 32, Small 16

## 최종 classification fine-tuning

- 12개 모두 30 epochs
- AdamW, lr 2e-5, weight decay 0.05, warmup 3 epochs
- batch: Tiny 384, Small 192 (실측 검증 후 필요 시 안전값으로 조정)
- 첫 비교에는 distillation을 넣지 않는다. 정책 자체의 영향을 공정하게 비교한 뒤 recovery가
  부족한 후보에만 동일 teacher distillation ablation을 별도 수행한다.
- epoch마다 model/optimizer/scheduler/scaler를 저장해 재개 가능

## W&B

- project: `vit-patch-slimming-final-12way-20260806`
- search run: `{model}__{policy}__rel002__search__{config_filename}`
- fine-tune run: `{model}__{policy}__rel002__finetune30`
- W&B에는 loss/accuracy와 reconstruction search의 기본 지표만 기록한다.
- Throughput, analytical MAC, layer activation memory는 12개 완료 뒤 reduced baseline과 동일한
  측정 harness로 일괄 비교하며 search/fine-tune run에는 섞지 않는다.

## 재현 및 실행

행렬 원본은 `configs/final/final_12way_matrix.yaml`, 각 resolved config와 manifest는
`output/ps_final_12way_20260806/` 아래 저장된다. GPU별 queue는 architecture 또는 마지막
fine-tune checkpoint를 검사해 완료된 작업을 건너뛰며 재개할 수 있다.

이 문서에는 검색 및 fine-tuning 완료 순서대로 schedule, accuracy와 장애/재개 기록을 추가한다.
