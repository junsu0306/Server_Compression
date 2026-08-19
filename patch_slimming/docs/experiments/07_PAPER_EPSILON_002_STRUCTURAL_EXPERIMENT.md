# Paper-style epsilon 0.02 search on structural-pruned ViTs

이 문서는 epsilon 0.05 stress test 이후 수행하는 epsilon 0.02 논문식 top-down search의
실행 과정과 결과를 기록한다.

## 1. 목적

epsilon 0.05에서는 1-epoch recovery가 9.14~18.73%p로 컸지만 token 수가 지나치게
감소해 structural baseline과 16.34~49.00%p 차이가 남았다. 이번 실험은 논문에서
사용된 더 엄격한 global tolerance인 epsilon 0.02가 정확도와 token 감소의 균형을
개선하는지 확인한다.

## 2. 비교 가능성을 위한 고정 조건

- 대상: `output/vit_*_{30,50}_final/reduced.pt` 네 structural 모델
- dataset: ImageNet-1K
- calibration: train 1,024장, seed 2022, validation preprocessing
- search: block 10에서 block 0까지 top-down; block 11은 CLS-only
- score: paper path-energy DP
- candidate 증가: 10 tokens
- acceptance: next-layer reconstruction element-wise MSE <= 0.02
- candidate reconstruction: AdamW, LR 1e-5, weight decay 0, 3 epochs
- recovery: ImageNet 전체 train 1 epoch, LR 2e-5, warmup 없음
- validation: ImageNet validation 전체 50,000장
- GPU: 물리 GPU 4와 6만 사용
- output: `output/ps_structural_paper_eps002/`
- W&B project: `vit-structural-patch-slimming-paper-eps002`

epsilon 외에는 04의 epsilon 0.05 실험과 동일한 조건을 유지한다.

## 3. 실행 상태

| 단계 | 상태 |
|---|---|
| 설정 및 새 W&B project 분리 | 완료 |
| Tiny30/50 top-down search | 완료 |
| Small30/50 top-down search | 완료 |
| pre-fine-tuning full validation | 완료 |
| 1-epoch recovery | 완료 |
| epsilon 0.05 및 medium 비교 | 완료 |

결과는 각 단계가 끝날 때 이 문서에 추가한다. 모든 candidate는 각 output 폴더의
`search_records.jsonl`과 W&B에 기록한다.

## 4. Tiny search 결과

| 모델 | epsilon 0.02 schedule | block 10 accepted MSE | block 0 accepted MSE | W&B run |
|---|---|---:|---:|---|
| Tiny 30 | `[197, 41 x11, 1]` | 0.017884 | 0.016838 | `xcckugqv` |
| Tiny 50 | `[197, 31 x11, 1]` | 0.018642 | 0.019024 | `jriwcr5l` |

epsilon 0.05에서 두 Tiny 모델 모두 11 tokens였지만 epsilon 0.02에서는 Tiny30이 41,
Tiny50이 31 tokens를 유지했다. 가장 깊은 searchable block에서 필요한 token 수가
결정된 뒤 더 얕은 block들은 추가 token 없이 tolerance를 만족했다. 실제 classification
개선 여부는 full validation으로 확인한다.

## 5. Small search 결과

| 모델 | epsilon 0.02 schedule | block 10 accepted MSE | block 9 accepted MSE | W&B run |
|---|---|---:|---:|---|
| Small 30 | `[197, 111 x10, 81, 1]` | 0.016298 | 0.019985 | `aolux4pn` |
| Small 50 | `[197, 101 x10, 81, 1]` | 0.017052 | 0.019019 | `kqnahytk` |

두 모델 모두 block 10에서 81 tokens가 필요했다. 바로 앞 block 9에서 Small30은 111,
Small50은 101 tokens로 증가했고, 이후 더 얕은 block은 해당 mask로 모두 tolerance를
통과했다.

## 6. 파인튜닝 전 전체 ImageNet validation

| 모델 | Structural baseline | epsilon 0.05 | epsilon 0.02 | 0.05 대비 개선 | Baseline까지 차이 | W&B run |
|---|---:|---:|---:|---:|---:|---|
| Tiny 30 | 73.53% | 5.80% | 41.86% | +36.06%p | -31.67%p | `25vjyesp` |
| Tiny 50 | 66.14% | 3.24% | 20.23% | +16.99%p | -45.91%p | `78z2u2r4` |
| Small 30 | 81.05% | 55.57% | 71.92% | +16.35%p | -9.13%p | `d9hms7rb` |
| Small 50 | 77.41% | 33.56% | 62.04% | +28.48%p | -15.37%p | `b60n4k06` |

epsilon 감소로 네 모델 모두 pre-FT accuracy가 크게 개선됐다. Small30은 이미 기존
medium pre-FT 72.44%에 근접했지만 Tiny, 특히 Tiny50은 여전히 큰 차이가 있다.
동일한 1-epoch recovery를 수행해 실제 회복 가능 범위를 확인한다.

## 7. Tiny 1-epoch recovery

| 모델 | Pre-FT | 1 epoch 후 | 회복량 | Structural baseline까지 차이 | Medium 1 epoch | W&B run |
|---|---:|---:|---:|---:|---:|---|
| Tiny 30 | 41.86% | 52.44% | +10.59%p | -21.09%p | 68.42% | `z2l36fkt` |
| Tiny 50 | 20.23% | 39.69% | +19.46%p | -26.45%p | 61.19% | `siaucwtg` |

epsilon 0.05 recovery의 24.53/20.18%보다 크게 개선됐지만 기존 medium 후보와는
각각 15.98/21.50%p 차이가 남는다. Tiny에서는 epsilon 0.02도 production accuracy
관점에서 여전히 공격적이다.

## 8. Small 1-epoch recovery

| 모델 | Pre-FT | 1 epoch 후 | 회복량 | Structural baseline까지 차이 | Medium 1 epoch | W&B run |
|---|---:|---:|---:|---:|---:|---|
| Small 30 | 71.92% | 73.34% | +1.42%p | -7.71%p | 75.34% | `9fg99nt6` |
| Small 50 | 62.04% | 65.38% | +3.34%p | -12.03%p | 70.94% | `6rusg63k` |

Small30은 medium 1-epoch 결과와 2.00%p 차이까지 접근하면서 medium보다 transformer
본체 MAC을 추가로 19.03% 줄인다. Small50은 medium과 5.56%p 차이이며 추가 MAC
감소는 27.78%다. Small30은 장기 fine-tuning 비교 가치가 있는 후보지만 1 epoch
결과만으로 production medium을 교체하지 않는다.

## 9. 전체 accuracy 비교

| 모델 | Structural baseline | Medium 1 epoch | epsilon .05 1 epoch | epsilon .02 1 epoch |
|---|---:|---:|---:|---:|
| Tiny 30 | 73.53% | 68.42% | 24.53% | 52.44% |
| Tiny 50 | 66.14% | 61.19% | 20.18% | 39.69% |
| Small 30 | 81.05% | 75.34% | 64.71% | 73.34% |
| Small 50 | 77.41% | 70.94% | 50.55% | 65.38% |

epsilon 0.02는 0.05보다 모든 모델에서 낫지만 하나의 global epsilon이 네 structural
topology에 동일한 accuracy trade-off를 만들지는 않는다. Tiny50은 31 tokens로도 큰
손실이 남는 반면 Small30은 111/81 tokens에서 medium에 근접한다.

## 10. Transformer MAC 분석

아래 body MAC은 기존 실험과 동일하게 Q/K/V, rectangular attention, output projection,
structurally reduced MLP를 포함하며 explicit selection-matrix 비용은 제외한다.

| 모델 | epsilon .02 body MACs | Structural 대비 감소 | Original full-width 대비 감소 | Medium 대비 추가 감소 |
|---|---:|---:|---:|---:|
| Tiny 30 | 165,093,888 | 81.40% | 86.52% | 71.50% |
| Tiny 50 | 91,917,312 | 86.15% | 92.49% | 78.21% |
| Small 30 | 1,691,340,288 | 47.78% | 62.75% | 19.03% |
| Small 50 | 1,102,970,880 | 53.48% | 75.71% | 27.78% |

05에서 확인한 것처럼 현재 NPU export는 각 block에서 Q와 residual을 위해 selection
MatMul을 두 번 사용하고 identity selection도 명시적으로 남길 수 있다. compiler가
제거하지 않는 현재-graph MAC 추정은 다음과 같다.

| 모델 | 현재 selection MACs | Body+selection | Structural 대비 감소 | Identity 제거+1회 선택 시 selection MACs | 최적화 후 감소 |
|---|---:|---:|---:|---:|---:|
| Tiny 30 | 9,572,352 | 174,666,240 | 80.33% | 1,558,656 | 81.23% |
| Tiny 50 | 6,047,232 | 97,964,544 | 85.24% | 1,178,496 | 85.97% |
| Small 30 | 108,923,904 | 1,800,264,192 | 44.42% | 11,880,576 | 47.41% |
| Small 50 | 92,135,424 | 1,195,106,304 | 49.60% | 10,813,056 | 53.03% |

이는 analytical 추정이다. 실제 qbcompiler의 identity/one-hot MatMul 최적화와 device
latency는 production ONNX profiling으로 확인해야 한다.

## 11. 판단

- epsilon 0.02는 epsilon 0.05의 과도한 token 붕괴를 확실히 완화했다.
- Tiny 두 모델은 1 epoch 후에도 medium보다 15.98/21.50%p 낮아 production 후보가 아니다.
- Small50도 medium보다 5.56%p 낮아 현재는 medium 유지가 안전하다.
- Small30은 medium보다 2.00%p 낮지만 body MAC을 19.03% 추가 절감하므로 장기
  fine-tuning 대조 실험 가치가 있다.
- 하나의 global epsilon보다 topology별 epsilon 또는 classification guard가 필요하다.
  후속으로 Tiny는 epsilon 0.01, Small은 epsilon 0.02를 우선 비교하는 것이 타당하다.
- 현재 production 선택은 네 모델 모두 기존 3/6/9 medium으로 유지한다.

## 12. W&B와 산출물

- Project: <https://wandb.ai/junsu000306/vit-structural-patch-slimming-paper-eps002>
- Search runs: `xcckugqv`, `jriwcr5l`, `aolux4pn`, `kqnahytk`
- Pre-FT full-validation runs: `25vjyesp`, `78z2u2r4`, `d9hms7rb`, `b60n4k06`
- Recovery runs: `z2l36fkt`, `siaucwtg`, `9fg99nt6`, `6rusg63k`
- Configs: `configs/structural/paper_eps002_*.yaml`
- Search records/checkpoints: `output/ps_structural_paper_eps002/`
- Compute: physical GPUs 4 and 6 only

최종 검증에서 네 best checkpoint가 preserved structural `mlp_dims`로 strict-load됐고
runtime token schedule이 각 `architecture.json`과 일치했다. 33/33 regression test와
`git diff --check`도 통과했다. 전체 epsilon 0.02 산출물 크기는 약 2.003 GB다.
