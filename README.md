# Server Compression

`Server_Compression`은 Vision Transformer를 학습 서버에서 압축하고, 실제로 크기가
줄어든 모델과 ONNX 배포 파일을 만드는 저장소다. 현재 최종 파이프라인은
ViT-Tiny/Small에 **Global L2 structured pruning + knowledge distillation**을 적용한 뒤,
정적 **Patch Slimming**을 추가하는 방식이다.

이 저장소의 책임은 ONNX 생성까지다. ONNX→Mobilint MXQ 컴파일과 실제 Aries2/Orin NPU
평가는 별도 [`npu`](../npu/) 저장소에서 수행한다. 이전 `Orin_Compression` 저장소는
현재 최종 워크플로에 사용하지 않는다.

## 전체 흐름

```text
ImageNet pretrained ViT
  └─ Global L2 structured pruning + same-architecture KD
       └─ 물리적 channel reducing
            └─ 정적 Patch Slimming 검색
                 └─ block reconstruction fine-tuning (후보마다 3 epochs)
                      └─ 최종 recovery fine-tuning (30 epochs)
                           └─ baseline / reduced / patch ONNX export
                                └─ npu 저장소에서 ONNX → MXQ → NPU benchmark
```

### 1. Structured pruning과 reducing

각 Transformer block의 MLP hidden channel 중요도를 L2 norm으로 계산하고, 전체 block의
채널을 global ranking하여 목표 압축률만큼 pruning한다. 학습 중에는 채널을 마스킹하며,
완료 후 `pruning/reduce.py`가 해당 행과 열을 물리적으로 제거해 block별 MLP 폭이 작은 dense
모델을 생성한다. 최종 실험은 Progressive Taylor를 사용하지 않는다.

Knowledge distillation은 동일 아키텍처의 pretrained teacher를 사용한다.

- Tiny student ← Tiny teacher
- Small student ← Small teacher
- KD alpha: `0.5`
- temperature: `4.0`

### 2. Patch Slimming

Reduced 모델에서 데이터셋 전체에 공통으로 적용할 patch ID를 오프라인 검색한다. 입력마다
TopK를 다시 계산하는 dynamic token pruning과 달리 레이어별 token 수와 index가 고정되므로,
고정 shape ONNX와 NPU graph를 만들 수 있다.

논문 재현 검색 설정은 path-energy score, `epsilon=0.02`, sample-relative
Frobenius error를 사용한다. 후보 평가 시 다음 block까지 통과한 reconstruction error를
측정하고, 각 후보 block을 3 epochs fine-tuning한다. 검색 후 compact 모델 전체를 30
epochs fine-tuning한다.

NPU 정확도 회복을 위한 완화형은 같은 Global-L2 reduced checkpoint에서 시작하며,
1-based layer 3·6·9 뒤의 token 수를 각각 `171 → 151 → 131`로 고정한다. 따라서 block
입출력 token schedule은 `[197,197,197,171,171,171,151,151,151,131,131,131,1]`이다.
이 버전도 검색용 score 계산과 30-epoch recovery fine-tuning을 거쳐 별도 ONNX로 export한다.

NPU 배포 시에는 실제 token 수가 감소하는 stage에만 고정 selection 연산을 남겨야 한다.
token 수가 같은 stage의 identity selection MatMul은 MXQ 양자화 오차를 크게 누적시키므로
`npu/compile/onnx/remove_identity_selections.py`로 제거한다. 이 후처리는 float logits를
변경하지 않는다.

## 최종 실험 대상

| Run | 모델 | MLP 목표 압축률 | Reduced Top-1 | Patch Top-1 |
|---|---|---:|---:|---:|
| `tiny30` | ViT-Tiny | 30% | 72.80% | 69.39% |
| `tiny50` | ViT-Tiny | 50% | 64.95% | 60.38% |
| `small30` | ViT-Small | 30% | 80.22% | 78.68% |
| `small50` | ViT-Small | 50% | 75.45% | 73.33% |

완화형 3·6·9의 30-epoch fine-tuning 후 Top-1은 Tiny30 `70.942%`, Tiny50
`63.862%`, Small30 `79.014%`, Small50 `74.152%`다. 논문 방식보다 각각
`+1.556`, `+3.478`, `+0.336`, `+0.822%p` 높다. 자세한 설정과 ONNX 검증 결과는
[`FINAL_STAGE369_RELAXED_20260826.md`](docs/FINAL_STAGE369_RELAXED_20260826.md)에 있다.

전체 정확도, token schedule, 메모리 및 activation 결과는
[`FINAL_RESULTS.md`](output/final_global_kd_l2_paper_20260814/FINAL_RESULTS.md)에 있다.
W&B 프로젝트 이름은 `vit-final-global-kd-l2-paper-20260814`다.

## 저장소 구조

```text
Server_Compression/
├── configs/
│   └── final_global_kd_l2_20260814.yaml  최종 4개 실험 공통 설정
├── pruning/
│   ├── train.py                          structured pruning + KD 학습 진입점
│   ├── engine.py                         공용 학습·평가 epoch loop
│   ├── reduce.py                         checkpoint → 물리적 reduced.pt 변환
│   ├── vit_pruning.py                    structured pruning
│   ├── vit_reducing.py                   물리적 MLP channel 제거
│   └── vit_flops.py                      연산량 분석
├── patch_slimming/
│   ├── psvit/                            검색·reconstruction·compact forward
│   ├── scripts/                          search, fine-tune, export, profile
│   └── docs/                             학술·구현·실험 문서
├── scripts/
│   ├── evaluation/
│   │   ├── eval_baseline.py              pretrained baseline ImageNet 평가
│   │   └── eval_reduced.py               reduced.pt ImageNet 평가
│   ├── profiling/
│   │   └── measure_memory.py             ViT 구조·파라미터 그룹 분석
│   ├── export_onnx.py                    reduced.pt 단일 ONNX export(레거시 CLI)
│   ├── run_final_global_kd_l2.py         재개 가능한 최종 파이프라인
│   ├── run_stage369_relaxed.py           완화형 3·6·9 파이프라인
│   ├── export_final_onnx.py              3단계 통합 ONNX exporter
│   └── build_final_global_kd_l2_report.py
├── output/                               checkpoint, ONNX, 결과 보고서
├── token_pruning_archive/                NPU 부적합 dynamic pruning 실험 기록
└── IMPLEMENTATION.md                     structured pruning 상세 구현 문서
```

## 데이터와 환경

최종 실험은 ImageNet-1K을 사용한다.

```text
data/imagenet/
├── train/<class directories>
└── val/<class directories>
```

기본 경로는 `configs/final_global_kd_l2_20260814.yaml`의 `data_path`에서 변경할 수 있다.
주요 Python 의존성은 PyTorch, torchvision, timm, PyYAML, NumPy, ONNX,
ONNX Runtime 및 Weights & Biases다. 실제 학습 전 CUDA/PyTorch 호환성과 ImageNet 경로를
먼저 확인한다.

## 최종 파이프라인 실행

모든 명령은 저장소 루트에서 실행한다.

```bash
cd /workspace/etri_iitp/JS/Server_Compression
```

먼저 네 실험의 디렉터리와 개별 설정을 생성한다.

```bash
python scripts/run_final_global_kd_l2.py --phase materialize
```

GPU별 queue를 병렬 실행한다. 설정상 GPU 6은 Tiny, GPU 7은 Small을 담당한다.

```bash
python scripts/run_final_global_kd_l2.py --phase structural --gpu 6
python scripts/run_final_global_kd_l2.py --phase structural --gpu 7

python scripts/run_final_global_kd_l2.py --phase patch --gpu 6
python scripts/run_final_global_kd_l2.py --phase patch --gpu 7

python scripts/run_final_global_kd_l2.py --phase export --gpu 6
python scripts/run_final_global_kd_l2.py --phase export --gpu 7

python scripts/run_final_global_kd_l2.py --phase memory --gpu 6
python scripts/run_final_global_kd_l2.py --phase memory --gpu 7
```

각 phase는 완료 checkpoint와 marker를 확인하고 끝난 단계를 건너뛴다. 중단된 학습은
가능한 경우 `checkpoint_last.pt`에서 재개한다.

최종 결과 보고서를 다시 생성하려면:

```bash
python scripts/build_final_global_kd_l2_report.py
```

완화형 3·6·9 실험은 다음 설정과 실행기를 사용한다.

```bash
python scripts/run_stage369_relaxed.py --help
```

실험 산출물은 `output/final_global_kd_l2_stage369_relaxed_20260826/` 아래에 있으며,
네 개의 NPU-safe ONNX는 `npu/assets/onnx/`에도 전달되어 있다.

## 개별 평가·분석 도구

Structured pruning의 활성 코드는 모두 `pruning/`에 둔다. `pruning/train.py`와
`pruning/reduce.py`는 실제 최종 파이프라인의 진입점이고, `pruning/engine.py`는 structured
pruning뿐 아니라 Patch Slimming의 fine-tuning과 평가에서도 공유한다. 참고용 파일이 아니다.
평가, 프로파일링, 단일 ONNX export 도구는 역할별로 `scripts/` 아래에 둔다.

```bash
# pretrained Tiny/Small baseline 평가
python scripts/evaluation/eval_baseline.py \
  --data-path ./data/imagenet --wandb

# reduced.pt 평가
python scripts/evaluation/eval_reduced.py \
  --reduced /path/to/reduced.pt --data-path ./data/imagenet --gpu 0

# timm ViT 파라미터 그룹과 목표 압축률 분석
python scripts/profiling/measure_memory.py

# reduced.pt 단일 ONNX export 및 ORT 검증
python scripts/export_onnx.py \
  --input /path/to/reduced.pt --output /path/to/reduced.onnx --verify
```

여기서 `scripts/profiling/measure_memory.py`는 모델 구조와 파라미터 감소량을 분석하는
초기 도구다. baseline/reduced/patch의 실제 CUDA peak memory와 layer별 activation 비교는
최종 파이프라인이 사용하는 `patch_slimming/scripts/profile_memory.py`가 담당한다.

## ONNX 산출물

각 run은 다음 세 ONNX를 생성한다.

```text
output/final_global_kd_l2_paper_20260814/<run>/04_onnx/
├── <run>__baseline__nhwc_b1.onnx
├── <run>__reduced__nhwc_b1.onnx
└── <run>__patch_slimming__nhwc_b1.onnx
```

공통 contract:

- input: NHWC `[1,224,224,3]`, FP32
- output: `[1,1000]`
- opset: 17
- static batch: 1
- PyTorch–ONNX Runtime 최대 절대오차 검증

단일 파일을 직접 export할 때는 다음 형식을 사용한다.

```bash
python scripts/export_final_onnx.py --kind baseline \
  --model vit_tiny_patch16_224 --output baseline.onnx

python scripts/export_final_onnx.py --kind reduced \
  --reduced /path/to/reduced.pt --output reduced.onnx

python scripts/export_final_onnx.py --kind patch \
  --arch /path/to/architecture.json \
  --weights /path/to/compact_finetuned_best.pt \
  --output patch_slimming.onnx
```

## NPU 저장소로 인계

최종 4개 run × 3단계, 총 12개 ONNX를 `npu/assets/onnx/`로 전달한다. Baseline과
Reduced는 그대로 사용하고, Patch Slimming은 identity selection을 제거한 NPU-safe
파일을 사용한다.

```text
Server_Compression/output/.../04_onnx
  → npu/assets/onnx
  → timm calibration 생성
  → qbcompiler로 MXQ 변환
  → Aries2/Orin에서 정확도·latency·메모리 측정
```

기존 12개에 완화형 Patch Slimming 4개를 더하면 현재 NPU 전달 대상은 총 16개다. MXQ
컴파일 명령, timm 전처리, 실제 NPU benchmark 및 Patch Slimming 양자화 주의사항은
`npu/docs/`를 기준으로 한다. `Server_Compression`에서 MXQ를 생성하거나
`Orin_Compression`으로 파일을 전달하는 것은 현재 표준 절차가 아니다.

## 문서 안내

- [`IMPLEMENTATION.md`](IMPLEMENTATION.md): structured pruning/reducing 상세 구현
- [`FINAL_GLOBAL_KD_L2_PAPER_20260814.md`](docs/FINAL_GLOBAL_KD_L2_PAPER_20260814.md): 최종 실험 설정과 실행
- [`FINAL_IMPLEMENTATION_GUIDE.md`](patch_slimming/docs/FINAL_IMPLEMENTATION_GUIDE.md): Patch Slimming 전체 학술·코드 구현
- [`EXPERIMENT_WORKFLOW_AND_RESULTS.md`](patch_slimming/docs/EXPERIMENT_WORKFLOW_AND_RESULTS.md): 실험 순서와 결과
- [`PRESENTATION_STRUCTURED_PRUNING_PATCH_SLIMMING_NPU.md`](docs/PRESENTATION_STRUCTURED_PRUNING_PATCH_SLIMMING_NPU.md): Structured pruning부터 NPU 평가까지 발표 슬라이드 원고
- [`token_pruning_archive/README.md`](token_pruning_archive/README.md): dynamic token pruning을 아카이브한 이유

## 주의사항

- `output/`, ImageNet, checkpoint, ONNX는 크기가 크므로 저장 공간과 Git 추적 여부를
  확인한다.
- ONNX가 GitHub 단일 파일 제한에 접근하면 Git LFS 또는 외부 artifact storage를 쓴다.
- GPU latency는 NPU latency를 대신하지 않는다. Patch Slimming selection MatMul과 작은
  rectangular attention의 최적화 여부는 실제 장치에서 측정한다.
- Reduced 모델은 MLP channel이 물리적으로 제거되지만 Patch Slimming은 trainable
  parameter 수를 더 줄이지 않는다. 대신 token sequence와 attention activation을 줄인다.
- 과거 Progressive Taylor와 dynamic token-pruning 결과는 최종 제출 결과와 구분한다.
