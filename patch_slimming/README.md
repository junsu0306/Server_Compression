# Patch Slimming

최종 단일 구현 문서: [`docs/FINAL_IMPLEMENTATION_GUIDE.md`](docs/FINAL_IMPLEMENTATION_GUIDE.md)

정적 Patch Slimming을 timm ViT/DeiT와 structural-pruned ViT에 적용하는 구현이다.
토큰 위치는 ImageNet calibration 통계로 offline에서 확정하며, 각 block은
`Q: N_out`, `K/V: N_in` rectangular attention으로 실제 sequence 길이를 줄인다.

## 먼저 읽을 문서

1. [전체 구현 및 실험 결과](docs/EXPERIMENT_WORKFLOW_AND_RESULTS.md) — 실제 진행 순서,
   결과, 현재 선택과 다음 단계
2. [최초 구현 계획과 명세](docs/IMPLEMENTATION_PLAN.md) — 논문 해석, tensor contract,
   알고리즘 및 NPU 요구사항
3. [문서 인덱스](docs/README.md) — smoke/main/structural 상세 로그

실제 production 대상은 `output/vit_{tiny,small}_{30,50}_final/reduced.pt` 네 모델이다.
원본 DeiT-Tiny 결과는 reference 검증이며 production schedule은 structural 모델에서
별도로 산출했다.

## 디렉터리 구조

```text
patch_slimming/
├── README.md                    이 파일: 탐색과 실행 진입점
├── docs/
│   ├── IMPLEMENTATION_PLAN.md   최초 계획·알고리즘 명세
│   ├── EXPERIMENT_WORKFLOW_AND_RESULTS.md
│   │                            전체 구현 과정과 통합 결과
│   └── experiments/             단계별 원시 실험 로그
├── psvit/                       재사용 가능한 핵심 모델 라이브러리
├── scripts/                     실제 executable workflow 구현
│   ├── search.py                full top-down search
│   ├── stage_ablation.py        structural 모델 3·6·9 후보 실험
│   ├── evaluate.py              compact ImageNet 평가
│   ├── finetune.py              DDP classification fine-tuning + W&B
│   ├── export_onnx.py           NPU-safe ONNX export
│   └── toy_npu_validation.py    rectangular attention 사전 검증
├── configs/
│   ├── reference/               원본 DeiT smoke/main 설정
│   └── structural/              production structural 모델 설정
├── tests/                        CPU/GPU 회귀 및 통합 테스트
├── references/                   논문 원문
├── artifacts/                    toy 검증 산출물
└── *.py                          기존 명령을 보존하는 얇은 호환 wrapper
```

`psvit/`에는 알고리즘과 모델 코드만 두고, 데이터셋을 실제로 돌리는 workflow는
`scripts/`에 둔다. 루트 Python 파일에는 구현이 없으며 기존 명령 호환만 담당한다.

## 핵심 파일 책임

| 경로 | 책임 |
|---|---|
| `psvit/ids.py` | global/local token ID와 nested mask 검증 |
| `psvit/scoring.py` | path-energy DP significance score |
| `psvit/reconstruct.py` | candidate block reconstruction |
| `psvit/search.py` | Algorithm 1 top-down search와 resume |
| `psvit/slim_block.py` | rectangular attention block |
| `psvit/compact.py` | 실제 token tensor를 축소하는 compact model |
| `psvit/model_utils.py` | timm 및 structural-reduced model adapter |
| `psvit/architecture.py` | static architecture JSON 직렬화 |

## 실행 순서

모든 명령은 repository root에서 실행한다. 큰 cache와 임시 파일은 workspace의
`.cache/`, `.tmp/`, `.deps/`를 사용한다.

### 1. 테스트

```bash
python patch_slimming/tests/run_all.py
```

현재 기준 33/33 테스트가 통과해야 한다.

### 2. Reference full search

```bash
python patch_slimming/scripts/search.py \
  --config patch_slimming/configs/reference/deit_tiny_ps.yaml \
  --gpu 4
```

기존 `patch_slimming/run_search.py` 명령도 동일하게 동작한다.

### 3. Structural 3·6·9 후보 산출과 평가

```bash
PYTHONPATH=$PWD/.deps:$PYTHONPATH \
python patch_slimming/scripts/stage_ablation.py \
  --reduced output/vit_tiny_30_final/reduced.pt \
  --output-dir output/ps_structural/vit_tiny_30_final \
  --data-path data/imagenet \
  --gpu 4 --wandb
```

동일 명령을 Tiny/Small × structural 30/50 네 모델에 실행한다. 현재 선택된 production
정책은 layer 3·6·9 기준 medium `(157, 118, 79)`이다.

### 4. Compact 평가

```bash
python patch_slimming/scripts/evaluate.py \
  --arch output/ps_structural/vit_tiny_30_final/medium/architecture.json \
  --weights output/ps_structural/vit_tiny_30_final/medium/searched_student.pt \
  --data-path data/imagenet --gpu 4
```

### 5. Fine-tuning과 W&B

```bash
PYTHONPATH=$PWD/.deps:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
  patch_slimming/scripts/finetune.py \
  --config patch_slimming/configs/structural/structural_tiny30_medium_short.yaml
```

W&B 프로젝트는 `vit-structural-patch-slimming`이다. `*_short.yaml`은 후보 선택용
1-epoch 설정이며 최종 배포 학습 설정이 아니다.

### 6. NPU-safe ONNX

```bash
python patch_slimming/scripts/export_onnx.py \
  --arch output/ps_structural/vit_tiny_30_final/medium/architecture.json \
  --weights output/ps_structural/vit_tiny_30_final/medium/compact_finetuned_best.pt \
  --output output/ps_structural/vit_tiny_30_final/medium/compact_npu.onnx \
  --verify
```

export graph에는 `Gather`, `GatherElements`, `ScatterElements`, `TopK`, `Equal`이 없어야
한다. 입력은 static batch-one NHWC다.

## 현재 결과

- Production schedule: `[197, 197, 197, 157, 157, 157, 118, 118, 118, 79, 79, 79, 1]`
- Structural 모델 이후 추가 transformer MAC 감소: 약 35–36%
- 원본 full-width 모델 대비 combined 감소: 52.70–66.37%
- Medium 1-epoch 회복: 모델별 `+2.34`~`+5.84%p`
- W&B: <https://wandb.ai/junsu000306/vit-structural-patch-slimming>
- 테스트: 33/33 통과

상세 수치와 실험 해석은
[전체 구현 및 실험 결과](docs/EXPERIMENT_WORKFLOW_AND_RESULTS.md)를 기준으로 한다.
