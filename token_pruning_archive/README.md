# token_pruning_archive — EViT Token Pruning 실험 (아카이브)

이 폴더는 **EViT 스타일 token pruning(Stage 2)** 실험의 전체 코드·설정·기록을
보존한 아카이브다. 활성 파이프라인(루트의 Stage 1 channel pruning)에서 분리해
따로 모아뒀다.

## 왜 아카이브인가

- **GPU/CPU에서는 정상 동작한다.** 학습·평가·정확도 회복까지 문제없이 되며,
  실제로 압축된 모델을 만들어낸다.
- **그러나 목표 NPU(Mobilint Aries2 / qbcompiler) 배포에는 실패했다.** EViT의
  핵심인 "이미지마다 다른 토큰을 런타임에 선택"하는 구조가 정적 shape·정적
  스케줄을 전제로 하는 이 NPU와 근본적으로 맞지 않는다. 4차례에 걸친 우회
  시도(Gather → index_select → onehot_matmul, cpu_offload 등)가 전부 같은
  지점에서 막혔다.
- 상세한 실패 분석과 시도별 결과는 **[TOKEN_PRUNING.md](TOKEN_PRUNING.md)** 에 있다.
- 후속 방향은 정적(static) 토큰 선택 기법인 **Patch Slimming**으로 이동했다
  (repo 루트 `patch_slimming/` 참고). 그쪽은 토큰 인덱스가 컴파일타임 상수라
  이 NPU 제약을 회피한다.

## 파일

| 파일 | 설명 |
|------|------|
| `TOKEN_PRUNING.md` | 전체 구현·실패 분석 보고서 (가장 먼저 읽을 것) |
| `token_pruning.py` | `EvitTokenPruner` — CLS attention 기반 token 선택 + fusion 핵심 |
| `token_pruning_npu.py` | NPU 우회 실험 forward 변형 (index_select / onehot_matmul) |
| `train_token_pruning.py` | Stage 2 학습 진입점 (reduced.pt 입력) |
| `eval_token_pruned.py` | token pruned 모델 평가 |
| `export_onnx.py` | token pruned 모델 → ONNX (루트 export_onnx.py에서 분리) |
| `configs/*.yaml` | 학습 설정 5종 |

## 실행 (repo 루트에서)

스크립트들은 `sys.path`에 repo 루트를 자동으로 추가하므로 **repo 루트에서**
아카이브 경로를 지정해 실행한다. 학습 로직은 GPU에서 그대로 재현 가능하다.

```bash
# 학습 (GPU/CPU에서 정상 동작)
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 \
  token_pruning_archive/train_token_pruning.py \
  --config token_pruning_archive/configs/vit_tiny_30_token_prune70.yaml

# 평가
python token_pruning_archive/eval_token_pruned.py \
  --token-pruned ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet --gpu 4

# ONNX 변환 (GPU/CPU 검증용 — NPU 컴파일은 실패, §9.2)
python token_pruning_archive/export_onnx.py \
  --input ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt --verify
```

## 루트에서 바뀐 것

- 루트 `export_onnx.py`는 이제 **Stage 1(reduced.pt) 전용**이다. token pruning
  관련 인자(`--npu-safe`/`--npu-mode`)와 로직은 이 폴더의 `export_onnx.py`로 옮겼다.
- `pruning/` 패키지에서 `token_pruning.py`, `token_pruning_npu.py`가 빠졌다
  (활성 파이프라인은 `vit_pruning.py`/`vit_reducing.py`/`vit_flops.py`만 사용).
