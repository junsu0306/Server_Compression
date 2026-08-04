# Patch Slimming (정적 PS-ViT) — 구현

Tang et al., "Patch Slimming for Efficient Vision Transformers" (CVPR 2022)의 **정적
PS-ViT**를 timm ViT/DeiT에 구현한 것. 사전학습 ViT의 마지막 CLS 출력에서 시작해
top-down으로 layer별 **고정** patch mask를 찾고, 각 block을 `Q: N_out, K/V: N_in`의
rectangular attention으로 바꿔 토큰을 실제로 줄인다.

- 구현 명세: [PATCH_SLIMMING_IMPLEMENTATION_SPEC.md](PATCH_SLIMMING_IMPLEMENTATION_SPEC.md) (이하 SPEC)
- NPU 배포는 EViT와 달리 **가능**하다 — 토큰 선택이 컴파일타임 상수라 상수 선택행렬
  matmul로 표현되고, Gather/런타임 인덱스가 없다 (SPEC §14.0, 실측 컴파일 완주 확인).
  대비: [../token_pruning_archive/TOKEN_PRUNING.md](../token_pruning_archive/TOKEN_PRUNING.md) (EViT NPU 실패).

> **왜 정적인가**: EViT(token_pruning_archive)는 이미지마다 다른 토큰을 런타임에
> 골라(TopK+Gather) NPU 컴파일이 불가능했다. Patch Slimming은 데이터셋 통계로
> **오프라인에서 고정** 위치를 정하므로 런타임 선택이 없다 → NPU 배포 가능.

## 파일 구조

```
patch_slimming/
├── psvit/                    핵심 라이브러리
│   ├── ids.py               §3  global/local ID, nested mask
│   ├── model_utils.py           timm ViT adapter/검증
│   ├── data.py                  calibration subset (deterministic)
│   ├── slim_block.py        §5  SlimBlock (rectangular attention)
│   ├── instrument.py        §6  attention/feature 계측
│   ├── scoring.py           §7  path-energy DP significance score
│   ├── reconstruct.py       §9  block reconstruction fine-tuning
│   ├── search.py          §8,11 Algorithm 1 top-down search
│   ├── architecture.py      §4  architecture.json 직렬화
│   └── compact.py         §12,14 CompactPSViT (NPU-safe 상수 선택행렬)
├── tests/                    §16 게이팅 테스트 (ImageNet 불필요, CPU 수 초)
├── run_search.py            Phase 5  top-down mask search
├── build_compact.py         Phase 6a compact 모델 평가
├── finetune_compact.py      Phase 6b 최종 classification fine-tuning
├── export_ps_onnx.py        Phase 7  NPU-safe ONNX export
├── toy_rectangular_attention.py   NPU 사전검증 toy (이미 통과)
└── configs/deit_tiny_ps.yaml
```

## 논문에 수치가 없는 부분 → 검증으로 대체

논문은 정확한 optimizer/lr/error normalization/head-path 집계 코드를 안 준다
(SPEC §19). 그래서 **각 Phase는 테스트로 게이트**한다. 특히:

- **SlimBlock 수치 등가성** (§16.3, `test_slim_block.py`): all-token 출력 = 원본 block,
  subset 출력 = 원본의 subset row, matmul = index_select. 이게 안 되면 이후 전부 무의미.
- **DP score = brute-force** (§16.5, `test_score_dp.py`): path-energy DP가 head-path
  완전 열거와 일치. **이 테스트 통과 전엔 search를 시작하지 않는다** (SPEC §7.6).
- **계측 logit 불변** (§16.1), **nested mask invariant** (§16.4), **compact 등가성** (§16.9).

## 실행 순서 (repo 루트에서, 서버)

```bash
# ── Phase 1~2 + scoring 게이트: ImageNet 없이 먼저 (필수) ──
python patch_slimming/tests/run_all.py
#   전부 PASS해야 아래로 진행

# ── Phase 5: top-down mask search (GPU 1장) ──
python patch_slimming/run_search.py \
  --config patch_slimming/configs/deit_tiny_ps.yaml --gpu 4
#   → output/ps_deit_tiny/architecture.json, searched_student.pt, search_records.jsonl
#   (search_last.pt로 layer 단위 재개 가능)

# ── Phase 6a: compact 모델 평가 (최종 fine-tuning 전 baseline) ──
python patch_slimming/build_compact.py \
  --arch    ./output/ps_deit_tiny/architecture.json \
  --weights ./output/ps_deit_tiny/searched_student.pt \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet --gpu 4

# ── Phase 6b: 최종 전체 fine-tuning (DDP) ──
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29520 \
  patch_slimming/finetune_compact.py --config patch_slimming/configs/deit_tiny_ps.yaml
#   → output/ps_deit_tiny/compact_finetuned_best.pt

# ── Phase 7: NPU-safe ONNX (NHWC 입력, 상수 선택행렬 matmul) ──
python patch_slimming/export_ps_onnx.py \
  --arch    ./output/ps_deit_tiny/architecture.json \
  --weights ./output/ps_deit_tiny/compact_finetuned_best.pt \
  --output  ./output/ps_deit_tiny/ps_compact_npu.onnx --verify
#   → 이 .onnx를 qbcompiler에 넣는다 (Gather/TopK/Equal 없어야 함)
```

## 검증 상태 (중요)

- 이 코드는 **서버(GPU)에서 실행/검증**한다. 로컬에서 돌린 적 없다.
- **먼저 `tests/run_all.py`를 돌려 게이트를 통과**시킨 뒤 search로 넘어갈 것. 테스트가
  깨지면 그 지점(SlimBlock 등가성 / DP 정확성 등)부터 수정한다.
- search는 계산량이 크다(§7의 hybrid forward가 layer마다 downstream attention NxN을
  누적). 처음엔 config의 `search.score_max_batches`를 작은 값(예: 8)으로 두고
  `calibration.num_samples`도 줄여 파이프라인이 도는지 확인한 뒤 전체로 키운다.

## 알려진 [DECISION] / ambiguity (SPEC §19)

| 항목 | 이 구현의 결정 |
|------|----------------|
| head-path 집계 | path-energy DP (§7.5) + brute-force 검증 (§16.5) |
| error metric | MSE (acceptance 기본), raw/relative frobenius도 로깅 (§9.4) |
| 후보 weight 정책 | cumulative fine-tuning (§8.6) |
| block optimizer/lr | AdamW 1e-5 (config에 명시, 논문 미기재) |
| calibration | deterministic subset, sample id 저장 (§9.6) |
| 토큰 선택(NPU) | 상수 선택행렬 matmul (§14.0 실측, Gather 금지) |
| 입력 레이아웃(NPU) | NHWC [1,224,224,3] (§14.0) |
