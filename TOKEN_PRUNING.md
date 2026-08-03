# EViT Token Pruning (Stage 2) — 구현 보고서

> 작성 기준: 2026-07 (Stage 1 문서 [IMPLEMENTATION.md](IMPLEMENTATION.md)에서 분리)  
> 최종 업데이트: 2026-08  
> 환경: timm 1.0.27 · torch 2.9.1 · Python 3.13.5  
> 서버: `root@59bfae69b3a9:/workspace/etri_iitp/JS/Server_Compression`  
> 레퍼런스: EViT — Liang et al., "Not All Patches are What You Need: Expediting
> Vision Transformers via Token Reorganizations", ICLR 2022,
> [youweiliang/evit](https://github.com/youweiliang/evit)
> (알고리즘만 참고해 timm 1.0.x 호환 형태로 재구현 — repo 코드 직접 이식 아님, §1 참고)

> **이 문서는 Stage 2(token pruning, sequence 차원 압축)만 다룬다.**
> Stage 1(channel pruning, FFN width 압축) · 환경 설정 · 아키텍처 분석 ·
> `train.py`/`reduce.py` 등은 [IMPLEMENTATION.md](IMPLEMENTATION.md)를 참고할 것.
> Stage 2는 Stage 1이 끝난 `reduced.pt`를 입력으로 받는 후속 단계다.

> **구현 파일**: `pruning/token_pruning.py`(핵심 알고리즘), `pruning/token_pruning_npu.py`
> (NPU 호환 변형, 실험적), `train_token_pruning.py`(학습), `eval_token_pruned.py`(평가),
> `export_onnx.py`(공용 — Stage 1/2 모두 처리)  
> **config**: `configs/vit_{tiny,small}_token_prune70.yaml`(50% 채널압축 기반),
> `configs/vit_{tiny,small}_30_token_prune70.yaml`(30% 채널압축 기반),
> `configs/vit_tiny_30_token_prune70_npu_test.yaml`(NPU 컴파일 테스트, 10 epoch)

---

## 목차

1. [개요 — 왜, 그리고 무엇을 이식했는가](#1-개요--왜-그리고-무엇을-이식했는가)
2. [파이프라인 순서 — 왜 reduced 모델을 대상으로 하는가](#2-파이프라인-순서--왜-reduced-모델을-대상으로-하는가)
3. [알고리즘 — CLS Attention 기반 Token 선택 + Fusion](#3-알고리즘--cls-attention-기반-token-선택--fusion)
4. [Progressive Keep Rate 스케줄](#4-progressive-keep-rate-스케줄)
5. [Knowledge Distillation — Teacher 선택](#5-knowledge-distillation--teacher-선택)
6. [WandB 프로젝트 분리](#6-wandb-프로젝트-분리)
7. [실행 명령어](#7-실행-명령어)
8. [체크포인트 로드 방법](#8-체크포인트-로드-방법)
9. [NPU 배포](#9-npu-배포)
10. [주의사항 & 트러블슈팅](#10-주의사항--트러블슈팅)

---

## 1. 개요 — 왜, 그리고 무엇을 이식했는가

Stage 1(channel pruning)은 FFN의 **폭(width)**을 줄인다. Stage 2는 **시퀀스 길이
(패치 토큰 개수)**를 줄인다 — 서로 직교하는 압축 축이라 같은 backbone에 순차
적용 가능하다.

레퍼런스는 [youweiliang/evit](https://github.com/youweiliang/evit) (EViT,
Liang et al., ICLR 2022)이지만, **저장소를 그대로 이식하지 않았다.** 원 저장소는
`torch==1.9.0`, `timm==0.4.12` 기준으로 timm의 `Attention`/`Block`을 통째로
포크해서 고쳐놓은 코드라, 지금 쓰는 `timm==1.0.27`과 구조가 크게 다르다
(LayerScale, `fused_attn`/`F.scaled_dot_product_attention` 도입 등). 그래서
**알고리즘(CLS-attention 기반 top-k 선택 + fused token)만 가져오고, 구현은
현재 timm 버전에 맞게 새로 짰다.**

가장 중요한 차이점 하나: timm 1.0.x의 `Attention.forward`는 기본적으로
`fused_attn=True`라 `F.scaled_dot_product_attention`을 쓰며, 이 경로는 attention
행렬을 아예 만들지 않는다. 즉 EViT가 필요로 하는 "CLS→patch attention score"를
얻을 수 없다. 여기서 두 가지 선택지가 있었다:

1. `fused_attn`을 강제로 끄고 `Attention.forward` 전체를 eager 모드로 재구현
2. `Attention.forward`는 그대로 두고, CLS attention score만 별도로 계산

**2번을 선택했다.** 1번은 원본 attention 출력(x)을 손으로 재현해야 하는데,
사소한 구현 실수(reshape/permute 순서, scale 위치 등)가 조용히 정확도를 깎아먹을
위험이 있다. 2번은 `self.attn.qkv` Linear를 한 번 더 통과시켜 `q, k`만 뽑고
CLS row(`q[:, :, 0:1, :] @ k.T`)만 계산하는 것이라, 원본 attention 경로를 전혀
건드리지 않는다. 추가 비용은 O(N) 크기의 작은 matmul 하나뿐 — 전체 attention의
O(N²) 대비 무시할 수준이다.

---

## 2. 파이프라인 순서 — 왜 reduced 모델을 대상으로 하는가

```
[Stage 1] Soft channel pruning + KD (IMPLEMENTATION.md 참고, train.py)
    → checkpoint_best.pt

[Reduce] reduce.py
    → reduced.pt   (FFN이 물리적으로 축소된 순수 Dense 모델)

[Stage 2] EViT Token Pruning fine-tuning (train_token_pruning.py)
    → reduced.pt를 시작점으로 로드
    → checkpoint_last/best.pt (재개용) + token_pruned_best.pt (배포용)
```

동시 진행(같은 학습 루프에서 channel pruning + token pruning)이 아니라 **순차
2단계**로 설계했다. 이유:

- Channel pruning만으로도 ViT-Small에서 epoch 20 근처 collapse가 관찰됐고
  (IMPLEMENTATION.md §13-❹), Taylor EMA로 겨우 안정화했다. 여기에 토큰 단위
  동적 변화까지 같은 루프에서 동시에 켜면 val/top1 하락의 원인이 sparsity
  스케줄 때문인지 keep_rate 스케줄 때문인지 구분할 수 없다.
- `reduce.py` 완료 후 결과물은 이미 검증된 안정적인 Dense 모델이다. EViT 원
  논문도 잘 학습된 dense backbone 위에 token pruning을 얹어 짧게 fine-tuning하는
  방식을 쓴다 — 여기서는 그 backbone 자리에 "channel-pruned dense 모델"을
  놓은 것뿐이다.
- 두 메커니즘은 서로 다른 텐서 축(FFN width vs. sequence length)에서 작동하고,
  channel pruning은 QKV/proj를 건드리지 않으므로(IMPLEMENTATION.md §4의
  G_QKV, G_PROJ는 pruning 대상이 아님) embed_dim이 그대로 유지돼 순서 종속성
  문제가 없다.

---

## 3. 알고리즘 — CLS Attention 기반 Token 선택 + Fusion

선택된 일부 block(기본: `depth // 4` 등분 지점, 12-block 모델이면 block 3, 6, 9)에서:

```
① CLS attention score 계산 (attn.forward는 건드리지 않고 별도 계산)
   x_norm = norm1(x)
   attn_out = attn(x_norm)                    ← 원본 그대로, fused_attn 유지
   cls_attn = CLS→patch attention score        ← qkv Linear 재사용해 별도 계산
                                                  (B, N-1), head 평균, softmax

② 잔차 연결 (원본과 동일)
   x = x + drop_path1(ls1(attn_out))

③ Token 선택 (keep_rate < 1인 block만)
   n_keep = ceil((N-1) × keep_rate)            ← 모든 입력에 대해 동일한 상수!
   top-k index = topk(cls_attn, n_keep)         ← "어떤" 토큰인지는 입력마다 다름
   x_kept = gather(x[:, 1:], top-k index)

④ Fusion (fuse_token=True 시)
   버려지는 (N-1-n_keep)개 토큰을 각자의 cls_attn 가중치로 가중합 →
   fused_token 1개로 합쳐서 유지 (완전 폐기 대비 정보 손실 최소화)
   x = cat([CLS, x_kept, fused_token])
   → 다음 block으로 넘어가는 시퀀스 길이 = 1(CLS) + n_keep + 1(fused) = n_keep + 2

⑤ MLP (원본과 동일, 단 짧아진 시퀀스에 대해)
   x = x + drop_path2(ls2(mlp(norm2(x))))
```

`keep_rate`가 고정 비율이고 패치 개수 N도 고정(입력 해상도 224 고정)이므로
**n_keep은 모든 입력에 대해 동일한 컴파일타임 상수**다 — 텐서 shape은 완전히
정적이고, `topk`가 고르는 인덱스 "값"만 입력마다 다르다. DynamicViT류(threshold
기반이라 샘플마다 남는 토큰 "개수" 자체가 다름)보다 ONNX/NPU 컴파일에 훨씬
우호적인 이유가 이것이다 — 다만 실제로는 이 정적 shape 조건 하나만으로 NPU
컴파일이 보장되지는 않았다 (§9 참고).

---

## 4. Progressive Keep Rate 스케줄

Stage 1의 progressive sparsity(IMPLEMENTATION.md §2, Zhu & Gupta cubic
ease-out)와 동일한 형태를 keep_rate에 적용했다:

```python
# pruning/token_pruning.py: EvitTokenPruner._scheduled_keep_rate()
if epoch < warmup_epochs:
    keep_rate = 1.0                              # pruning 없음, 정상 학습
elif epoch < warmup_epochs + ramp_epochs:
    progress = (epoch - warmup_epochs) / ramp_epochs
    drop = 1.0 - base_keep_rate
    keep_rate = 1.0 - drop × (1 - (1-progress)³)  # cubic ease-out
else:
    keep_rate = base_keep_rate                    # 목표치 유지
```

`ViTPruner`(channel pruning)와의 핵심 차이: `EvitTokenPruner`는 **weight를
마스킹하지 않는다.** 순수하게 forward 시점의 시퀀스 길이만 바꾸는 것이라
`optimizer.step()` 이후에 호출할 `.apply()`가 없다 — 매 epoch 시작 시
`set_epoch()`만 호출하면 된다 (`pruner.apply()` 같은 매 step 마스크 재적용이
필요 없음).

### 4.1 `is_best` 판정 — Stage 1과 동일한 함정을 Stage 2에서도 발견

`vit_tiny_30_final`로 첫 Stage 2 run을 돌려보니, `val/top1_best`가 epoch 3
(=`keep_rate_warmup_epochs` 직후, keep_rate가 아직 1.0에 가까운 시점)에서
73.6%로 찍힌 뒤 나머지 26 epoch 내내 한 번도 안 움직였다 — IMPLEMENTATION.md
§13-❽에서 발견한 channel pruning의 `checkpoint_best.pt` 함정이 token pruning에도
그대로 재현된 것이다. `train_token_pruning.py`도 원래는 `train.py`와 동일하게
"keep_rate 상태 무관하게 그냥 val_top1 최고값"만 봤기 때문이다.

**수정** (`train_token_pruning.py` 학습 루프):
```python
ramp_end = token_pruner.warmup_epochs + token_pruner.ramp_epochs
fully_pruned = epoch >= ramp_end
is_best = fully_pruned and acc1 > best_acc1   # ramp 끝난 이후 epoch만 후보
```
`keep_rate_warmup_epochs`/`keep_rate_ramp_epochs`가 모두 0(non-progressive)이면
`ramp_end=0`이라 모든 epoch이 후보가 되어 기존과 동일하게 동작한다.

**엣지 케이스**: `epochs <= warmup_epochs + ramp_epochs`로 설정하면 `fully_pruned`인
epoch이 학습 루프 안에 하나도 안 생겨서 `is_best`가 끝까지 한 번도 안 켜지고,
`checkpoint_best.pt`/`token_pruned_best.pt` 자체가 생성되지 않는다
(`checkpoint_last.pt`는 항상 저장됨). 지금 쓰는 4개 config는 `epochs=30`,
`ramp_end=15`라 15 epoch 여유가 있어 안전하다.

**이미 이 문제로 오염된 run 복구**: `token_pruned_best.pt`가 이미 (수정 전 코드로)
epoch 3 상태로 저장돼버린 경우, `checkpoint_last.pt`(마지막 epoch, keep_rate=target
도달)의 EMA weight로 배포용 아티팩트를 직접 재구성해야 한다:
```python
import torch
ckpt         = torch.load("checkpoint_last.pt", map_location="cpu", weights_only=False)
reduced_ckpt = torch.load("../reduced.pt", map_location="cpu", weights_only=False)  # Stage 1 산출물
tp = ckpt["token_pruner"]
torch.save({
    "state_dict":      ckpt["model_ema"],
    "model_name":      reduced_ckpt["model_name"],
    "mlp_dims":        reduced_ckpt["mlp_dims"],
    "token_pruning": {
        "prune_layers":   tp["prune_layers"],
        "base_keep_rate": tp["keep_rate"],       # 실제 도달한 값(=target)
        "fuse_token":     tp["fuse_token"],
    },
    "n_params_before": reduced_ckpt.get("n_params_before", 0),
    "n_params_after":  reduced_ckpt.get("n_params_after", 0),
}, "token_pruned_last.pt")
```
이렇게 만든 `token_pruned_last.pt`는 `eval_token_pruned.py`/`export_onnx.py`가
`token_pruned_best.pt`와 동일하게 로드할 수 있는 포맷이다.

---

## 5. Knowledge Distillation — Teacher 선택

Stage 2는 학습 가능한 파라미터를 추가하지 않는다(DynamicViT의 learned predictor와
달리, EViT는 기존 attention을 그대로 재활용하는 training-free 선택 방식). 대신
KD로 정확도 회복을 돕는다. `--kd-teacher-mode`로 두 가지를 지원:

| 모드 | Teacher | 특징 |
|------|---------|------|
| `reduced` (기본값) | token pruning 적용 **전**의 동일 reduced 모델 (전체 토큰 사용) | Self-distillation. Stage 1 정확도를 기준점으로 삼아 token pruning으로 인한 손실만 회복하도록 유도 |
| `original` | 원본 pretrained dense 모델 | 더 강한 teacher지만 이미 Stage 1에서 한 번 압축된 student와 capacity gap이 커서 신호가 덜 직접적 |

기본값(`reduced`)을 권장한다 — "이 reduce된 모델이 토큰을 줄이기 전엔 냈던
성능"을 직접 타깃으로 삼는 게 가장 직접적인 신호이기 때문이다.

---

## 6. WandB 프로젝트 분리

Stage 1(channel pruning, `train.py`/`eval_reduced.py`)과 Stage 2(token pruning)는
서로 다른 WandB 프로젝트를 쓴다 — 압축 축이 달라서 같은 프로젝트에 섞으면 비교가
헷갈리기 때문:

| 스크립트 | 기본 `--wandb-project` |
|----------|------------------------|
| `train.py`, `eval_reduced.py` (Stage 1) | `vit-pruning` |
| `train_token_pruning.py`, `eval_token_pruned.py` (Stage 2) | `vit-token-pruning` |

`configs/*_token_prune70.yaml` 4개 모두 `wandb_project: vit-token-pruning`이 이미
박혀 있어서 별도 인자 없이 `--config`만 써도 자동으로 새 프로젝트로 간다.

---

## 7. 실행 명령어

`_final` 네이밍 컨벤션(IMPLEMENTATION.md §1 참고) 기준. 동일 서버에서 여러 job을
동시에 돌릴 때는 `torchrun`에 `--master_port`를 job마다 다르게 지정해야
포트 충돌이 안 난다 (기본 29500 하나만 쓰면 두 번째 torchrun이 바인딩 실패).

```bash
# Stage 1 → Reduce (checkpoint_best.pt가 IMPLEMENTATION.md §13-❽ 문제로 오염됐으면
# checkpoint_last.pt 사용)
python reduce.py --model vit_tiny_patch16_224 \
  --checkpoint ./output/vit_tiny_30_final/checkpoint_last.pt \
  --output     ./output/vit_tiny_30_final/reduced.pt

# Stage 2 — Token Pruning fine-tuning (GPU 여러 개 동시 돌릴 땐 --master_port 다르게)
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29501 train_token_pruning.py \
  --config configs/vit_tiny_30_token_prune70.yaml

# 평가 (§4.1 문제로 token_pruned_best.pt가 오염됐으면 token_pruned_last.pt 사용)
python eval_token_pruned.py \
  --token-pruned ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt \
  --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \
  --gpu 4 --wandb

# ONNX 변환 (reduced.pt / token_pruned_*.pt 공용, "token_pruning" 키로 자동 판별)
# --output 생략 시 vit_tiny_c30_reduced.onnx / vit_tiny_c30_token70.onnx로 자동 네이밍
# (IMPLEMENTATION.md §9 참고)
python export_onnx.py --input ./output/vit_tiny_30_final/reduced.pt --verify

python export_onnx.py \
  --input ./output/vit_tiny_30_final/token_prune70/token_pruned_last.pt --verify
```

---

## 8. 체크포인트 로드 방법

`token_pruned_best.pt` / `token_pruned_last.pt` 둘 다 포맷이 동일하다 (`best`는
정상적으로 post-ramp에서 갱신된 경우, `last`는 §4.1 문제로 수동 재구성한 경우) —
로드 코드는 똑같다.

```python
import torch, timm
from pruning.vit_reducing import apply_reduced_config
from pruning.token_pruning import apply_token_pruning

ckpt   = torch.load("token_pruned_best.pt", map_location="cpu")  # 또는 token_pruned_last.pt
model  = timm.create_model(ckpt["model_name"], pretrained=False)
apply_reduced_config(model, ckpt["mlp_dims"])                 # Stage 1 구조 축소

tp_cfg = ckpt["token_pruning"]
apply_token_pruning(
    model,
    prune_layers=tp_cfg["prune_layers"],
    base_keep_rate=tp_cfg["base_keep_rate"],
    fuse_token=tp_cfg["fuse_token"],
)                                                               # Stage 2 forward 패치
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## 9. NPU 배포

### 9.1 개요 — TopK + Gather 리스크

전체 파이프라인에서 가장 리스크가 큰 지점이다. Channel pruning 결과물(reduced
모델)은 그냥 더 작은 표준 ViT라 ONNX/NPU 변환에 특별한 리스크가 없지만, token
pruning은 그래프에 `TopK`와 **런타임에 계산된 인덱스로 하는 Gather**를 추가한다.
이건 conv/matmul/elementwise 위주로 설계된 edge NPU 컴파일러 상당수가 지원하지
않거나 CPU fallback으로 빠지는 연산 패턴이다.

§3에서 설명했듯 **텐서 shape 자체는 완전히 정적**이다(keep_rate가 고정 비율이라
n_keep이 상수). 이 조건 덕분에 Mobilint NPU 컴파일러 통과 여부를 사전에
확인했다 — TopK/Gather op 자체가 컴파일 가능함을 확인한 뒤에 이 Stage 2 구현에
착수했다. (**후속**: 이건 최소 toy 그래프 테스트였다 — 실제 구현이 쓰는 정확한
연산까지는 검증하지 못했고, 실제로 문제가 됐다. §9.2 참고.)

**향후 다른 NPU 타겟으로 이식할 때는 반드시 이 순서를 지킬 것**: 전체
fine-tuning을 다 돌리기 전에, 실제 구현이 쓰는 정확한 op(§9.2 참고 — 단순
`Gather`가 아니라 `GatherElements`/`Equal` 등)이 들어간 최소 toy ONNX 그래프를
먼저 그 컴파일러에 넣어보고 통과 여부(및 실제 온칩 실행인지, CPU fallback인지)를
확인한다.

### 9.2 실제 NPU(Aries2 / qbcompiler) 컴파일 실패 사례와 해결 시도

`vit_tiny_c30_token70.onnx`(Stage 2 산출물)를 Mobilint qbcompiler에 넣었더니
컴파일이 실패했다. 원인 3가지:

| 문제 | 원인 |
|------|------|
| `ScatterElements` 미지원 (blocks 3, 6, 9) | `_complement_idx()`가 `torch.scatter`로 "top-k에 안 뽑힌 나머지 인덱스"를 구하는 트릭을 씀 |
| `GatherElements` 미지원 (9개 노드, 전부 Unsupported) | `torch.gather(dim=1, index=(B,k,C))`가 배치마다 다른 인덱스를 한 번에 처리하려고 element-wise gather로 export됨 — §9.1에서 확인한 단순 `Gather`(index_select류)와는 다른 op |
| ONNX 그래프 output이 1개가 아니라 7개 | blocks 3/6/9의 `TopK` 중간 결과(값+인덱스)가 그래프 output으로 노출됨 → quantizer가 output 1개를 가정하는데 7개가 나와서 calibration 대상을 못 정함 → `modelInputNames` 개수 불일치로 quantize 자체가 실패 |

핵심 교훈: **"TopK/Gather가 컴파일된다"와 "이 구현이 실제로 쓰는 TopK/Gather 변형이
컴파일된다"는 다른 이야기다.** `torch.gather`/`torch.scatter`는 배치별·요소별
동적 인덱스를 다루는 `GatherElements`/`ScatterElements`로 export되는데, 이건
단순 "N개 중 K개를 인덱스 리스트로 뽑는" `Gather`보다 훨씬 지원이 안 되는
연산이다. 다음에 다른 NPU 타겟으로 이식할 때는 최소 toy 그래프도 실제 코드가
쓰는 정확한 op(이 경우 `GatherElements`/`ScatterElements`)로 맞춰서 테스트해야
한다.

**해결 시도 — 알고리즘은 그대로 두고 구현만 재작성** (`pruning/token_pruning_npu.py`):

1. **ScatterElements 제거**: `cls_attn`이 이미 정렬 가능한 점수 배열이므로,
   "top-k에 안 뽑힌 나머지"는 `_complement_idx()`(scatter+sort 트릭) 없이
   그냥 `largest=False`인 두 번째 `TopK`로 바로 구할 수 있다.
   ```python
   _, idx   = torch.topk(cls_attn, n_keep, largest=True)
   _, compl = torch.topk(cls_attn, n_patch - n_keep, largest=False)  # scatter 불필요
   ```
   결과는 수학적으로 완전히 동일하다 — 학습 중에도 항상 이 버전을 쓴다(정확도에
   영향 없음, 그래프만 단순해짐). **결과: 성공.** 실제 컴파일 로그에
   `ScatterElements`가 더 이상 나타나지 않음을 확인.

2. **GatherElements → 배치=1이면 plain Gather**: NPU 추론은 거의 항상 batch=1
   (streaming)이므로, export 시점에만 배치 축을 떼어내고
   `torch.index_select(dim=0, index=1D)`로 바꾼다. 이건 "N개 중 K개 행을
   1차원 인덱스로 뽑기"라는 단순 연산이라 `Gather`로 export된다. 학습(batch>1)
   중에는 여전히 기존 `torch.gather`를 쓴다 — `block._evit_npu_export` 플래그로
   전환한다. **결과: 실패.** `GatherElements`는 그래프에서 사라졌지만, 실제
   컴파일 로그에서 plain `Gather`도 9개 전부 `Unsupported(0%)`로 나왔다. op
   이름 문제가 아니라 **런타임에 계산된 인덱스로 하는 gather 자체가 Aries2에서
   안 된다**는 뜻.

3. **output 7개 → dynamo=False**: `/blocks/blocks.3/TopK_output_0` 같은
   네이밍은 torch 2.x의 dynamo 기반 ONNX exporter 흔적으로 보인다. **결과:
   애초에 문제가 아니었음.** `[parser]` 로그가 HL 컴파일 전/후 두 번 나오는데,
   HL 컴파일 후에는 qbcompiler가 자체적으로 안 쓰이는 TopK 중간 출력을
   dead-code로 정리해서 이미 output 1개(`output`, (1,1000))였다. 실제 최종
   실패 원인은 output 개수가 아니라 2번(Gather 9개 unsupported)이 CPU
   fallback되면서 그 경계 텐서들이 quantizer 기준 "input"으로 잘못 잡히는
   것(`modelInputNames expected 1 got 10`)이었다 — output 개수와는 무관.

4. **(추가 시도) Gather를 One-hot + MatMul로 대체**: 2번이 하드 블로커로
   보이자, gather를 아예 없애는 대신 수학적으로 동일한 형태로 표현을 바꿔봤다.
   `gather(src, idx)[i] == src[idx[i]]`는 `onehot[i,j]=1 if j==idx[i] else 0`
   일 때 `(onehot @ src)[i] == src[idx[i]]`와 같다 — `Equal`(onehot 생성) +
   `Cast` + `MatMul`. `mode="onehot_matmul"`로 `pruning/token_pruning_npu.py`에
   구현.
   **결과: 부분 성공 + 근본 벽 재확인.** `Gather`는 그래프에서 완전히
   사라졌고 `MatMul`은 33개 중 33개(100%) Supported로 확인됐다 — one-hot
   행렬곱 자체는 Aries2가 잘 받는다. 다만:
   - **버그 하나 발견**: `(k,N)@(N,)` 형태(2D×1D, `non_topk_attn` 계산)의
     matmul 3개가 `matmul shape not implemented for (58,196), (196,)`로
     변환 실패. `(N,C)`/`(N,1)` 형태(2D×2D) 30개는 전부 정상이었다. 원인은
     구현 버그(1D 벡터를 그대로 matmul에 넣음) — `_onehot_gather()`에서
     1D 입력을 `(N,1)`로 unsqueeze해서 항상 2D×2D로 맞추도록 수정함.
   - **더 근본적인 문제**: 이 버그와 별개로, **`Equal`(TensorCmp)과 `Cast`가
     9개 전부 여전히 `Unsupported(0%)`**로 나왔다. `idx`(top-k 결과)가
     런타임 값이라 `onehot = (idx == arange_n)` 단계에서부터 막힌다. 이번엔
     quantizer의 정상 에러가 아니라 파싱 단계 크래시(`[EXC] ... what=map::at`)
     까지 발생 — 미지원 서브그래프 구조가 컴파일러 내부 로직 자체를 못
     감당하는 것으로 보인다.

**결론: `Gather`(직접 인덱싱)와 `Equal`(비교 기반 인덱싱) 둘 다 런타임 값이
들어가면 실패한다.** ONNX에서 "런타임 값으로 무언가를 선택/비교"하는 걸 표현하는
가장 기본적인 두 가지 방법을 다 시도해본 셈이라, 더 단순화할 op이 마땅치 않다.
이 시점부터는 "구현을 더 파봐야 하는 문제"가 아니라 **"Aries2가 이 클래스의
연산(데이터 의존적 인덱싱/비교) 자체를 지원하지 않는다"는 결론에 훨씬 가깝다.**

**남은 선택지:**
- Mobilint에 이 두 번의 실측 결과(GatherElements/Gather/Equal/Cast 전부
  런타임 입력에서 실패)를 들고 직접 문의 — 혹시 지원되는 특정 패턴이 있는지
  확인
- **알고리즘을 static/global token pruning으로 전환** — 이미지마다 다른
  토큰을 고르는 게 아니라 학습 후 고정된 위치를 항상 잘라내는 방식으로
  바꾸면 런타임 인덱싱 자체가 사라져(순수 `Slice`, 컴파일타임에 인덱스 확정)
  어떤 NPU에서도 컴파일된다. EViT의 핵심 강점(이미지별 적응)은 잃지만, 이
  시점에서는 사실상 유일하게 남은 실질적인 NPU 배포 경로로 보인다.
- `_reduced.onnx`(Stage 1, channel pruning만)만 배포 — 확실히 되는 안전한 기본선

**기존 파이프라인은 전혀 안 건드림 — 격리된 실험 경로로 추가:**

- `pruning/token_pruning_npu.py` (신규 파일) — 위 1, 2, 4를 구현한
  `_evit_block_forward_npu` (`_evit_npu_mode`로 index_select/onehot_matmul 선택)
- `pruning/token_pruning.py`의 `apply_token_pruning()`/`EvitTokenPruner`에
  `forward_fn` 파라미터 추가 (기본값 = 기존 함수라 다른 3개 run에 영향 없음)
- `train_token_pruning.py --npu-safe` 플래그 (기본 False) — 켜면 위 forward를
  쓰고, 저장되는 `token_pruned_best.pt`에 `"npu_safe": true`를 기록해둠
- `export_onnx.py`가 그 `npu_safe` 값을 자동 감지해서 NPU forward + batch=1
  강제 + `dynamo=False`(고침 3)를 자동 적용, `--npu-mode`로 index_select/
  onehot_matmul 선택. export 직후 그래프 output 개수와 `ScatterElements`/
  `GatherElements`/(onehot_matmul 모드면)`Gather` 잔존 여부, `Equal`/`MatMul`
  존재 여부를 바로 출력해서 qbcompiler를 다시 돌리기 전에 1차로 확인 가능하다.
- `configs/vit_tiny_30_token_prune70_npu_test.yaml` — tiny 30% 기반, **10 epoch만**
  (컴파일 가능 여부 확인이 목적이지 정확도 검증이 아님), `output_dir`을
  `token_prune70_npu_test/`로 분리해 정식 30-epoch run을 안 건드림. 재학습
  없이(NPU 우회는 export 시점에만 적용되므로) 이 하나의 checkpoint로 mode만
  바꿔가며 여러 번 재검증했다.

---

## 10. 주의사항 & 트러블슈팅

- **표준 단일 CLS 토큰 ViT만 지원.** `dist_token`이 있는 distilled 모델이나
  `no_embed_class` 변형은 `pruning/token_pruning.py`의 `_validate_model()`에서
  즉시 예외를 던진다 (조용히 잘못된 결과를 내지 않도록).
- **timm 버전이 바뀌면 `Attention` 내부 속성명(`qkv`, `num_heads`, `q_norm`,
  `k_norm`)이 달라질 수 있다.** `_validate_model()`이 `hasattr` 체크로 조기
  실패하지만, 정확한 CLS attention score 계산을 보장하려면 timm 업그레이드 시
  `_cls_attention_scores()`를 다시 확인해야 한다.
- **실제로 겪은 timm 버전 호환성 문제**: 서버에 설치된 timm의 `Attention.forward()`가
  `is_causal` 키워드 인자를 안 받아서 `TypeError: Attention.forward() got an
  unexpected keyword argument 'is_causal'`가 났다 (GitHub의 최신 `vision_transformer.py`
  소스와 실제 설치 버전이 미묘하게 다름). 이 repo의 ViT 분류 파이프라인은
  `attn_mask`/`is_causal`을 애초에 안 쓰므로(항상 `None`/`False`),
  `_evit_block_forward()`에서 이 kwarg들을 `self.attn(...)`에 아예 전달하지
  않도록 고쳐서 해결했다 — 대신 `attn_mask is not None or is_causal`이면
  `NotImplementedError`로 조기 실패한다. **timm을 업그레이드하거나 다른 서버에
  이식할 때 이 부분이 다시 깨질 수 있으니, 처음 실행할 때는 아래 스모크 테스트로
  먼저 검증할 것** (실제 학습 3 epoch을 기다리지 않고 forward+backward만 즉시 확인):
  ```python
  import torch, timm
  from pruning.token_pruning import apply_token_pruning

  model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
  apply_token_pruning(model, prune_layers=[3, 6, 9], base_keep_rate=0.7, fuse_token=True)

  model.eval()
  with torch.no_grad():
      out = model(torch.randn(2, 3, 224, 224))
  print("forward OK:", tuple(out.shape))

  model.train()
  model(torch.randn(2, 3, 224, 224)).sum().backward()
  print("backward OK:", model.blocks[3].attn.qkv.weight.grad is not None)
  ```
- **`is_best`가 keep_rate ramp 종료 이전 epoch을 잘못 채택하는 문제 — 겪었고
  수정함.** §4.1 참고. 이미 오염된 run은 `checkpoint_last.pt`에서
  `token_pruned_last.pt`를 수동으로 재구성해야 한다.
- **`export_onnx.py`의 CLI 인자가 `--reduced` → `--input`으로 바뀌었다** (Stage 1
  전용이던 로더를 Stage 1/2 공용으로 바꾸면서 이름을 일반화함). 기존 스크립트/
  문서에 `--reduced`로 남아있는 부분은 `--input`으로 교체해야 한다.
- **Stage 2에는 channel pruning(`ViTPruner`)이 관여하지 않는다.** `mlp_dims`는
  Stage 1에서 결정된 그대로 유지되고, Stage 2는 순수하게 시퀀스 길이만 바꾼다.

---

*작성: 2026-07 | Stage 2 EViT Token Pruning — IMPLEMENTATION.md §14에서 분리 이전*
*업데이트: 2026-08 | is_best가 keep_rate ramp 종료 전 epoch을 잘못 채택하는 문제 발견 및 수정(§4.1),*
*timm `is_causal` 호환성 이슈 수정(§10), WandB 프로젝트 분리(§6), tiny/small 30% config 추가*
*업데이트: 2026-08 | Aries2/qbcompiler 컴파일 실패(ScatterElements/GatherElements 미지원,*
*output 7개 노출) 발견 및 격리된 해결 시도 — `pruning/token_pruning_npu.py`,*
*`train_token_pruning.py --npu-safe`, `export_onnx.py` 자동 감지 추가 (§9.2)*
*업데이트: 2026-08 | §9.2 실측 결과 확정 — index_select(Gather)와 onehot_matmul*
*(Equal+Cast+MatMul) 둘 다 런타임 인덱스 관련 op에서 실패. "Aries2가 데이터 의존적*
*인덱싱을 지원 안 함"으로 결론, static/global token pruning 전환을 다음 단계로 제시*
