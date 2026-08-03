# Patch Slimming for Efficient Vision Transformers 구현 명세서

> 대상 논문: Yehui Tang et al., **“Patch Slimming for Efficient Vision Transformers,” CVPR 2022**  
> 목적: 기존 `npu pruning` 프로젝트에서 Claude Code가 정적 Patch Slimming(PS-ViT)을 구현할 수 있도록, 논문의 알고리즘과 필요한 공학적 결정을 하나의 문서로 고정한다.

---

## 0. 이 문서의 사용 원칙

이 문서는 논문과 함께 제공한다. 구현 시 우선순위는 다음과 같다.

1. 이 문서에 명시된 구현 동작과 데이터 구조
2. 원 논문의 수식, Algorithm 1, 실험 설정
3. 기존 `npu pruning` 프로젝트의 코딩 규칙과 모델 추상화

논문에 명시된 사실과, 논문에 없는 부분을 구현 가능하게 만들기 위해 이 문서가 선택한 공학적 결정을 반드시 구분한다.

- **[PAPER]**: 논문 본문에서 직접 확인되는 내용
- **[DECISION]**: 논문에 세부 구현이 없어서 이 문서가 고정하는 구현 선택
- **[OPTION]**: 기본 구현 이후 비교 실험을 위해 선택적으로 지원할 항목

Claude Code는 논문에 없는 세부사항을 임의로 조용히 결정하지 말고, 이 문서의 `[DECISION]`을 따른다. 기존 프로젝트 구조와 충돌하면 구현 전에 충돌 지점만 보고하고, 알고리즘을 다른 token-pruning heuristic으로 대체하지 않는다.

---

# 1. 구현 목표와 범위

## 1.1 1차 구현 목표

사전학습된 ViT/DeiT 분류 모델에 대해 다음 파이프라인을 구현한다.

```text
Pretrained ViT
  → attention/feature 계측
  → 데이터셋 기반 patch significance score 계산
  → 마지막 레이어부터 첫 레이어 방향의 mask 탐색
  → 현재 block 단위 reconstruction fine-tuning
  → 레이어별 고정 patch mask 확정
  → compact PS-ViT 생성
  → 전체 classification fine-tuning
  → 고정 shape NPU inference graph export
```

최종 정적 PS-ViT는 inference 때 입력 이미지별 importance를 계산하지 않는다. 레이어별 patch index는 offline search 단계에서 고정된다.

## 1.2 1차 구현 범위

- 정적 **PS-ViT**만 구현한다.
- image classification용 ViT/DeiT를 대상으로 한다.
- 고정 입력 해상도와 고정 patch grid를 사용한다.
- 첫 기준 모델은 non-distilled DeiT-Tiny 또는 DeiT-Small을 권장한다.
- CLS token을 classifier output token으로 사용한다.
- mask search, block reconstruction training, 최종 fine-tuning, compact inference를 분리한다.
- 최종 graph는 실제로 token tensor 길이를 줄여야 한다.
- NPU export를 고려하여 모든 레이어의 입력·출력 token 수를 정적으로 확정한다.

## 1.3 1차 구현에서 제외할 내용

- 동적 DPS-ViT
- 입력 이미지별 TopK/gating
- detection, segmentation 등 dense prediction task
- patch embedding 자체의 parameter pruning
- head pruning, channel pruning, weight pruning
- quantization-aware training
- token merging
- 가변 해상도와 가변 patch grid
- distilled DeiT의 두 개 classifier token 처리

추가 prefix/register token이 있는 모델을 사용할 경우, 첫 구현에서는 오류를 발생시키거나 모든 prefix token을 항상 보존하는 명시적 adapter를 둔다. 논문 재현 기준 모델은 prefix token이 CLS 하나인 모델로 제한한다.

---

# 2. 논문의 핵심 동작

## 2.1 Patch Slimming이 줄이는 것

**[PAPER]** Patch Slimming은 ViT의 weight matrix나 embedding dimension을 줄이지 않는다. 레이어마다 계산되는 patch token 수를 줄인다.

- Q/K/V와 MLP weight shape는 유지된다.
- parameter count와 모델 파일 크기는 거의 줄지 않는다.
- attention 및 MLP의 activation과 연산량이 감소한다.
- 깊은 레이어일수록 더 많은 patch를 제거하는 피라미드 구조가 만들어진다.

## 2.2 Top-down mask search

**[PAPER]** 마지막 출력 레이어에서 필요한 token을 먼저 정한 뒤, 이전 레이어로 이동하면서 필요한 patch를 추가한다.

분류용 ViT에서는 마지막 layer의 유효 output을 CLS token으로 둔다.

\[
m_{L,\mathrm{CLS}}=1,\qquad m_{L,i}=0\;(i\neq \mathrm{CLS})
\]

그 뒤:

\[
m_{l+1}\subseteq m_l
\]

를 유지한다. 즉 깊은 레이어에서 유지한 patch는 모든 이전 레이어에서도 유지되어야 한다.

## 2.3 데이터셋 통계 기반 significance score

**[PAPER]** 각 layer를 pruning하기 전에 training dataset의 일부를 무작위로 샘플링한다. 각 이미지에서 patch significance score를 계산한 뒤 데이터 전체 평균을 사용한다.

\[
\bar s_{l,i}=\frac{1}{M}\sum_{x\in\mathcal D_{cal}}s_{l,i}(x)
\]

정적 PS-ViT에서는 이 평균 score에 따라 모든 이미지에 공통으로 적용할 고정 spatial mask를 결정한다.

## 2.4 Patch 수 결정

**[PAPER]** 어느 patch를 우선 보존할지는 significance score 순위로 결정하고, 몇 개를 남길지는 reconstruction error 허용치 \(\epsilon\)으로 결정한다.

현재 layer에서 score가 큰 patch를 \(r'\)개 단위로 추가하고, 다음 layer의 유효 feature reconstruction error가 \(\epsilon\) 이하가 될 때 멈춘다.

논문 실험 설정:

- \(\epsilon\in\{0.01,0.02\}\)
- search granularity \(r'=10\)
- 후보 mask를 확장할 때마다 현재 block 3 epochs fine-tuning
- 모든 layer mask 확정 후 전체 모델 fine-tuning

---

# 3. 용어와 인덱싱 규칙

인덱싱 오류가 가장 치명적이므로 아래 규칙을 코드 전체에서 고정한다.

## 3.1 Layer index

코드에서는 Transformer block을 0-based로 표기한다.

```text
blocks[0], blocks[1], ..., blocks[L-1]
```

`keep_ids[l]`은 **block `l`의 출력으로 유지되어 block `l+1`에 전달되는 global token ID**를 의미한다.

- `keep_ids[L-1]`: 마지막 block 출력에서 유지되는 token. 분류 모델에서는 `[CLS_ID]`.
- mask search 대상: `l = L-2, L-3, ..., 0`
- `keep_ids[l+1]`이 정해진 상태에서 `keep_ids[l]`을 찾는다.

## 3.2 Global token ID

Global token ID는 원본 unpruned sequence에서의 영구적인 위치다.

예: 224×224, patch 16×16, CLS 한 개인 DeiT

```text
global token 0       = CLS
patch global 1       = image patch 0
patch global 2       = image patch 1
...
patch global 196     = image patch 195
총 N = 197
```

모든 mask, score, serialization은 global token ID 기준으로 관리한다.

## 3.3 Local token index

Compact tensor 안에서의 현재 위치다.

예:

```text
active_global_ids = [0, 3, 7, 10]
local indices      = [0, 1, 2, 3]
```

`next_keep_global_ids=[0,7]`이면 현재 tensor에서의 local index는 `[0,2]`다.

코드에는 다음 함수가 반드시 있어야 한다.

```python
def global_to_local(
    active_global_ids: Tensor,
    requested_global_ids: Tensor,
) -> Tensor:
    """requested_global_ids가 active_global_ids의 부분집합인지 검증하고 local index를 반환."""
```

## 3.4 Mask invariant

모든 layer에서 다음을 검사한다.

```python
assert CLS_ID in keep_ids[l]
assert set(keep_ids[l + 1]).issubset(set(keep_ids[l]))
assert keep_ids[l] is sorted and unique
```

---

# 4. 저장해야 할 핵심 데이터 구조

## 4.1 Architecture specification

검색 결과는 모델 weight와 별도로 JSON 또는 YAML로 저장한다.

```json
{
  "method": "patch_slimming_static",
  "model_name": "deit_small_patch16_224",
  "num_blocks": 12,
  "num_global_tokens": 197,
  "num_prefix_tokens": 1,
  "classifier_token_ids": [0],
  "epsilon": 0.02,
  "error_metric": "mse",
  "search_step": 10,
  "score_mode": "paper_path_energy_dp",
  "layers": [
    {
      "block_index": 0,
      "keep_global_ids": [0, 1, 2],
      "num_output_tokens": 3,
      "accepted_error": 0.0
    }
  ]
}
```

실제 `keep_global_ids`는 전체 값을 저장한다.

## 4.2 Per-layer search record

각 후보마다 다음을 기록한다.

```text
layer index
candidate extra token count r
candidate keep IDs
train reconstruction loss by epoch
validation reconstruction error
score min/max/mean
candidate training time
accepted/rejected
```

CSV와 JSON Lines 중 프로젝트 표준에 맞춰 하나를 선택한다.

## 4.3 Checkpoint

최소한 다음 checkpoint를 분리한다.

```text
baseline_pretrained.pt
search_progress_layer_<l>.pt
searched_student.pt
final_finetuned_compact.pt
architecture.json
```

중간 중단 후 layer 단위로 재개할 수 있어야 한다.

---

# 5. ViT block의 compact 연산 정의

## 5.1 논문에 해당하는 token 흐름

현재 block `l`의 입력 active token 수를 \(N_{in}\), 출력으로 유지할 token 수를 \(N_{out}\)이라 한다.

\[
N_{out}\le N_{in}
\]

논문의 pruned MSA는 다음 의미로 구현한다.

- Query: 출력으로 유지할 \(N_{out}\)개 token에서만 계산
- Key/Value: 현재 block 입력의 \(N_{in}\)개 token 전체에서 계산
- attention map: \(N_{out}\times N_{in}\)
- residual: 입력 tensor에서 output token에 해당하는 row만 선택
- MLP: \(N_{out}\)개 output token에서만 계산

형상:

\[
Q\in\mathbb R^{B\times H\times N_{out}\times d_h}
\]

\[
K,V\in\mathbb R^{B\times H\times N_{in}\times d_h}
\]

\[
A\in\mathbb R^{B\times H\times N_{out}\times N_{in}}
\]

## 5.2 Slim block 인터페이스

```python
class SlimTransformerBlock(nn.Module):
    def forward(
        self,
        x: Tensor,                    # [B, N_in, D]
        active_global_ids: Tensor,    # [N_in]
        output_global_ids: Tensor,    # [N_out], subset of active_global_ids
        return_attention: bool = False,
    ) -> SlimBlockOutput:
        ...
```

반환:

```python
@dataclass
class SlimBlockOutput:
    x: Tensor                         # [B, N_out, D]
    output_global_ids: Tensor         # [N_out]
    attention_probs: Tensor | None    # [B, H, N_out, N_in]
```

## 5.3 Fused QKV weight 처리

기존 timm/프로젝트 ViT가 `qkv = Linear(D, 3D)`를 사용하더라도 weight를 새로 학습하지 않는다. 기존 weight를 논리적으로 분할한다.

```python
Wq = W_qkv[0:D]
Wk = W_qkv[D:2D]
Wv = W_qkv[2D:3D]
```

계산:

```python
x_norm = norm1(x)
query_x = x_norm.index_select(1, output_local_ids)

q = F.linear(query_x, Wq, bq)
k = F.linear(x_norm, Wk, bk)
v = F.linear(x_norm, Wv, bv)
```

NPU export 전에는 Q projection과 KV projection을 별도 operator/weight node로 물리적으로 분리해도 된다. 값은 기존 fused QKV의 slice를 사용한다.

## 5.4 Pre-norm block 동작

DeiT/timm의 일반적인 pre-norm block 기준:

```python
residual = x.index_select(1, output_local_ids)
attn_out = slim_attention(norm1(x), output_local_ids)
x_out = residual + drop_path1(attn_out)
x_out = x_out + drop_path2(mlp(norm2(x_out)))
```

기존 block에 LayerScale, DropPath, attention dropout 등이 있으면 기존 동작을 보존한다.

## 5.5 수치 등가성 요구

Eval mode에서 반드시 다음을 만족해야 한다.

### 전체 token 사용

```python
slim_block(x, output_ids=all_ids).x
≈ original_block(x)
```

### 일부 token 사용

```python
slim_block(x, output_ids=subset).x
≈ original_block(x).index_select(1, subset_local_ids)
```

이 등가성이 성립하는 이유는 attention의 각 query row가 독립적으로 계산되고, MLP가 token별로 독립적으로 적용되기 때문이다.

기본 허용 오차:

```text
FP32: atol=1e-6, rtol=1e-5
FP16/BF16: dtype에 맞춰 완화
```

---

# 6. Attention 및 feature 계측

## 6.1 필요한 값

Patch significance score와 reconstruction을 위해 다음 값이 필요하다.

```text
각 block 입력 feature Z_{l-1}
각 block 출력 feature Z_l
각 head의 post-softmax attention probability P_l^h
현재 active global token IDs
현재 output global token IDs
```

attention logits가 아니라 **softmax 이후 probability**를 저장한다.

## 6.2 Baseline instrumentation 조건

계측 기능을 켜도 baseline logits가 변하면 안 된다.

```python
logits_plain = model(images)
logits_instrumented = model(images, capture=True).logits
assert_allclose(logits_plain, logits_instrumented)
```

전역 monkey patch보다 프로젝트 모델 adapter 또는 block wrapper를 사용한다.

## 6.3 메모리 정책

ImageNet 전체 attention을 저장하지 않는다.

- calibration subset만 사용한다.
- target layer별로 필요한 값을 streaming 계산한다.
- 필요 시 batch별 score를 즉시 누적하고 attention을 해제한다.
- score accumulator는 CPU float64 또는 GPU float64/float32를 사용한다.
- cache를 사용할 경우 sample ID, model checksum, layer, mask checksum을 key로 둔다.

---

# 7. Significance score 구현

## 7.1 논문 수식

논문의 Theorem 1은 target layer \(t\)의 patch \(i\)에 대해 다음 score를 제시한다.

\[
s_{t,i}=\sum_h\left\|A_t^h[:,i]U_t^h[i,:]\right\|_F^2
\]

\[
A_t^h=\prod_{l=t+1}^{L}\operatorname{diag}(m_l)P_l^h
\]

\[
U_t^h=P_t^h|Z_{t-1}|
\]

개념적으로:

```text
patch importance
= 현재 layer에서 해당 patch가 운반하는 feature energy
× 깊은 layer들을 지나 최종 유효 output까지 전달되는 influence
```

## 7.2 논문 표기의 구현상 문제

논문은 여러 layer의 multi-head 경로를 간결한 기호로 표현하지만, 실제 tensor 코드와 head-path 집계 방법은 제공하지 않는다. head 조합을 순진하게 열거하면 \(H^{L-t}\)개의 경로가 생긴다.

**[DECISION]** 1차 구현은 head-path energy 합을 동적 계획법으로 계산한다. 이는 작은 모형에서 brute-force head-path 열거와 수치적으로 같음을 unit test로 검증한다.

## 7.3 Local attention을 global 좌표로 확장

Slim block의 attention은 다음 형상이다.

```text
[B, H, N_out, N_in]
```

score 계산에서는 원본 token 공간 \(N\times N\)에서 행렬을 곱할 수 있도록 global matrix로 scatter한다.

```python
def scatter_attention_to_global(
    attn_local: Tensor,          # [B,H,N_out,N_in]
    output_global_ids: Tensor,   # [N_out]
    input_global_ids: Tensor,    # [N_in]
    num_global_tokens: int,
) -> Tensor:                     # [B,H,N,N]
    ...
```

선택되지 않은 row/column은 0이다. 따라서 mask 효과가 global attention matrix에 이미 반영된다.

## 7.4 Target layer score 계산 시 hybrid network 상태

Target layer `t`의 score를 계산할 때:

```text
blocks < t : 아직 pruning하지 않은 full-token 경로
block  t   : 아직 mask를 정하기 전이므로 full output token
blocks > t : 이미 확정된 keep_ids를 사용하는 slim 경로
```

즉 `t` 이전은 원본 token 수, `t` 이후는 top-down search로 이미 정해진 mask를 사용한다.

## 7.5 Path-energy dynamic programming

각 calibration sample마다 downstream attention influence를 계산한다.

Global attention matrix를 \(P_k^h\in\mathbb R^{N\times N}\)라 하자. 이미 local→global scatter되어 있으며, 제거된 output row와 입력에 없는 column은 0이다.

초기값:

\[
G=I_N
\]

마지막 block부터 target 다음 block까지 역순으로:

\[
G\leftarrow\frac{1}{H}\sum_{h=1}^{H}(P_k^h)^T G P_k^h
\]

반복 구간:

```text
k = L-1, L-2, ..., t+1
```

그 뒤:

\[
downstream_i=G_{ii}
\]

현재 target block의 feature energy:

\[
U_t^h=P_t^h|Z_{t-1}|
\]

\[
current_i=\frac{1}{H}\sum_h\sum_d(U_t^h[i,d])^2
\]

최종 sample score:

\[
s_{t,i}=downstream_i\cdot current_i
\]

head 합 대신 head 평균을 사용하는 것은 target layer 안에서 모든 token에 동일한 양의 상수를 나누는 것이므로 score 순위를 바꾸지 않고 수치 폭주를 줄인다.

### 의사 코드

```python
def compute_sample_scores(
    z_prev: Tensor,                      # [B,N,D]
    target_attn_global: Tensor,          # [B,H,N,N]
    downstream_attn_globals: list[Tensor],
) -> Tensor:                             # [B,N]
    B, H, N, _ = target_attn_global.shape

    eye = torch.eye(N, device=z_prev.device, dtype=z_prev.dtype)
    G = eye.unsqueeze(0).expand(B, -1, -1).clone()  # [B,N,N]

    for P in reversed(downstream_attn_globals):
        # P: [B,H,N,N]
        # head별 P^T G P를 계산한 뒤 평균
        GP = torch.matmul(G.unsqueeze(1), P)
        PtGP = torch.matmul(P.transpose(-2, -1), GP)
        G = PtGP.mean(dim=1)

    downstream_energy = torch.diagonal(G, dim1=-2, dim2=-1)

    U = torch.matmul(
        target_attn_global,
        z_prev.abs().unsqueeze(1),
    )  # [B,H,N,D]

    current_energy = U.square().sum(dim=-1).mean(dim=1)  # [B,N]
    scores = downstream_energy.clamp_min(0) * current_energy
    return scores
```

실제 구현에서는 einsum 또는 batched matmul 중 프로젝트 backend에서 더 효율적인 것을 선택한다.

## 7.6 DP score 검증

작은 synthetic case에서 explicit head-path enumeration과 비교한다.

예:

```text
B=1, N=4, H=2, downstream layers=2
총 downstream head path=2^2=4
```

brute-force로 각 path의 attention product를 만든 score와 DP score가 동일해야 한다. head 평균을 사용하면 동일한 정규화 상수를 brute-force에도 적용한다.

이 unit test가 통과하지 않으면 전체 mask search를 시작하지 않는다.

## 7.7 Dataset 평균 score

```python
score_sum = torch.zeros(N, dtype=torch.float64)
num_samples = 0

for batch in calibration_loader:
    sample_scores = compute_sample_scores(...)
    score_sum += sample_scores.double().sum(dim=0).cpu()
    num_samples += sample_scores.shape[0]

mean_scores = score_sum / num_samples
```

- CLS/prefix token은 후보 ranking에서 제외한다.
- `keep_ids[t+1]`에 이미 포함된 token도 후보 ranking에서 제외한다.
- NaN/Inf를 즉시 오류로 처리한다.
- score 계산 시 model은 eval mode로 둔다.
- dropout, DropPath, augmentation randomness를 비활성화한다.

## 7.8 선택적 근사 모드

**[OPTION]** 성능 또는 메모리 문제로 DP 구현이 너무 무거울 경우 head-averaged attention rollout을 별도 mode로 지원할 수 있다.

```text
score_mode = "head_mean_rollout_approx"
```

그러나 기본 mode와 결과 파일에 반드시 `approx`를 명시하고, 논문 Eq. 6 구현으로 표시하지 않는다.

---

# 8. Algorithm 1의 실제 구현

## 8.1 전체 상태

```python
teacher = deepcopy(pretrained_model)
student = deepcopy(pretrained_model)

teacher.eval()
freeze(teacher)

keep_ids = [None] * L
keep_ids[L - 1] = tensor([CLS_ID])
```

- `teacher`: 원본 pretrained 모델, 항상 고정
- `student`: 이미 확정된 깊은 block weight와 현재 탐색 block weight를 보유
- `keep_ids`: layer별 고정 mask

## 8.2 Target layer 반복

```python
for t in range(L - 2, -1, -1):
    search_layer(t)
```

`search_layer(t)`의 입력:

```text
teacher
student
keep_ids[t+1 ... L-1]
calibration dataset
reconstruction dataset
epsilon
search_step
block_finetune_epochs
```

출력:

```text
keep_ids[t]
accepted reconstruction error
fine-tuned student.blocks[t] weight
search log
```

## 8.3 Base mask

```python
base_ids = keep_ids[t + 1].clone()
```

다음 layer에서 필요한 token은 현재 layer에서도 무조건 유지한다.

## 8.4 Candidate 순서

```python
mean_scores = estimate_mean_scores(target_layer=t)

candidate_ids = all_global_patch_ids excluding:
    - prefix/classifier token IDs
    - base_ids

candidate_ids = sort(candidate_ids, key=mean_scores, descending=True)
```

## 8.5 Candidate mask 확장

논문 Algorithm 1을 다음처럼 구현한다.

```python
r = 0
while True:
    current_ids = sorted_unique(base_ids + candidate_ids[:r])

    finetune_current_block(
        target_layer=t,
        current_keep_ids=current_ids,
        next_keep_ids=keep_ids[t + 1],
        epochs=block_finetune_epochs,
    )

    error = evaluate_next_layer_reconstruction(...)

    if error <= epsilon:
        keep_ids[t] = current_ids
        accept current student.blocks[t] weights
        break

    if r >= len(candidate_ids):
        keep_ids[t] = all_global_ids
        break

    r = min(r + search_step, len(candidate_ids))
```

## 8.6 후보별 block weight 정책

논문은 후보 mask가 커질 때 block weight를 초기화하는지 명시하지 않는다.

**[DECISION] 기본값은 cumulative fine-tuning이다.**

```text
r=0 후보에서 fine-tuned된 block weight
→ r=10 후보에서 이어서 fine-tuning
→ r=20 후보에서 이어서 fine-tuning
```

이 결정은 Algorithm 1의 while loop 안에서 연속적으로 “fine-tune current block”을 수행하는 흐름에 가장 가깝고 search 비용이 작다.

**[OPTION]** 디버그/비교를 위해 아래 mode를 추가할 수 있다.

```text
candidate_weight_policy = "reset_from_layer_start"
```

이 경우 target layer search를 시작할 때 저장한 checkpoint로 매 후보를 reset한다. 결과 metadata에 정책을 기록한다.

## 8.7 종료 조건

- error ≤ epsilon이면 현재 mask 확정
- 모든 candidate를 추가했는데도 error가 크면 all-token mask 사용
- 최대 iteration 보호 장치
- NaN/Inf 발생 시 즉시 중단하고 checkpoint 및 batch ID 기록

---

# 9. Block reconstruction fine-tuning

## 9.1 학습 대상

현재 탐색 중인 `student.blocks[t]`의 parameter만 학습한다.

```python
freeze(student)
unfreeze(student.blocks[t])
```

다음 block `student.blocks[t+1]`은 parameter는 고정하지만, gradient가 current block output으로 전달되어야 하므로 `torch.no_grad()`로 감싸지 않는다.

## 9.2 입력과 target

논문 설명에 따라 현재 block 입력은 원본 unpruned feature \(Z_{t-1}\)를 사용한다.

0-based 기준:

- `teacher_input`: teacher의 block `t` 입력
- `teacher_target`: teacher의 block `t+1` 출력
- `student_prediction`:
  1. `teacher_input`
  2. candidate `keep_ids[t]`를 적용한 student block `t`
  3. 확정된 `keep_ids[t+1]`를 적용한 frozen student block `t+1`

그 뒤 `keep_ids[t+1]`에 해당하는 output을 비교한다. Slim block `t+1`의 output 자체가 이미 해당 ID만 포함하므로 추가 gather가 필요 없을 수 있다.

### 예외: t = L-2

현재 block은 `L-2`, 다음 block은 마지막 block `L-1`이다. 마지막 block output은 CLS-only로 계산한다.

## 9.3 Reconstruction forward

```python
def reconstruct_next_layer(
    teacher_input_full,
    target_layer,
    current_keep_ids,
    next_keep_ids,
):
    all_ids = global_all_ids

    out_t = slim_block_t(
        teacher_input_full,
        active_global_ids=all_ids,
        output_global_ids=current_keep_ids,
    )

    out_next = slim_block_next(
        out_t.x,
        active_global_ids=current_keep_ids,
        output_global_ids=next_keep_ids,
    )

    return out_next.x
```

## 9.4 Reconstruction loss와 error metric

논문은 squared Frobenius norm을 쓰지만 실제 batch/dimension normalization과 \(\epsilon=0.01/0.02\)의 정확한 코드가 공개되지 않았다.

**[DECISION] 기본 metric은 element-wise MSE다.**

\[
E_{t+1}=\operatorname{mean}\left[(\hat Z_{t+1}-Z_{t+1})^2\right]
\]

```python
loss = F.mse_loss(student_prediction, teacher_target_selected)
```

동시에 다음 metric을 log한다.

```text
mse
raw_frobenius_sq
relative_frobenius_sq = ||pred-target||² / (||target||² + δ)
```

검색 종료 조건에는 config의 `error_metric`을 사용하며 기본은 `mse`다.

```yaml
error_metric: mse
epsilon: 0.02
```

논문 수치가 재현되지 않으면 metric을 몰래 변경하지 말고 실험 config로 비교한다.

## 9.5 Optimizer

논문은 block fine-tuning optimizer와 learning rate를 구체적으로 제시하지 않는다.

**[DECISION]** 기존 `npu pruning` 프로젝트의 fine-tuning optimizer factory를 우선 사용한다. 별도 기본값이 필요하면 다음으로 시작한다.

```yaml
block_finetune:
  optimizer: adamw
  learning_rate: 1.0e-5
  weight_decay: 0.0
  epochs: 3
  grad_clip_norm: 1.0
```

이 값은 논문 고정값이 아니며 반드시 config와 결과 metadata에 기록한다.

## 9.6 Train/eval dataset

**[DECISION]** 첫 구현에서는 하나의 deterministic calibration subset을 score 계산, block fine-tuning, error 평가에 사용한다. 이후 overfitting 검증을 위해 별도 `reconstruction_eval_subset`을 지원할 수 있다.

- subset sample ID를 파일로 저장한다.
- seed를 고정한다.
- random crop 대신 baseline validation preprocessing과 동일한 deterministic transform을 우선 사용한다.
- label은 reconstruction 단계에 필요하지 않다.

---

# 10. Teacher feature 획득과 cache

## 10.1 필요한 teacher 값

Target layer `t`마다 calibration sample에 대해:

```text
teacher block t input
teacher block t+1 output
```

이 필요하다.

## 10.2 권장 구현

첫 버전은 correctness를 위해 on-the-fly hook을 허용한다. 이후 layer 단위 cache를 추가한다.

```python
TeacherFeatureBatch(
    sample_ids,
    z_t_input,       # [B,N,D]
    z_t_plus_1_out,  # [B,N,D]
)
```

cache는 layer 검색이 끝나면 해제할 수 있다. 전체 12개 layer × 전체 subset feature를 한꺼번에 GPU에 유지하지 않는다.

## 10.3 Cache 무효화

Teacher는 고정이므로 다음이 같으면 재사용 가능하다.

```text
model name/checksum
input resolution
preprocessing config
sample IDs
layer index
```

Student attention은 깊은 block weight와 mask에 의존하므로 target layer score cache는 architecture/checkpoint checksum이 다르면 무효화한다.

---

# 11. 전체 top-down search 의사 코드

```python
def run_patch_slimming_search(cfg, pretrained_model, calibration_loader):
    teacher = deepcopy(pretrained_model).eval()
    teacher.requires_grad_(False)

    student = deepcopy(pretrained_model)

    L = len(student.blocks)
    N = get_num_global_tokens(student)
    all_ids = torch.arange(N)
    cls_ids = torch.tensor([get_cls_global_id(student)])

    keep_ids: list[Tensor | None] = [None] * L
    keep_ids[L - 1] = cls_ids

    records = []

    for t in range(L - 2, -1, -1):
        mean_scores = estimate_dataset_scores(
            student=student,
            target_layer=t,
            deeper_keep_ids=keep_ids[t + 1 :],
            loader=calibration_loader,
            score_mode=cfg.score_mode,
        )

        base_ids = keep_ids[t + 1].clone()
        candidate_ids = rank_candidate_patch_ids(
            mean_scores=mean_scores,
            excluded_ids=base_ids,
            prefix_ids=cls_ids,
        )

        r = 0
        accepted = False

        while not accepted:
            current_ids = sorted_unique(
                torch.cat([base_ids, candidate_ids[:r]])
            )

            finetune_one_block_for_candidate(
                teacher=teacher,
                student=student,
                target_layer=t,
                current_keep_ids=current_ids,
                next_keep_ids=keep_ids[t + 1],
                loader=calibration_loader,
                cfg=cfg.block_finetune,
            )

            metrics = evaluate_reconstruction(
                teacher=teacher,
                student=student,
                target_layer=t,
                current_keep_ids=current_ids,
                next_keep_ids=keep_ids[t + 1],
                loader=calibration_loader,
            )

            records.append(...)

            if metrics[cfg.error_metric] <= cfg.epsilon:
                keep_ids[t] = current_ids
                accepted = True
            elif r >= len(candidate_ids):
                keep_ids[t] = all_ids
                accepted = True
            else:
                r = min(r + cfg.search_step, len(candidate_ids))

        validate_nested_masks(keep_ids, upto=t)
        save_search_checkpoint(student, keep_ids, records, layer=t)

    architecture = build_architecture_spec(...)
    return student, architecture, records
```

---

# 12. Compact PS-ViT 모델 생성

## 12.1 Search 모델과 inference 모델을 분리

Search 중에는 debugging 편의를 위해 global mask와 full-token feature를 사용할 수 있다. 그러나 최종 inference에서는 zero-mask만 적용해서는 안 된다.

```text
잘못된 최종 구현:
[B,197,D]를 계속 계산하고 제거 token을 0으로만 만듦

올바른 최종 구현:
각 block에서 [B,N_in,D] → [B,N_out,D]로 실제 tensor 길이를 줄임
```

## 12.2 Block별 정적 shape

`keep_ids`가 정해지면:

```text
block 0: N_input = N_global,       N_output = len(keep_ids[0])
block 1: N_input = len(keep_ids[0]), N_output = len(keep_ids[1])
...
block L-1: N_input = len(keep_ids[L-2]), N_output = len(keep_ids[L-1])
```

각 block에 constant local gather index를 등록한다.

```python
register_buffer("output_local_ids", tensor([...]), persistent=True)
```

## 12.3 Compact model forward

```python
x = patch_embed(image)
x = add_cls_and_position_embedding(x)
active_ids = all_global_ids

for l, block in enumerate(compact_blocks):
    output_ids = architecture.keep_ids[l]
    x = block(
        x,
        active_global_ids=active_ids,
        output_global_ids=output_ids,
    ).x
    active_ids = output_ids

logits = classifier(x[:, local_index_of_cls])
```

CLS가 항상 첫 local position이 되도록 global IDs를 정렬하면 classifier는 `x[:,0]`을 계속 사용할 수 있다.

## 12.4 Position embedding

Patch Slimming은 position embedding을 더한 뒤 block 내부 token을 제거하므로 새로운 position embedding을 만들지 않는다. 원래 global position embedding이 포함된 token 중 선택된 row를 계속 전달한다.

---

# 13. 전체 fine-tuning

## 13.1 목적

mask search와 block reconstruction으로 얻은 compact architecture/weight를 실제 classification task에 다시 적응시킨다.

## 13.2 동작

- 모든 keep IDs를 고정한다.
- compact model의 전체 parameter를 학습 가능하게 둔다.
- baseline 모델의 기존 fine-tuning recipe를 재사용한다.
- 기본 objective는 classification cross-entropy다.
- 논문에서 별도 distillation objective를 핵심 알고리즘으로 요구하지 않으므로 자동 추가하지 않는다.

```python
logits = compact_model(images)
loss = cross_entropy(logits, labels)
```

## 13.3 Checkpoint 선택

- search가 끝난 student weight를 compact model로 변환한다.
- final fine-tuning 전/후 checkpoint를 모두 보관한다.
- architecture JSON과 checkpoint checksum을 연결한다.

---

# 14. NPU 배포 요구사항

## 14.1 우선 구현

논문에 충실한 최종 attention은 rectangular attention이다.

```text
Q length   = N_out
K/V length = N_in
attention  = N_out × N_in
```

NPU backend가 이를 지원하는지 먼저 확인한다.

## 14.2 최종 graph 원칙

- 입력 해상도 고정
- batch size별 별도 compile 허용
- block별 token shape 고정
- keep index는 constant
- runtime TopK 없음
- runtime similarity 계산 없음
- dynamic shape 없음
- bool mask보다 constant integer gather 사용
- Q와 KV projection을 분리

## 14.3 NPU가 rectangular attention을 지원하지 않을 때

아래 대안은 서로 결과와 연산량이 다르므로 mode를 분리한다.

### Mode A: full QKV 후 Q row 선택

```text
Q/K/V를 N_in 전체에서 계산
→ Q만 N_out row 선택
→ N_out × N_in attention
```

- 모델 출력은 논문 방식과 동일하게 유지 가능
- 제거된 query의 Q projection 연산은 절감되지 않음

### Mode B: block 입력을 먼저 N_out으로 gather

```text
N_in → N_out gather
→ Q/K/V 모두 N_out
→ N_out × N_out attention
```

- NPU 구현은 단순
- 현재 block의 K/V에서 제거 token 정보도 사라짐
- 논문 방식과 다름
- 별도 fine-tuning과 정확도 검증 필수
- 결과 파일에 `pre_gather_self_attention_variant`로 명시

기본 구현은 Mode A/B로 조용히 대체하지 않는다.

## 14.4 성능 측정

다음 값을 분리 측정한다.

```text
baseline end-to-end latency
compact PS-ViT end-to-end latency
block별 latency
constant Gather latency
Q/K/V projection latency
attention latency
MLP latency
peak activation memory
NPU utilization
```

FLOPs 감소만으로 가속을 주장하지 않는다.

---

# 15. 설정 파일 예시

```yaml
method: patch_slimming_static

model:
  name: deit_small_patch16_224
  input_size: [3, 224, 224]
  expected_num_blocks: 12
  expected_num_prefix_tokens: 1
  classifier_token_ids: [0]

calibration:
  num_samples: 2048
  batch_size: 16
  seed: 2022
  deterministic_preprocess: true
  sample_id_file: artifacts/calibration_ids.txt

score:
  mode: paper_path_energy_dp
  accumulator_dtype: float64
  model_eval_mode: true
  scatter_to_global: true

search:
  epsilon: 0.02
  error_metric: mse
  search_step: 10
  candidate_weight_policy: cumulative
  save_each_layer: true

block_finetune:
  epochs: 3
  optimizer: adamw
  learning_rate: 1.0e-5
  weight_decay: 0.0
  grad_clip_norm: 1.0

final_finetune:
  use_existing_project_recipe: true
  epochs: null

compact_inference:
  attention_mode: rectangular_q_kv
  constant_gather: true

export:
  backend: project_default
  static_batch_sizes: [1]
  dynamic_shape: false
```

`num_samples`, optimizer, learning rate는 논문에 명시되지 않은 공학적 설정이다. 프로젝트 상황에 맞게 변경할 수 있지만 모든 결과 metadata에 남긴다.

---

# 16. 필수 unit/integration test

## 16.1 Baseline instrumentation test

- 계측 on/off logits 동일
- 모든 block attention shape 확인
- attention row 합이 약 1인지 확인
- block input/output hook index 확인

## 16.2 Global/local ID test

- subset mapping 정확성
- 정렬되지 않은 ID 입력 처리 정책
- 없는 global ID 요청 시 오류
- duplicate ID 거부

## 16.3 Slim block equivalence test

- all-token output = original block output
- subset output = original block output의 동일 row gather
- eval mode에서 FP32 허용 오차 만족

## 16.4 Nested mask test

```python
for l in range(L - 1):
    assert set(keep_ids[l + 1]) <= set(keep_ids[l])
```

## 16.5 Score DP brute-force test

작은 N/H/L에서 explicit head-path enumeration과 DP score 비교.

## 16.6 Score sanity test

- shape `[B,N]`
- NaN/Inf 없음
- 음수 없음 또는 수치 오차 범위 내 clamp
- 같은 입력/seed에서 재현
- prefix token ranking 제외

## 16.7 Reconstruction gradient test

- current block parameter에 gradient 존재
- next block parameter gradient 없음
- next block 연산을 통과해 current block input/output까지 gradient 전달
- teacher parameter gradient 없음

## 16.8 Search state test

- r가 0, 10, 20 ... 순으로 증가
- candidate mask가 이전 mask의 superset
- accepted architecture serialize/deserialize 동일
- 중단 후 resume 동일

## 16.9 Compact model equivalence test

동일 searched weights와 keep IDs에 대해:

```text
reference slim/debug execution output
≈ compact model output
```

## 16.10 Export shape test

각 block의 실제 runtime shape가 architecture JSON과 동일한지 확인한다.

---

# 17. 실행 단계와 완료 조건

## Phase 1: Baseline adapter와 계측

완료 조건:

- 원본 pretrained logits 보존
- block별 attention/feature 추출
- unit test 통과

## Phase 2: Slim block

완료 조건:

- all-token/subset equivalence test 통과
- global/local ID 추적 통과

## Phase 3: 단일 layer score

먼저 마지막 바로 전 mask를 찾기 위한 target layer 하나만 구현한다.

완료 조건:

- DP brute-force test 통과
- calibration subset 평균 score 저장
- score 통계와 top patch IDs 출력

## Phase 4: 단일 layer candidate search

완료 조건:

- r=0,10,20... 후보 실행
- current block만 학습
- next-layer reconstruction metric 기록
- 하나의 mask 확정

## Phase 5: 전체 top-down search

완료 조건:

- 모든 layer keep IDs 생성
- nested invariant 통과
- layer별 checkpoint/로그 저장
- 피라미드형 token schedule 출력

피라미드형은 일반적으로 기대되지만, 임의로 강제하지 않는다. 오직 nested invariant만 강제한다.

## Phase 6: Compact 모델과 final fine-tuning

완료 조건:

- 실제 token shape 축소
- classification fine-tuning 실행 가능
- baseline 대비 accuracy/FLOPs/latency 보고

## Phase 7: NPU export

완료 조건:

- static graph
- operator 지원 확인
- block별 shape 검증
- end-to-end latency와 peak memory 측정

---

# 18. 필수 결과 보고서 항목

각 실험 결과는 최소한 다음을 포함한다.

```text
model/checkpoint
input resolution
dataset 및 sample IDs
score mode
error metric
ε
search step
block optimizer/LR/epochs
candidate weight policy
layer별 keep token 수
layer별 keep global IDs
layer별 accepted error
최종 Top-1/Top-5
이론 FLOPs
GPU/CPU latency
NPU latency
peak memory
export operator 목록
알려진 차이 및 실패
```

레이어별 token schedule 예시:

```text
input: 197
block 0: 197 → 187
block 1: 187 → 177
...
block 10: 31 → 21
block 11: 21 → 1
```

위 수치는 예시일 뿐 논문 결과로 고정하지 않는다.

---

# 19. 논문에서 불명확한 부분과 이 문서의 결정

| 항목 | 논문 상태 | 이 문서의 기본 결정 |
|---|---|---|
| calibration subset 크기 | 미기재 | config로 지정, sample IDs 저장 |
| block optimizer/LR | 미기재 | 프로젝트 optimizer 우선, 기본 AdamW 1e-5 |
| error normalization | 수식은 Frobenius², 코드 미기재 | MSE를 acceptance 기본값으로 사용하고 3개 metric 모두 log |
| 후보별 weight reset 여부 | 미기재 | cumulative fine-tuning |
| multi-layer head 경로 구현 | 코드 미기재 | path-energy DP + brute-force 검증 |
| search score 계산 precision | 미기재 | accumulator float64 |
| search transform | 미기재 | deterministic baseline preprocessing |
| compact Q/K/V 구현 | 개념 설명만 존재 | Q는 output subset, K/V는 input 전체 |
| NPU attention 미지원 대안 | 미기재 | 별도 variant로만 허용 |

이 표의 항목을 변경하면 architecture/result metadata에 변경 내용을 기록한다.

---

# 20. 구현 시 금지 사항

다음 동작은 Patch Slimming 구현으로 간주하지 않는다.

- class attention 값만으로 patch를 정렬하고 Eq. 6 구현이라고 표시
- inference마다 cosine similarity 또는 TopK를 계산
- 모든 layer에 동일 pruning ratio를 강제
- zero-mask만 적용한 뒤 FLOPs가 줄었다고 보고
- K/V도 현재 output subset으로 먼저 줄이면서 논문과 동일하다고 표시
- global patch ID를 버리고 local index만 저장
- 다음 layer mask가 현재 layer mask의 부분집합이 아닌 구조 허용
- 논문에 없는 optimizer/error metric을 사용하면서 기록하지 않음
- GPU 결과만으로 NPU 가속을 확정
- 공식 코드가 없는 세부사항을 논문 사실처럼 기술

---

# 21. Claude Code 작업 방식에 대한 최소 지침

기존 `npu pruning` 프로젝트 규칙을 그대로 따른다. 이 문서와 논문을 읽은 뒤 다음 순서로 작업한다.

1. 기존 ViT/DeiT 구현과 NPU export path를 먼저 조사한다.
2. 변경 파일과 새 abstraction을 간단히 계획한다.
3. Phase 1부터 순차 구현한다.
4. 각 Phase의 test를 통과하기 전 다음 Phase로 넘어가지 않는다.
5. 논문과 이 문서가 충돌하거나 기존 프로젝트 구조로 구현할 수 없는 지점은 코드로 우회하기 전에 보고한다.
6. 구현 완료 응답에는 변경 파일, 실행 명령, test 결과, 아직 미해결인 논문 ambiguity를 포함한다.

---

# 22. 최종 구현 완료 정의

다음이 모두 충족되어야 “Patch Slimming 구현 완료”로 본다.

1. 원본 ViT 계측이 logits를 변경하지 않는다.
2. Slim block이 original block의 선택된 output row와 수치적으로 동일하다.
3. 논문 score를 구현한 path-energy DP가 작은 brute-force case와 일치한다.
4. training subset 평균 score가 재현 가능하게 계산된다.
5. Algorithm 1의 top-down search가 layer별 고정 mask를 생성한다.
6. 모든 mask가 nested invariant를 만족한다.
7. 현재 block reconstruction fine-tuning과 error 평가가 분리되어 동작한다.
8. mask search가 layer checkpoint에서 resume 가능하다.
9. compact model이 실제로 token tensor 길이를 줄인다.
10. compact model을 classification task로 전체 fine-tuning할 수 있다.
11. architecture와 weight를 직렬화하고 동일 모델을 복원할 수 있다.
12. NPU graph가 고정 shape로 export된다.
13. baseline 대비 accuracy, FLOPs, 실제 latency, peak memory가 함께 보고된다.
14. 논문에 없는 구현 선택이 모두 config/result metadata에 기록된다.

---

# 23. 한 문장 요약

이 구현은 **사전학습 ViT의 마지막 CLS 출력에서 시작하여, 훈련 데이터에서 평균적으로 최종 출력에 영향이 큰 patch를 이전 layer마다 10개 단위로 추가하고, 다음 layer feature 복원 오차가 허용치 이하가 되는 최소 고정 patch 집합을 찾은 뒤, 각 block을 `Q: N_out`, `K/V: N_in`의 정적 compact 연산으로 변환하는 것**이다.
