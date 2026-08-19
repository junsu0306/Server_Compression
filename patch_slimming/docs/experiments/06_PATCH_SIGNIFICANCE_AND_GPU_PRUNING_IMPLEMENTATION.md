# Patch 중요도 계산과 GPU pruning 구현 상세

이 문서는 다음 질문에 코드 수준으로 답한다.

1. 각 patch의 중요도(significance score)는 무엇을 의미하는가?
2. attention 여러 layer와 여러 head의 영향을 어떻게 하나의 점수로 합치는가?
3. calibration dataset의 sample별 점수를 어떻게 model-level 고정 점수로 만드는가?
4. production 3·6·9 정책과 논문 Algorithm-1 search는 무엇이 다른가?
5. GPU에서는 중요도 순위로 선택된 patch를 실제 tensor에서 어떻게 제거하는가?
6. 선택 결과가 JSON, checkpoint, fine-tuning, ONNX export까지 어떻게 이어지는가?

관련 핵심 코드는 다음과 같다.

- `psvit/scoring.py`: path-energy significance score
- `psvit/search.py`: 논문식 top-down mask 및 token-count search
- `psvit/reconstruct.py`: candidate block reconstruction
- `scripts/stage_ablation.py`: production 3·6·9 고정-count mask 생성
- `psvit/ids.py`: global/local token ID와 nested-mask invariant
- `psvit/slim_block.py`: GPU row selection과 rectangular attention
- `psvit/compact.py`: 최종 compact forward

## 1. 전체 pipeline 요약

```text
structural-pruned ViT checkpoint
        |
        v
deterministic ImageNet calibration subset
        |
        v
target layer의 full attention + 이미 정한 downstream slim attention 측정
        |
        v
각 sample, 각 patch의 path-energy score 계산
        |
        v
calibration sample 전체 평균 score
        |
        v
score 내림차순으로 global patch ID 정렬
        |
        +-- production: 미리 정한 3/6/9 token 수만큼 선택
        |
        +-- paper search: candidate token 수를 늘리며 reconstruction MSE <= epsilon 탐색
        |
        v
nested keep_global_ids 확정
        |
        v
GPU CompactPSViT: index_select로 [B,N_in,D] -> [B,N_out,D]
        |
        v
architecture.json + weights + scores/records 저장
```

중요도 계산은 이미지마다 runtime에 수행하지 않는다. ImageNet calibration subset에서
미리 계산하고 평균낸 고정 mask를 모든 입력 이미지에 사용한다.

## 2. Token ID 규칙

224x224, patch-size 16의 표준 ViT는 다음 sequence를 사용한다.

```text
N = 197
global ID 0 = CLS
global ID 1..196 = 14x14 image patches
```

global patch ID `g >= 1`의 좌표는:

```python
patch_index = g - 1
row = patch_index // 14
col = patch_index % 14
```

global ID는 원본 197-token sequence에서의 영구적인 위치다. pruning 후 compact tensor
안에서의 위치는 local index이며 layer마다 달라진다.

```text
active global IDs = [0,3,7,10]
compact x index   = [0,1,2, 3]

global 7을 선택하려면 local index 2를 사용해야 한다.
```

`global_to_local()`은 정렬된 active ID에 `torch.searchsorted`를 적용하고 requested ID가
실제로 active set의 부분집합인지 검증한다.

## 3. Calibration data

중요도는 `psvit/data.py::build_calibration_loader()`가 만든 deterministic subset으로
계산한다.

현재 structural production 및 paper-style 실험 조건은:

```text
dataset split       ImageNet-1K train
number of samples   1,024
seed                2022
batch size          8
shuffle             false
augmentation        training random crop 아님; validation preprocessing
labels              중요도 계산에는 사용하지 않음
```

`torch.randperm`에 고정 seed를 사용해 sample index를 뽑고
`calibration_ids.txt`에 저장한다. 파일이 있으면 이후 실행에서 동일 ID를 다시 사용한다.
따라서 score, block reconstruction, error evaluation이 같은 image subset을 공유한다.

주의할 점은 이것이 class-balanced sampling은 아니라는 것이다. ImageNet train 전체에서
고정 seed로 무작위 1,024장을 선택한다.

## 4. 중요도의 직관

단순한 한 layer attention 평균만 사용하지 않는다. 어떤 patch가 현재 target layer에서
강한 feature energy를 만들고, 그 정보가 이후 slim layer들을 지나 최종 CLS 방향까지
얼마나 전달될 수 있는지를 곱한다.

```text
patch significance
  = target layer의 현재 feature energy
  x downstream network를 통한 영향력 energy
```

따라서 점수가 큰 patch는 두 조건을 함께 만족한다.

1. target layer에서 해당 token의 attention-weighted feature가 강하다.
2. 뒤쪽 layer들의 attention 경로를 따라 최종 출력에 영향을 전달할 가능성이 크다.

## 5. 기호와 tensor shape

target block을 `t`, 전체 token 수를 `N`, attention head 수를 `H`, embedding dimension을
`D`라고 한다.

```text
Z_{t-1}       target block 입력 feature           [B,N,D]
P_t           target block post-softmax attention [B,H,N,N]
P_k           downstream block attention          [B,H,N,N], global 좌표
U             target attention-weighted feature   [B,H,N,D]
score          sample별 token score               [B,N]
```

downstream slim attention은 실제로 `[B,H,N_out,N_in]`이지만 score DP에서는 원래 global
ID 위치에 scatter해 `[B,H,N,N]`으로 표현한다. 선택되지 않은 row/column은 0이다.

## 6. Hybrid forward: score 계산용 network

`scoring.py::hybrid_forward_for_target()`은 target `t`의 중요도를 계산할 때 다음 network를
구성한다.

```text
blocks 0 .. t-1  : full original forward
block t          : full input, full output, full attention 캡처
blocks t+1 .. 11 : 이미 확정된 downstream keep_ids로 slim forward
```

개념적인 코드는:

```python
x = embed_tokens(model, images)

for l in range(t):
    x = model.blocks[l](x)                 # full

z_prev = x                                # Z_{t-1}

out_t = SlimBlock(model.blocks[t])(
    x, all_ids, all_ids,
    return_attention=True,
)
target_attention = out_t.attention_probs  # full [B,H,N,N]

x = out_t.x
active = all_ids
for k in range(t + 1, L):
    out_k = SlimBlock(model.blocks[k])(
        x, active, keep_ids[k],
        return_attention=True,
    )
    downstream.append(scatter_to_global(out_k.attention_probs))
    x = out_k.x
    active = keep_ids[k]
```

뒤에서 앞으로 검색하므로 `t`의 score를 계산할 시점에는 `t+1..11`의 mask가 이미 정해져
있다. 이 downstream mask가 score에 반영된다.

## 7. Local slim attention을 global matrix로 변환

slim block의 attention shape은 `[B,H,N_out,N_in]`이다. 그러나 layer마다 local tensor
index가 다르므로 여러 layer의 attention을 곱하려면 공통 global ID 좌표가 필요하다.

`scatter_attention_to_global()`은 다음과 같이 변환한다.

```text
local output IDs = [0,2,4]
local input IDs  = [0,1,2,3,4]
local attention  = [3,5]

global attention = [5,5]
row 0,2,4에 local row 배치
row 1,3은 0
```

코드상 핵심은:

```python
Pg = zeros(B, H, N, N)
Pg[:, :, out_ids[:, None], in_ids[None, :]] = attn_local
```

이 0 row/column이 downstream mask의 효과를 DP에 반영한다.

## 8. Current feature energy 계산

target layer의 attention으로 입력 feature 절댓값을 집계한다.

```python
U = target_attention @ abs(z_prev)       # [B,H,N,D]
current_energy = U.square().sum(D).mean(H)  # [B,N]
```

수식으로는:

```text
U_t^h = P_t^h |Z_{t-1}|

current_energy_i
  = (1/H) sum_h sum_d U_t^h[i,d]^2
```

`abs(Z)`를 사용하는 이유는 feature의 양수·음수 상쇄를 피하고 magnitude를 집계하기
위해서다. embedding dimension에 대해 제곱합하고 head에 대해 평균낸다.

이 값만 사용하면 target layer 바로 뒤의 영향만 반영하며, 최종 classifier 방향으로
정보가 실제 전달되는지는 알 수 없다. 그래서 downstream energy를 추가한다.

## 9. Downstream path energy DP

여러 downstream layer와 여러 attention head를 모두 명시적으로 열거하면 downstream
layer가 `m`개일 때 head path가 `H^m`개가 된다. 이를 피하기 위해 동적 계획법을 쓴다.

초기값은 global token identity matrix다.

```python
G = I_N                              # [B,N,N]
```

가장 뒤쪽 downstream attention부터 역순으로 다음을 반복한다.

```python
for P in reversed(downstream):       # P [B,H,N,N]
    G = mean_h(P_h.T @ G @ P_h)      # [B,N,N]
```

마지막으로 대각 원소만 가져온다.

```python
downstream_energy = diagonal(G)      # [B,N]
```

`G[i,i]`는 token `i`에서 출발한 energy가 이미 정해진 downstream attention/head 경로를
통해 얼마나 전달되는지를 나타내는 값으로 사용한다.

이 head-path DP는 논문이 head path의 구체적인 구현을 제공하지 않은 부분에 대한 본
프로젝트의 구현 결정이다. 임의의 작은 attention에서 모든 `H^m` head path를 직접
열거하는 brute-force 결과와 DP가 일치하는지 `tests/test_score_dp.py`로 검증한다.

## 10. 최종 sample score와 dataset 평균

sample별 최종 점수는:

```python
sample_score = clamp_min(downstream_energy, 0) * current_energy
```

즉 token `i`에 대해:

```text
s_i = downstream_energy_i * current_energy_i
```

각 calibration batch에서 `[B,N]` 점수를 계산하고 sample 축으로 합산한다. 누적은
정밀도를 위해 CPU `float64`로 수행한다.

```python
score_sum += sample_scores.double().sum(dim=0).cpu()
mean_scores = score_sum / number_of_samples
```

NaN 또는 Inf가 하나라도 발생하면 즉시 실패한다. 결과 `mean_scores[g]`는 global token
ID `g`의 dataset-level 고정 중요도다.

## 11. 점수에서 patch ID 순위 만들기

후보 ranking에서는 다음 ID를 제외한다.

- CLS/prefix token
- downstream에서 이미 반드시 유지해야 하는 `base_ids`

나머지 global patch ID에 대해:

```python
order = argsort(mean_scores[candidate_ids], descending=True)
ranked_patch_ids = candidate_ids[order]
```

점수 순위를 만든 뒤 원래 global 공간 순서로 다시 정렬해서 mask를 저장한다. 즉 중요도
순위가 tensor 내부 순서가 되는 것은 아니다. 최종 `keep_ids`는 정렬된 global ID이며
원래 공간 순서를 보존한다.

## 12. Nested mask가 필요한 이유

모든 layer mask는 다음을 만족한다.

```text
CLS in keep[l]
keep[l+1] subset keep[l]
keep[l] is sorted and unique
```

한 layer에서 이미 제거된 token을 뒤쪽 layer가 다시 요청할 수 없기 때문이다.

그래서 target `t`의 후보는 항상 다음으로 만든다.

```python
current_ids = sorted_union(
    base_ids=keep_ids[t+1],
    extra_high_score_ids,
)
```

downstream 필수 token은 score가 낮더라도 무조건 포함되고, 남은 자리만 높은 score의
patch로 채운다.

## 13. Production 3·6·9 policy에서 ID를 선택하는 방법

production mild/medium/aggressive 실험은 논문식 free token-count search가 아니다.
token 수를 먼저 정책으로 지정한다.

```python
POLICIES = {
    "mild":             (177,157,137),
    "medium":           (157,118, 79),
    "aggressive":       (138, 89, 40),
    "extra_aggressive": (118, 69, 30),
}
```

경계 block은 0-based `(2,5,8)`, 즉 1-based layer 3·6·9다. 가장 깊은 경계부터 score를
계산한다.

medium 예시는:

```text
1. t=8 score 계산
   required = final CLS
   CLS + 높은 score patch를 합쳐 총 79개 선택
   keep[8], keep[9], keep[10] = 동일한 79 IDs

2. t=5 score 계산
   required = keep[8]의 79 IDs
   높은 score patch를 추가해 총 118개 선택
   keep[5], keep[6], keep[7] = 동일한 118 IDs

3. t=2 score 계산
   required = keep[5]의 118 IDs
   높은 score patch를 추가해 총 157개 선택
   keep[2], keep[3], keep[4] = 동일한 157 IDs
```

block 0,1은 전체 197개를 유지하고 block 11은 CLS만 출력한다. 최종 schedule은:

```text
[197,197,197,157,157,157,118,118,118,79,79,79,1]
```

각 structural checkpoint는 MLP width와 attention이 다르므로 네 production 모델에서
score와 patch ID를 각각 다시 계산했다. 한 모델의 ID를 다른 모델에 복사하지 않았다.

production stage ablation에서는 candidate별 block reconstruction을 수행하지 않는다.
지정된 count에 맞춰 significance ranking으로 ID를 바로 확정한 후 전체 ImageNet
validation과 1-epoch classification recovery로 정책을 비교했다.

## 14. 논문식 Algorithm-1에서 token 수까지 정하는 방법

04 실험의 paper-style search는 마지막 mask부터 시작한다.

```python
keep_ids[11] = [CLS]
for t in range(10, -1, -1):
    search_layer(t)
```

target `t`마다:

1. 현재 downstream mask를 반영한 mean significance score 계산
2. `base_ids = keep_ids[t+1]` 설정
3. base만 사용하는 `r=0` 후보부터 시작
4. reconstruction 학습 및 error 평가
5. MSE가 epsilon보다 크면 score 상위 patch 10개 추가
6. 처음 `MSE <= epsilon`인 mask를 채택
7. 끝까지 실패하면 all-token fallback

```python
r = 0
while True:
    current = union(base, ranked_candidates[:r])
    finetune_block_for_candidate(...)
    error = evaluate_reconstruction(...)
    if error <= epsilon:
        keep_ids[t] = current
        break
    r += search_step
```

본 실험의 `search_step=10`이므로 후보 token 수는 base size에서 10개씩 증가한다.

## 15. Candidate block reconstruction

중요도 순위만으로 candidate를 확정하지 않고, paper search에서는 다음 layer feature를
복원할 수 있는지 검사한다.

### 15.1 Teacher target

teacher는 search 시작 시 student를 deepcopy한 frozen structural 모델이다.

```text
teacher input
  -> full blocks 0..t+1
  -> block t+1 출력 중 next_keep_ids row만 선택
  -> reconstruction target [B,|next_keep|,D]
```

teacher가 동시에 제공하는 `Z_{t-1}` full feature를 student candidate의 입력으로 쓴다.

### 15.2 Student prediction

```text
teacher Z_{t-1} full
  -> student slim block t, current candidate IDs
  -> frozen student slim block t+1, next_keep_ids
  -> prediction [B,|next_keep|,D]
```

학습 가능한 parameter는 `student.blocks[t]`뿐이다. block `t+1` parameter는 고정하지만
gradient는 통과하므로 block `t`가 다음 block 출력 복원에 맞게 학습된다.

### 15.3 Loss와 acceptance

```python
loss = mse_loss(prediction, teacher_target)
```

후보별 기본 reconstruction 학습은 AdamW, LR `1e-5`, 3 epochs이며 MSE, raw Frobenius
제곱, relative Frobenius 제곱을 기록한다. acceptance 기본 metric은 element-wise MSE다.

현재 구현은 candidate가 커질 때 block weight를 원래 상태로 되돌리지 않는 cumulative
fine-tuning이다. 이것은 논문에 명확히 적히지 않은 부분에 대한 프로젝트 결정이다.

## 16. GPU에서 실제 pruning 선택이 일어나는 위치

GPU에서는 `SlimBlock.select_rows()`가 `torch.index_select`를 사용한다.

```python
def select_rows(x, out_local, mode):
    if mode == "index_select":
        return x.index_select(1, out_local)
```

예를 들어:

```text
x                 [B,197,D]
output global IDs 157개
out_local         [157]

x.index_select(1, out_local)
                  [B,157,D]
```

값을 0으로 만드는 mask가 아니라 새로운 작은 activation tensor를 만든다.

## 17. SlimBlock 내부의 정확한 GPU forward

입력:

```text
x                   [B,N_in,D]
active_global_ids   [N_in]
output_global_ids   [N_out], active의 부분집합
```

### 17.1 Global ID를 local tensor index로 변환

```python
out_local = global_to_local(active_global_ids, output_global_ids)
```

### 17.2 K/V는 현재 입력 전체에서 계산

```python
x_norm = norm1(x)                       # [B,N_in,D]
k = linear_k(x_norm)                    # [B,H,N_in,dh]
v = linear_v(x_norm)                    # [B,H,N_in,dh]
```

### 17.3 Q는 유지할 token만 선택한 뒤 계산

```python
x_norm_selected = x_norm.index_select(1, out_local)  # [B,N_out,D]
q = linear_q(x_norm_selected)                        # [B,H,N_out,dh]
```

Q projection부터 N_out만 처리하므로 실제 연산이 감소한다.

### 17.4 Rectangular attention

```python
attn = softmax(q @ k.transpose(-2,-1))  # [B,H,N_out,N_in]
ctx = attn @ v                          # [B,H,N_out,dh]
ctx = output_projection(ctx)            # [B,N_out,D]
```

### 17.5 Residual과 MLP

```python
x_selected = x.index_select(1, out_local)  # [B,N_out,D]
x_out = x_selected + ctx
x_out = x_out + mlp(norm2(x_out))          # N_out만 처리
```

출력 `[B,N_out,D]`가 다음 block의 입력이 된다. 제거된 token은 이후 activation에
존재하지 않는다.

## 18. CompactPSViT 전체 forward

`CompactPSViT`는 architecture의 block별 mask로 12개의 `SlimBlock`을 감싼다.

```python
x = embed_tokens(model, images)  # patch embedding + CLS + positional embedding

for l, slim_block in enumerate(self.slim):
    out = slim_block(x, active_ids[l], keep_ids[l])
    x = out.x

logits = classifier(norm(x)[:,0])
```

positional embedding은 pruning 전에 더해진다. 예를 들어 global patch 120이 compact
tensor local index 50으로 이동해도 global patch 120의 position 정보가 유지된다.

마지막 keep mask는 `[0]`이므로 최종 tensor는 CLS-only `[B,1,D]`이고 classifier가 이를
사용한다.

## 19. 저장되는 산출물

### Production stage policy

```text
architecture.json       block별 keep_global_ids
searched_student.pt     structural model weights와 mlp_dims
mean_scores.pt          boundary layer별 mean score [197]
metrics.json            full validation 결과와 analytical MAC
calibration_ids.txt     재현 가능한 calibration sample IDs
```

### Paper-style search

```text
architecture.json
searched_student.pt
search_records.jsonl    모든 layer/candidate IDs, loss, MSE, 시간, acceptance
search_layer_t.pt       layer 단위 resume checkpoint
search_last.pt
calibration_ids.txt
```

`architecture.json`은 mask의 source of truth이며 GPU compact 모델에서는 index-select ID,
NPU export에서는 상수 선택행렬의 source가 된다.

## 20. 검증 테스트

중요도와 선택 경로에는 다음 gate가 있다.

- DP downstream energy와 brute-force head-path 열거 일치
- global attention scatter가 올바른 row/column에 배치되는지 확인
- score shape `[B,N]`, finite, non-negative 확인
- nested mask의 CLS 포함, 정렬·유일, subset invariant
- ranking이 downstream required ID를 반드시 보존하는지 확인
- full keep일 때 SlimBlock과 원본 block 수치 일치
- subset일 때 SlimBlock 출력과 원본 block selected rows 일치
- GPU index-select와 NPU selection-matrix 경로 수치 일치
- compact checkpoint strict load와 runtime token schedule 일치

현재 Patch Slimming regression suite는 33개 test를 통과한다.

## 21. 구현상 중요한 해석과 한계

### 21.1 Dataset-level static importance

importance는 개별 입력에 최적화된 dynamic score가 아니다. 1,024장 평균에 잘 맞는
고정 공간 mask다. 물체 위치가 크게 달라지는 이미지에서는 dynamic pruning보다
정확도가 불리할 수 있지만 NPU static graph를 만들 수 있다는 장점이 있다.

### 21.2 Score는 classification loss가 아니다

path-energy score와 reconstruction MSE는 feature 전달과 복원을 측정하며 Top-1을 직접
최적화하지 않는다. 04 실험에서 epsilon 0.05가 local MSE를 통과했지만 Tiny 정확도가
붕괴한 것이 이 차이를 보여준다.

### 21.3 DP head aggregation은 프로젝트 구현 결정

논문이 여러 head path를 실제 코드로 어떻게 합치는지 제공하지 않아 평균 energy DP를
채택했다. brute-force와의 일치는 구현한 수학의 정확성을 보장하지만 이것이 저자 원본
비공개 코드와 완전히 동일하다는 의미는 아니다.

### 21.4 Production count는 논문 search 결과가 아니다

mild/medium/aggressive의 `(N3,N6,N9)` 값은 실험용 정책으로 정한 것이며 논문 공식이
자동 산출한 값이 아니다. 논문에서 가져온 부분은 path-energy 기반 patch ID ranking과
nested top-down 선택 개념이다.

### 21.5 GPU와 NPU의 선택 operator가 다르다

GPU는 efficient `index_select`, NPU ONNX는 Aries2 Gather 제약 때문에 상수 one-hot
MatMul을 쓴다. 결과 tensor와 ID는 동일하지만 NPU 경로에는 추가 selection 비용이
있다. 자세한 비용과 최적화는
`05_STATIC_TOKEN_SELECTION_AND_NPU_COST.md`에 정리되어 있다.

## 22. 한 문장 요약

현재 구현은 calibration 이미지에서 target-layer feature energy와 downstream attention
path energy를 곱해 global patch 중요도를 만들고, 뒤에서 앞으로 downstream 필수 ID를
보존하며 높은 점수의 patch를 선택한 다음, GPU에서는 `index_select`와 rectangular
attention으로 실제 activation 길이를 줄이는 static Patch Slimming이다.

