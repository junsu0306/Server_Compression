# Structural Pruning 이후 Patch Slimming 최종 구현 문서

이 문서는 이 저장소의 Patch Slimming 구현을 학술적 배경부터 실제 코드, 실험, 학습,
checkpoint, ONNX/NPU 변환까지 독립적으로 설명하는 단일 기준 문서다. 다른 문서를 먼저 읽지
않아도 전체 설계와 재현 방법을 파악할 수 있도록 작성했다.

---

## 1. 목적과 최종 적용 대상

목표는 ImageNet 학습이 끝난 ViT에 대해 데이터셋 수준에서 중요한 patch 위치를 한 번
탐색하고, 모든 입력에서 동일한 고정 patch ID를 사용해 token tensor를 물리적으로 줄이는
것이다. 이미지마다 token을 동적으로 선택하는 방식이 아니므로 최종 graph를 정적으로
컴파일할 수 있다.

Patch Slimming을 적용하기 전에 MLP channel structural pruning이 완료되어 있다. 최종 적용
대상은 다음 네 checkpoint다.

| 별칭 | Reduced checkpoint | ViT 계열 | Structural pruning |
|---|---|---|---:|
| Tiny-30 | `output/vit_tiny_30_final/reduced.pt` | ViT-Tiny | 약 30% |
| Tiny-50 | `output/vit_tiny_50_final/reduced.pt` | ViT-Tiny | 약 50% |
| Small-30 | `output/vit_small_30_final/reduced.pt` | ViT-Small | 약 30% |
| Small-50 | `output/vit_small_50_final/reduced.pt` | ViT-Small | 약 50% |

Structural pruning은 MLP hidden channel 수를 변경하고 Patch Slimming은 token 축을 줄인다.
두 방법은 서로 다른 tensor 차원을 압축한다.

```text
Structural pruning: [B, N, D] -> MLP 내부 [B, N, F_reduced]
Patch Slimming:     [B, N_in, D] -> block 출력 [B, N_out, D]
```

---

## 2. 원 논문의 문제 정의

원 논문은 다음 layer로 갈수록 attention mixing으로 patch 표현이 중복된다는 관찰에서
출발한다. 깊은 layer에서는 더 많은 patch를 제거하고 얕은 layer에서는 더 많은 patch를
유지하는 top-down 탐색을 사용한다.

Layer `l`의 binary mask를 `m_l`이라 할 때 논문의 전체 목적은 유지 token 수를 최소화하면서
최종 유효 출력의 reconstruction error를 제한하는 것이다.

```math
\min_{m_1,\ldots,m_{L-1}} \sum_{l=1}^{L-1}\|m_l\|_0
```

```math
E_L = \left\|\operatorname{diag}(m_L)(\hat Z_L-Z_L)\right\|_F^2 \le \epsilon
```

분류 ViT의 마지막 layer에서는 classifier가 CLS token만 사용하므로 마지막 mask는 CLS만
남긴다. 그 다음 `L-2`부터 `0`까지 역순으로 mask를 찾는다.

Nested mask 조건은 반드시 유지한다.

```math
\operatorname{supp}(m_{l+1}) \subseteq \operatorname{supp}(m_l)
```

즉 깊은 layer가 사용하는 patch 위치는 얕은 layer에도 존재해야 한다. 이 조건 덕분에 token을
제거한 뒤 다시 생성하거나 zero padding할 필요가 없다.

논문 Algorithm 1은 layer `l`의 mask 후보를 변경한 뒤 현재 block을 fine-tuning하고
`(l+1)`번째 layer의 reconstruction error를 계산한다. 이 구현도 동일하게 현재 block 출력이
아니라 다음 block까지 통과한 출력을 비교한다.

원 논문의 공개되지 않은 부분도 있다.

- Frobenius error의 batch 및 element 정규화 관례
- block fine-tuning에 사용하는 정확한 데이터 규모
- 모든 head path 합을 실제로 계산하는 구현
- candidate 반복 사이 weight reset 여부

따라서 이 저장소는 공개 수식과 재현 가능한 구현 결정을 구분해 기록한다.

---

## 3. 전체 파이프라인

```text
Reduced checkpoint
    |
    v
Frozen teacher + trainable student 생성
    |
    v
Score / reconstruction-train / acceptance-eval 데이터 분리
    |
    v
Top-down layer search (L-2 -> 0)
    |  - patch significance
    |  - candidate mask
    |  - 현재 block 3 epoch fine-tuning
    |  - 다음 block reconstruction error
    v
architecture.json + searched_student.pt
    |
    v
CompactPSViT(index_select): 실제 token 길이가 줄어든 forward
    |
    v
30-epoch classification fine-tuning
    |
    v
compact_finetuned_best.pt + architecture.json
    |
    v
CompactPSViT(selection MatMul, static B=1, NHWC)
    |
    v
ONNX opset 17 -> NPU compiler
```

---

## 4. 코드 구조와 각 파일의 책임

### 4.1 핵심 library

| 파일 | 책임 |
|---|---|
| `psvit/model_utils.py` | Reduced ViT 복원, MLP dimension 적용, embedding/head 보조 함수 |
| `psvit/ids.py` | Global patch ID, global-to-local 변환, nested-mask 검증 |
| `psvit/slim_block.py` | Rectangular attention과 실제 token 축소 |
| `psvit/scoring.py` | 논문 path-energy 중요도 근사와 head-path DP |
| `psvit/reconstruct.py` | Candidate block fine-tuning과 다음-layer error 측정 |
| `psvit/search.py` | Top-down Algorithm-1 search, progressive prior, checkpoint/resume |
| `psvit/architecture.py` | `architecture.json` schema와 검증 |
| `psvit/compact.py` | 검색된 mask로 실제 배포/학습 forward 구성 |
| `psvit/data.py` | 결정론적이고 상호 배타적인 search 데이터 subset |

### 4.2 실행 진입점

| 파일 | 책임 |
|---|---|
| `scripts/search.py` | 설정 로드, teacher/student와 데이터 구성, 검색, W&B, 결과 저장 |
| `scripts/evaluate.py` | Compact model의 전체 ImageNet validation |
| `scripts/finetune.py` | Mask 고정 compact model의 classification fine-tuning |
| `scripts/export_onnx.py` | 상수 선택행렬 기반 NPU-safe ONNX export와 검증 |
| `scripts/run_final_12way.py` | 4 models x 3 policies queue 생성, 실행과 재개 |

### 4.3 최종 실험 설정

`configs/final/final_12way_matrix.yaml`이 최종 12-way 실험의 단일 설정 원본이다.
Runner는 각 조합의 resolved YAML을 해당 output directory에 저장한다.

---

## 5. Global patch ID와 nested mask

224x224 입력과 patch size 16에서는 image patch가 14x14=196개다. CLS를 포함하면 전체
global token 수는 197이다.

```text
Global ID 0      : CLS
Global ID 1..196 : 고정된 14x14 spatial patch 위치
```

각 layer의 `keep_ids[l]`은 다음 조건을 만족한다.

1. 정렬되어 있다.
2. 중복이 없다.
3. CLS ID 0이 포함된다.
4. `keep_ids[l+1]`은 `keep_ids[l]`의 부분집합이다.

현재 tensor의 local row와 원본 spatial 위치를 혼동하지 않기 위해 architecture에는 항상
global ID를 저장한다. 실제 block forward 직전에 `global_to_local()`이 현재 active token에서
선택할 local row를 계산한다.

---

## 6. 실제 token 축소: SlimBlock

일반 self-attention은 입력과 출력 token 수가 모두 `N`이다. SlimBlock은 다음 layer에서
필요한 output token에 대해서만 query와 후속 MLP를 계산한다.

입력:

```text
x                 [B, N_in, D]
active_global_ids [N_in]
output_global_ids [N_out], N_out <= N_in
```

Q, K, V shape:

```text
Q: [B, H, N_out, d_h]
K: [B, H, N_in,  d_h]
V: [B, H, N_in,  d_h]
```

Rectangular attention:

```math
A=\operatorname{softmax}(QK^\top/\sqrt{d_h})
```

```text
A:       [B, H, N_out, N_in]
A @ V:   [B, H, N_out, d_h]
MLP:     [B, N_out, D]에서만 실행
output:  [B, N_out, D]
```

Residual path도 입력의 해당 row만 선택한다.

```text
x_selected = select(x, output IDs)
x_out = x_selected + Attention(Q_selected, K_all, V_all)
x_out = x_out + MLP(x_out)
```

따라서 제거된 token을 0으로 바꾸는 것이 아니다. 다음 block에 전달되는 tensor 자체가
`[B,N_out,D]`로 작아지며 attention activation도 `N_out x N_in`으로 감소한다.

QKV는 timm의 fused `qkv.weight [3D,D]`를 논리적으로 Wq/Wk/Wv view로 나눈다. 원본 weight를
복제하거나 새로운 projection parameter를 만들지 않는다.

구현: `psvit/slim_block.py::SlimBlock`.

---

## 7. Patch significance score

논문의 Theorem 1 score는 target layer `t`의 patch `i`가 최종 유효 출력에 미치는 영향을
근사한다.

```math
s_{t,i}=\sum_{h\in[H]^{L\sim t+1}}
\left\|A_t^h[:,i]U_t^h[i,:]\right\|_F^2
```

```math
A_t^h=\prod_{l=t+1}^{L}\operatorname{diag}(m_l)P_l^h,
\quad U_t^h=P_t^h|Z_{t-1}|
```

모든 head 조합을 직접 열거하면 downstream layer 수 `k`에 대해 `H^k`로 증가한다. 구현은
다음 dynamic programming을 사용한다.

```text
G = I
for downstream attention P in reverse order:
    G = mean_h(P_h^T @ G @ P_h)

downstream_energy_i = diag(G)_i
U = P_t @ abs(Z_{t-1})
current_energy_i = mean_h(sum_d(U[h,i,d]^2))
score_i = max(downstream_energy_i, 0) * current_energy_i
```

Head 합 대신 평균을 쓰면 같은 layer의 모든 token score에 layer-dependent 상수가 곱해진다.
Score는 layer 내부 ranking에만 사용되므로 mask 순위에는 영향을 주지 않는다. Cross-layer
score threshold에는 사용하지 않는다.

Score 계산 시 hybrid network는 다음과 같다.

```text
blocks < t : full token forward
block t    : full output, attention capture
blocks > t : 이미 확정된 deeper mask로 slim forward
```

Local rectangular attention은 global 197x197 matrix 위치에 scatter한 뒤 DP에 넣는다. 선택되지
않은 row와 column은 0이다. 데이터셋의 sample score는 float64로 누적해 평균한다.

DP 결과는 작은 tensor에서 `H^k` head path를 완전 열거한 결과와 unit test로 비교한다.

구현: `psvit/scoring.py`.

---

## 8. Top-down mask search

12-block ViT에서 마지막 block 출력은 CLS만 유지한다.

```text
keep_ids[11] = [0]
search order = block 10, 9, ..., 0
```

Layer `t`의 절차:

1. 이미 확정된 deeper masks로 path-energy score를 계산한다.
2. `base_ids = keep_ids[t+1]`로 둔다.
3. base와 CLS를 제외한 patch를 score 내림차순으로 정렬한다.
4. Candidate mask를 만든다.
5. Student block `t`만 3 epochs fine-tuning한다.
6. 독립 eval subset에서 block `t+1` 출력 reconstruction error를 계산한다.
7. Error가 epsilon 이하이면 mask를 확정한다.
8. 실패하면 score 상위 token을 search step 10개씩 더 추가한다.
9. 모든 token으로도 실패하면 full-token fallback을 사용한다.

Candidate fine-tuning은 cumulative 방식이다. 같은 layer에서 다음 candidate로 확장할 때 현재
block weight를 초기 checkpoint로 reset하지 않는다. 원 논문 while loop가 반복 fine-tuning을
명시하고 reset을 언급하지 않기 때문에 채택한 결정이다.

### 8.1 다음-layer reconstruction이 실제로 구현된 위치

Teacher:

```text
blocks 0..t-1 -> z_prev
block t -> block t+1 -> target rows at keep_ids[t+1]
```

Student:

```text
z_prev -> slim block t(current candidate)
       -> frozen slim block t+1(next mask)
       -> prediction
```

Block `t+1` parameter는 frozen이지만 gradient graph를 끊지 않으므로 loss gradient가 block
`t`까지 전달된다. Teacher 전체에는 gradient가 없다. 이 동작은
`test_teacher_target_is_next_block_output`과 gradient-routing test로 고정한다.

구현: `psvit/reconstruct.py`.

---

## 9. Reconstruction error 정의

초기 구현은 element-wise MSE를 사용했다.

```math
E_{MSE}=\frac{\sum(\hat Z-Z)^2}{BND}
```

논문의 epsilon 0.02를 이 MSE에 그대로 적용하자 Tiny-30에서 block 10이 41 tokens만 남고
대부분 layer가 같은 수를 유지했으며, MAC 81.40% 감소와 심각한 정확도 손실이 발생했다.
논문 수식은 unnormalized squared Frobenius지만 공개 구현의 batch convention이 없어 숫자를
직접 대응시킬 근거가 없다.

최종 search acceptance는 sample별 relative squared Frobenius 평균이다.

```math
E_{rel}=\frac{1}{B}\sum_b
\frac{\|\hat Z_b-Z_b\|_F^2}{\|Z_b\|_F^2+\delta}
```

최종 설정:

```yaml
epsilon: 0.02
error_metric: sample_relative_frobenius_sq_mean
```

모든 candidate에 대해 다음 지표를 함께 저장한다.

- element-wise MSE
- raw squared Frobenius
- global relative squared Frobenius
- per-sample relative squared Frobenius mean
- per-sample relative squared Frobenius p50/p95

Acceptance에는 mean per-sample relative 값만 사용한다.

---

## 10. Search 데이터 분리

동일한 1,024장을 candidate fine-tuning과 acceptance에 함께 사용하면 validation leakage가
생긴다. 최종 실험에서는 역할을 분리한다.

| 역할 | ImageNet train 표본 수 | Seed | Transform |
|---|---:|---:|---|
| Significance score | 2,048 | 2022 | deterministic validation transform |
| Block reconstruction train | 4,096 | 2023 | deterministic validation transform, shuffle |
| Acceptance eval | 2,048 | 2024 | deterministic validation transform |

세 집합은 index 교집합이 없음을 실행 시 검사하고 다음 파일로 저장한다.

```text
score_ids.txt
reconstruction_train_ids.txt
reconstruction_eval_ids.txt
```

기존 config에 새 section이 없으면 하나의 calibration loader를 공유하는 legacy 경로도
지원하지만 최종 실험에서는 사용하지 않는다.

---

## 11. Paper 대조군과 progressive policies

### 11.1 `paper`

최종 12-way 실험에서 `paper`는 progressive architecture prior를 넣지 않은 top-down
Algorithm-1 형태의 대조군이다.

```yaml
min_growth_per_shallower_layer: 0
```

단, 논문에 공개되지 않은 error normalization을 raw epsilon이라고 임의 가정하지 않는다.
세 정책을 공정하게 비교하기 위해 corrected relative metric과 분리 데이터를 공통 적용한다.
따라서 `paper`는 논문 수식의 완전한 bit-exact 재현이 아니라 `prior-free paper-style`이라는
뜻이다.

### 11.2 `progressive_g3`, `progressive_g5`

Relative error만 사용하면 가장 깊은 searchable block에서 큰 base mask가 결정되고 이전
layer들이 그대로 통과해 plateau가 생길 수 있다. 논문의 pyramid 경향과 NPU 친화적인 예측
가능한 축소를 얻기 위해 얕은 layer로 갈 때 최소 token 증가 prior를 추가했다.

```text
deepest searchable block: error search만 사용
shallower block t:
    base = keep[t+1]
    base에 현재 layer path-energy 상위 g개를 최소 추가
    error가 epsilon을 넘으면 기존대로 10개씩 추가
```

`g=3` 또는 `g=5`이며 마지막 CLS-only block은 예외다. 추가 위치는 임의가 아니라 각 layer의
path-energy ranking으로 결정된다. 이 방식은 다음 hybrid로 표기한다.

```text
paper path-energy score
+ next-layer reconstruction guard
+ progressive deployment prior
```

Tiny-30 pilot 결과:

| 정책 | Schedule 요약 | Structural baseline 대비 body MAC 감소 | Pre-FT Top-1 |
|---|---|---:|---:|
| prior 없음 | 131/121 plateau | 42.59% | 68.482% |
| progressive g=5 | 171 -> 121 | 30.04% | 70.220% |
| progressive g=3 | 151 -> 121 | 35.77% | 69.626% |
| manual medium | 3-stage 157/118/79 | 34.75% | 66.078% |

이 pilot에서는 g=3가 manual medium보다 MAC을 더 줄이면서 Top-1이 3.55%p 높았다. 숫자
3 또는 5는 모든 구조의 이론적 상수가 아니므로 최종 12-way에서 네 reduced model 모두
비교한다.

---

## 12. Search checkpoint와 산출물

각 layer가 끝날 때 다음 checkpoint를 저장한다.

```text
search_layer_{t}.pt
search_last.pt
```

포함 내용:

- 현재 student state dict
- 확정된 layer별 keep IDs
- accepted errors
- 모든 candidate record
- 다음에 검색할 layer 번호

검색 완료 산출물:

```text
architecture.json
searched_student.pt
search_records.jsonl
```

`architecture.json`은 block별 global keep IDs, output token 수, accepted error, epsilon,
metric과 score mode를 가진다. `searched_student.pt`는 search 과정에서 block reconstruction으로
갱신된 weight를 가진다.

`.pt`만으로 token 구조가 완결되지 않는다. PyTorch compact model을 재구성하려면 반드시
동일 폴더의 `architecture.json`이 필요하다.

---

## 13. Compact model과 최종 classification fine-tuning

검색이 끝나면 다음과 같이 compact model을 만든다.

```python
spec = ArchitectureSpec.from_json("architecture.json")
keep_ids = spec.keep_ids_list(device=device)
model = CompactPSViT(base, keep_ids, select_mode="index_select")
```

Fine-tuning 중에는:

- Layer별 keep ID와 token 수는 고정한다.
- 선택된 token만 실제 forward한다.
- Attention, MLP, normalization, patch embedding과 classifier weight를 학습한다.
- Cross-entropy와 label smoothing 0.1을 사용한다.
- RandAugment, color jitter, random erasing을 포함한 timm training transform을 사용한다.

최종 12-way recipe:

```text
epochs             30
optimizer          AdamW
learning rate      2e-5
weight decay       0.05
warmup             3 epochs
gradient clipping  1.0
Tiny batch         384
Small batch        192
```

Fine-tuning checkpoint:

```text
compact_finetuned_last.pt
compact_finetuned_best.pt
```

Checkpoint에는 base model state, architecture 경로, epoch, best Top-1과 함께 optimizer,
scheduler, AMP scaler state가 저장된다. `--resume`으로 다음 epoch부터 재개한다.

현재 기본 12-way 실험에는 distillation이 없다. 이는 세 architecture policy를 동일한 CE
조건에서 비교하기 위함이다. Distillation은 recovery가 부족한 최종 후보에 대해 teacher,
temperature, hard/soft loss weight를 고정한 별도 ablation으로 추가해야 한다. 현재
`scripts/finetune.py`에는 distillation loss가 구현되어 있지 않으므로 config flag만 추가해서
활성화할 수 있는 상태가 아니다.

---

## 14. `.pt + architecture.json`에서 ONNX로 변환

최종 ONNX에는 두 입력 산출물이 필요하다.

```text
compact_finetuned_best.pt : 학습된 weight
architecture.json         : layer별 고정 token 구조
```

Export 명령:

```bash
python patch_slimming/scripts/export_onnx.py \
  --arch output/.../architecture.json \
  --weights output/.../compact_finetuned_best.pt \
  --output output/.../ps_compact_npu.onnx \
  --opset 17 \
  --verify
```

최종 ONNX 특성:

- 입력 layout: NHWC `[1,224,224,3]`
- Static batch: 1
- Token selection: 상수 selection matrix MatMul
- Attention: rectangular `N_out x N_in`
- 마지막 출력: CLS-only tensor의 constant-axis Squeeze
- Constant folding 활성화

ONNX 입력은 모델의 normalization이 graph 밖에 있는 float image tensor다. 실제 calibration과
추론 파이프라인은 timm data config의 mean/std와 동일한 정규화를 적용해야 한다.

---

## 15. GPU 선택과 NPU 선택의 차이

### 15.1 학습/검증: index_select

GPU에서는 global ID를 local row로 변환한 뒤 `x.index_select(1, indices)`를 사용한다. 선택
연산은 미분 가능하고 구현이 간단하다.

### 15.2 ONNX/NPU: 상수 선택행렬

NPU graph에서 Gather 지원 문제가 있었기 때문에 다음 one-hot selection matrix를 사용한다.

```math
P\in\{0,1\}^{N_{out}\times N_{in}},
\quad P[i,\text{selected\_local\_id}_i]=1
```

```math
X_{selected}=PX
```

Architecture가 고정되어 있으므로 `P`는 입력이 아니라 ONNX initializer/constant다. JSON을
매 inference마다 읽거나 선택 ID를 동적으로 계산하지 않는다. Export 전에 모든 block의 P를
`bake_selection()`으로 만든다.

이 MatMul은 선택을 위한 추가 연산이다.

```math
N_{out}N_{in}D\;\text{MACs per selection}
```

현재 block에서는 Q 경로와 residual 경로에 같은 선택이 논리적으로 사용된다. Compiler가
상수 sparse selection을 최적화하는지에 따라 실제 비용이 달라진다. 따라서 최종 효율 표는
다음을 분리해 보고해야 한다.

1. Transformer body analytical MAC
2. Explicit selection MatMul 포함 MAC
3. 실제 NPU latency와 peak memory

Export script는 index-select 모델과 selection-matmul 모델의 PyTorch output 최대 오차를 먼저
확인하고, ONNX checker 및 선택적으로 ONNX Runtime output을 검증한다. Graph에 Gather,
GatherElements, ScatterElements, TopK, Equal이 남으면 경고한다.

ONNX가 생성된 뒤에는 patch JSON이 별도 runtime 입력으로 필요하지 않다.

---

## 16. Analytical MAC 계산

Block `l`의 입력/output token 수를 `N_in`, `N_out`, embedding width를 `D`, structural-pruned
MLP hidden width를 `F_l`이라 한다. 현재 body MAC 근사는 다음 항을 더한다.

```text
Q projection                  N_out * D * D
K/V projections             2*N_in  * D * D
Output projection             N_out * D * D
QK and attention-value       2*N_out * N_in * D
Two MLP projections          2*N_out * D * F_l
```

모델마다 실제 reduced `F_l`을 읽으므로 structural pruning과 token slimming을 동시에 반영한다.
Patch embedding, norm, classifier와 explicit selection MatMul은 body MAC 표에서 별도다.

구현: `scripts/stage_ablation.py::transformer_macs`.

---

## 17. 최종 12-way 실험

실험 행렬:

```text
4 reduced models x 3 policies = 12 searches + 12 final fine-tunes
```

정책:

```text
paper
progressive_g3
progressive_g5
```

GPU queue:

```text
GPU 6: Tiny-30과 Tiny-50의 6 jobs
GPU 4: Small-30과 Small-50의 6 jobs
```

W&B project:

```text
vit-patch-slimming-final-12way-20260806
```

Run naming:

```text
{model}__{policy}__rel002__search__{resolved_config_filename}
{model}__{policy}__rel002__finetune30
```

Resolved configs와 전체 manifest:

```text
output/ps_final_12way_20260806/experiment_manifest.json
output/ps_final_12way_20260806/{model}__{policy}__rel002/{run}.yaml
```

검색 실행:

```bash
python patch_slimming/scripts/run_final_12way.py \
  --matrix patch_slimming/configs/final/final_12way_matrix.yaml \
  --gpu 6 --phase search

python patch_slimming/scripts/run_final_12way.py \
  --matrix patch_slimming/configs/final/final_12way_matrix.yaml \
  --gpu 4 --phase search
```

Fine-tuning 실행:

```bash
python patch_slimming/scripts/run_final_12way.py \
  --matrix patch_slimming/configs/final/final_12way_matrix.yaml \
  --gpu 6 --phase finetune

python patch_slimming/scripts/run_final_12way.py \
  --matrix patch_slimming/configs/final/final_12way_matrix.yaml \
  --gpu 4 --phase finetune
```

Runner는 `architecture.json`이 있으면 완료된 search를 건너뛰고, final checkpoint의 epoch이
29이면 완료된 30-epoch job을 건너뛴다.

W&B에는 loss, accuracy, 기본 reconstruction 지표만 기록한다. Throughput, MAC, layer별
activation memory는 모든 학습이 끝난 뒤 12 compact models와 4 reduced baselines를 동일한
측정 harness에서 일괄 비교한다.

---

## 18. 검증 테스트

주요 test:

| Test | 보장 내용 |
|---|---|
| `test_ids.py` | Global/local ID와 nested-mask invariant |
| `test_slim_block.py` | Full-output 및 subset-output 수치 등가성 |
| `test_score_dp.py` | DP와 brute-force head path 계산 일치 |
| `test_reconstruct.py` | 다음 block target과 gradient routing |
| `test_search_state.py` | Candidate 순서, progressive prior, checkpoint round-trip |
| `test_compact.py` | 실제 token schedule과 searched/compact 수치 일치 |
| `test_structural_stage.py` | Reduced MLP 구조와 MAC 계산 |

실행:

```bash
python patch_slimming/tests/run_all.py
```

ONNX 후보는 추가로 다음을 통과해야 한다.

1. index-select와 selection-matmul PyTorch output 일치
2. ONNX checker 통과
3. PyTorch와 ONNX Runtime 최대 오차 1e-4 미만
4. 금지/위험 op 검사
5. 최종 NPU compiler와 실제 device 실행

---

## 19. 재현에 필요한 파일 보존 규칙

각 최종 후보 폴더에서 다음 파일을 함께 보존한다.

```text
resolved_config.yaml
architecture.json
searched_student.pt
search_records.jsonl
compact_finetuned_last.pt
compact_finetuned_best.pt
search.log
finetune.log
score_ids.txt
reconstruction_train_ids.txt
reconstruction_eval_ids.txt
```

최소 배포 재현 단위는 다음 두 파일이다.

```text
architecture.json + compact_finetuned_best.pt
```

실험 재현에는 resolved config와 세 subset ID 파일도 필요하다.

---

## 20. 알려진 한계와 해석 시 주의점

1. 논문의 epsilon 0.02와 현재 relative epsilon 0.02는 수학적으로 같은 normalization이라고
   증명된 값이 아니다. 현재 값은 논문 FLOPs 범위와 실제 accuracy-MAC Pareto를 기준으로
   보정한 구현 결정이다.
2. Progressive g3/g5는 원 논문 Algorithm 1에 없는 deployment prior다. 논문식 score와
   reconstruction guard는 유지하지만 별도 방법으로 명시해야 한다.
3. Importance와 reconstruction subset은 ImageNet 전체보다 작다. 최종 설정은 기존보다
   확대하고 train/eval을 분리했지만 full-dataset block reconstruction은 아니다.
4. Candidate fine-tuning은 cumulative이다. Candidate마다 reset하는 ablation은 기본 실험에
   포함되지 않는다.
5. Fine-tuning 기본 비교에는 distillation이 없다.
6. Analytical MAC 감소가 실제 GPU/NPU throughput 향상을 자동으로 의미하지 않는다.
   Rectangular attention kernel과 selection MatMul 최적화 여부를 실측해야 한다.
7. ONNX는 static batch 1, NHWC용이다. 다른 batch/layout이 필요하면 별도 graph를 export해야
   한다.
8. `compact_finetuned_best.pt`는 architecture를 내장한 TorchScript가 아니다. JSON 없이 동일
   compact graph를 복원할 수 없다.

---

## 21. 최종 권장 사용 흐름

```text
1. 12-way search/fine-tuning 완료 확인
2. Validation Top-1과 baseline gap 비교
3. Body MAC + selection MAC 계산
4. 동일 batch/static 조건에서 throughput과 layer activation memory 측정
5. accuracy-efficiency Pareto 후보 선정
6. 필요 후보에만 동일 조건 distillation ablation
7. best.pt + architecture.json으로 ONNX export
8. ONNX Runtime 검증
9. NPU compile/calibration
10. 실제 device latency, memory, accuracy 최종 보고
```

학술 보고에서는 `prior-free paper-style`, `progressive-g3`, `progressive-g5`를 구분하고,
structural-only baseline과 비교해 다음을 함께 제시한다.

- Layer별 token schedule
- ImageNet Top-1/Top-5
- Structural baseline 대비 accuracy gap
- Transformer body MAC과 selection 포함 MAC
- GPU/NPU throughput 및 latency
- Peak와 layer별 activation memory
- Fine-tuning 및 distillation 조건

이렇게 해야 structural pruning 이후 Patch Slimming이 제공하는 실제 정확도-연산량-메모리
trade-off를 재현 가능하게 설명할 수 있다.
# 메모리 및 activation 상세 프로파일링

ONNX 변환 전 최종 `CompactPSViT`의 실제 CUDA 메모리와 레이어별 activation을
`patch_slimming/scripts/profile_memory.py`로 측정한다. MH/COMPRESSION notebook의
`memory_allocated()` 전후 비교를 확장하여 allocator peak, 프로세스 메모리 및
functional attention 중간 tensor까지 함께 기록한다.

측정 항목은 다음과 같다.

- 모델 parameter, buffer, checkpoint 파일 크기
- CUDA 시작/종료 allocated·reserved 및 peak allocated·reserved
- 백그라운드 샘플링 기반 프로세스 RSS·USS·PSS peak
- 시스템 RAM used peak 및 available minimum
- leaf module forward hook의 실제 input/output shape, dtype, byte 수
- 블록별 입력·출력 토큰 수
- Q/K/V, attention score, attention probability, context, MLP hidden tensor 크기
- 블록별 attention matrix 크기와 activation tensor-volume 상한
- latency, images/s

Tiny30 paper 최종 모델 예시:

```bash
cd /workspace/etri_iitp/JS/Server_Compression

CUDA_VISIBLE_DEVICES=6 python patch_slimming/scripts/profile_memory.py \
  --arch output/ps_final_12way_20260806/tiny30__paper__rel002/architecture.json \
  --weights output/ps_final_12way_20260806/tiny30__paper__rel002/compact_finetuned_best.pt \
  --device cuda:0 \
  --batch-size 1 \
  --dtype float32 \
  --select-mode matmul \
  --warmups 10 \
  --runs 50 \
  --output output/memory_profiles/tiny30_paper_rel002_matmul_b1_fp32.json
```

실제 학습·평가 경로인 `index_select`도 별도로 비교한다.

```bash
CUDA_VISIBLE_DEVICES=6 python patch_slimming/scripts/profile_memory.py \
  --arch output/ps_final_12way_20260806/tiny30__paper__rel002/architecture.json \
  --weights output/ps_final_12way_20260806/tiny30__paper__rel002/compact_finetuned_best.pt \
  --device cuda:0 --batch-size 1 --select-mode index_select \
  --warmups 10 --runs 50 \
  --output output/memory_profiles/tiny30_paper_rel002_index_b1_fp32.json
```

원본 timm ViT와 비교할 수도 있다.

```bash
CUDA_VISIBLE_DEVICES=6 python patch_slimming/scripts/profile_memory.py \
  --model vit_tiny_patch16_224 --device cuda:0 --batch-size 1 \
  --warmups 10 --runs 50 \
  --output output/memory_profiles/vit_tiny_baseline_b1_fp32.json
```

Structural-pruned reduced checkpoint는 `--reduced`로 직접 측정한다.

```bash
CUDA_VISIBLE_DEVICES=6 python patch_slimming/scripts/profile_memory.py \
  --reduced output/vit_tiny_30_final/reduced.pt \
  --device cuda:0 --batch-size 1 --dtype float32 \
  --warmups 10 --runs 50 \
  --output output/memory_profiles/tiny30_reduced_b1_fp32.json
```

JSON과 함께 `_blocks.csv`가 생성된다. JSON의 `cuda.peak_allocated_bytes`는 실제
PyTorch allocator peak이고, `hook_activation.sum_module_output_bytes`는 forward 중
생성된 module 출력의 합계라서 동시 생존 메모리와 같지 않다.
`analytical_block_activation.sum_tensor_bytes_upper_bound` 역시 Q/K/V 등의 개별
tensor-volume 합계인 투명한 상한 지표다. 실제 peak 비교에는 CUDA peak를 사용하고,
어떤 tensor가 증가했는지 설명할 때 hook 및 block CSV를 사용한다.
