# Static token selection과 NPU 실행 비용 검토

이 문서는 production Patch Slimming 모델이 특정 patch token을 실제 forward에서 어떻게
제거하는지, `architecture.json`의 정적 mask가 NPU용 상수 선택행렬과 어떻게 연결되는지,
그리고 Gather 우회 때문에 추가되는 연산 비용을 정리한다.

중요도 점수가 계산되고 GPU에서 `keep_ids`가 결정되는 앞단의 전체 구현은
[`06_PATCH_SIGNIFICANCE_AND_GPU_PRUNING_IMPLEMENTATION.md`](06_PATCH_SIGNIFICANCE_AND_GPU_PRUNING_IMPLEMENTATION.md)를 참조한다.

## 1. 결론

- token은 0으로 masking한 채 유지되지 않는다. `[B,N_in,D]`에서 선택된 행만 남긴
  `[B,N_out,D]` tensor가 만들어지고 다음 block은 작은 tensor를 입력받는다.
- `architecture.json`에는 행렬 자체가 아니라 block별 `keep_global_ids`가 저장된다.
- GPU 학습·평가는 `index_select`, NPU ONNX export는 동일한 ID로 만든 one-hot 상수
  선택행렬 `P @ x`를 사용한다.
- Aries2에서 activation `Gather`는 지원되지 않았고, 상수 선택행렬 `MatMul`과
  rectangular attention은 toy graph의 `qbcompiler` 컴파일 성공을 확인했다.
- 선택행렬 방식에는 추가 MatMul 비용이 있다. 현재 analytical transformer MAC 표에는
  이 비용이 포함되어 있지 않으므로 실제 NPU 이득은 compiler/device에서 측정해야 한다.
- production 전체 모델의 `.mxq` 컴파일, device latency, peak memory는 아직 최종 검증
  단계로 남아 있다.

## 2. 정적 patch ID의 의미

표준 224x224, patch-size 16 ViT의 global token ID는 다음과 같다.

```text
global ID 0     = CLS
global ID 1..196 = 14x14 image patches
```

patch ID `g >= 1`의 공간 좌표는 다음과 같다.

```python
patch_index = g - 1
row = patch_index // 14
col = patch_index % 14
```

검색 결과는 각 모델의 `architecture.json`에 다음 형태로 저장된다.

```json
{
  "block_index": 8,
  "keep_global_ids": [0, 3, 7, 10],
  "num_output_tokens": 4
}
```

mask는 입력 이미지마다 달라지지 않는다. calibration dataset에서 미리 결정된 동일한
공간 patch 위치를 모든 이미지에 적용하는 static Patch Slimming이다. positional
embedding은 token 제거 전에 더해지므로 compact tensor에서 local index가 바뀌어도
원래 patch의 공간 위치 정보는 유지된다.

## 3. JSON에서 실제 token 선택까지

전체 변환 경로는 다음과 같다.

```text
architecture.json의 keep_global_ids
    -> ArchitectureSpec.keep_ids_list()
    -> block별 active_global_ids / output_global_ids
    -> global_to_local()
    -> GPU: index_select
       NPU: constant selection matrix P @ x
    -> [B,N_out,D]를 다음 block에 전달
```

현재 tensor에는 이전 block에서 살아남은 token만 있으므로 global ID를 tensor index로
바꾸어야 한다. 예를 들면:

```text
active global IDs    = [0, 3, 7, 10]
requested global IDs = [0, 7]
local indices        = [0, 2]
```

`psvit/ids.py::global_to_local()`이 정렬된 ID에 `searchsorted`를 적용하고 requested ID가
active ID의 부분집합인지 검증한다. nested-mask invariant는 CLS 포함, 정렬·유일성,
`keep[l+1] subset keep[l]`을 보장한다.

`CompactPSViT`는 block 0의 입력 ID를 전체 `[0..196]`, 이후 block의 입력 ID를 이전
block의 `keep_ids`로 등록한다. forward에서 각 `SlimBlock`의 작은 출력을 바로 다음
block 입력으로 대입한다.

## 4. Rectangular attention forward

`SlimBlock` 하나의 tensor contract는 다음과 같다.

```text
input x       [B,N_in,D]
K,V           [B,H,N_in,head_dim]
selected Q    [B,H,N_out,head_dim]
attention     [B,H,N_out,N_in]
block output  [B,N_out,D]
```

개념적인 구현은 다음과 같다.

```python
out_local = global_to_local(active_global_ids, output_global_ids)
x_norm = norm1(x)

k = project_k(x_norm)                     # N_in
v = project_v(x_norm)                     # N_in
x_norm_selected = select_rows(x_norm, out_local)
q = project_q(x_norm_selected)            # N_out

attn = softmax(q @ k.transpose(-2, -1))   # N_out x N_in
ctx = attn @ v                            # N_out

x_selected = select_rows(x, out_local)    # residual N_out
x_out = x_selected + project(ctx)
x_out = x_out + mlp(norm2(x_out))         # MLP도 N_out만 처리
```

제거되는 token도 해당 축소 block에서 마지막으로 K/V 정보를 제공할 수 있지만 block
출력에서는 사라진다. 이후 모든 block은 감소한 activation 길이를 사용한다.

production medium schedule의 실제 흐름은 다음과 같다.

```text
[197,197,197,157,157,157,118,118,118,79,79,79,1]

block 2:  197 -> 157
block 5:  157 -> 118
block 8:  118 -> 79
block 11:  79 -> 1 (CLS only)
```

## 5. GPU와 NPU 선택 경로

### 5.1 GPU 학습·평가

GPU에서는 실제 row gather를 수행한다.

```python
x_selected = x.index_select(1, out_local)  # [B,N_in,D] -> [B,N_out,D]
```

### 5.2 NPU ONNX export

Aries2에서 activation `Gather`가 지원되지 않아 NPU 경로는 같은 local ID로 one-hot
선택행렬을 만든다.

```text
out_local = [0,2]

P = [[1,0,0,0],
     [0,0,1,0]]       # [N_out,N_in]

P @ x = [x0,x2]       # [B,N_out,D]
```

`CompactPSViT.bake_for_export()`가 ONNX export 전에 block별 `P`를 미리 생성한다.
따라서 NPU가 runtime에 JSON을 읽거나 ID를 해석하지 않는다. JSON은 export-time
architecture source이고, ONNX에는 `P`가 constant/initializer와 `MatMul`로 들어간다.

```text
JSON = 어떤 global patch를 남길지 정의하는 설계도
P    = 설계도를 NPU가 실행할 수 있게 변환한 고정 연산
```

## 6. 추가 연산 비용

상수 선택행렬은 무료가 아니다. dense MatMul로 실행될 경우 한 번의 비용은 대략:

```text
B * N_out * N_in * D MACs
```

현재 구현은 Q 입력의 `P @ norm1(x)`와 residual 입력의 `P @ x`를 별도로 계산하므로
축소 block당 대략 다음 비용이 추가된다.

```text
2 * B * N_out * N_in * D MACs
```

예를 들어 Tiny의 197 -> 157, D=192, batch 1에서는:

```text
한 번: 157 * 197 * 192 = 약 5.94M MAC
두 번: 약 11.88M MAC
```

또한 현재 matmul export 경로는 `N_in == N_out`이고 ID가 완전히 같은 block에서도
identity `P`를 만들 수 있다. compiler가 이를 제거하지 않으면 불필요한 operator와
memory traffic이 생긴다.

기존 `transformer_macs()` 수치는 Q/K/V, attention, output projection, MLP만 포함하며
selection-matrix MatMul은 포함하지 않는다. 따라서 문서의 MAC 감소율은 transformer
본체 기준이고 NPU end-to-end 비용을 의미하지 않는다.

## 7. 필요한 최적화

### 7.1 Identity selection 제거

입력과 출력 ID가 같으면 선택 연산 없이 `x`를 그대로 사용해야 한다.

```python
if torch.equal(active_ids, output_ids):
    x_selected = x
else:
    x_selected = P @ x
```

medium schedule에서는 실제 감소 지점 네 곳만 선택 연산이 필요하다.

### 7.2 Q와 residual 선택 통합

LayerNorm은 token별 독립 연산이므로 수학적으로 다음이 성립한다.

```text
LayerNorm(P @ x) = P @ LayerNorm(x)
```

따라서 선택을 한 번만 수행하고 residual과 Q 양쪽에 재사용할 수 있다.

```python
x_norm_full = norm1(x)         # full K/V용
x_selected = P @ x             # 한 번만 선택
x_norm_selected = norm1(x_selected)

k, v = project_kv(x_norm_full)
q = project_q(x_norm_selected)
```

이 방식은 selection MatMul을 두 번에서 한 번으로 줄이는 대신 N_out token에 대한 작은
LayerNorm을 추가한다. 실제 ONNX 수치 동등성 테스트가 필요하다.

### 7.3 CLS-only 출력 특수 처리

마지막 79 -> 1 출력은 항상 첫 token인 CLS다. Aries2가 constant-axis `Slice`를
지원한다면 `P @ x` 대신 `x[:,0:1,:]`로 처리할 수 있다. compiler 지원 여부를 별도로
검증해야 한다.

### 7.4 Selection 비용을 포함한 분석

MAC 보고에 최소한 다음 두 값을 병기해야 한다.

```text
transformer-body MACs
transformer-body + explicit selection MACs
```

단, compiler가 identity MatMul 제거, one-hot MatMul 최적화 또는 operator fusion을 할 수
있으므로 최종 판단은 compiled graph와 device profiling을 기준으로 한다.

## 8. 현재 검증 범위와 남은 gate

| 항목 | 상태 |
|---|---|
| 실제 `[B,N_in,D] -> [B,N_out,D]` 축소 | 구현·테스트 완료 |
| index-select와 selection-matrix 수치 동등성 | 테스트 완료 |
| static NHWC batch-one ONNX | smoke 모델 완료 |
| PyTorch 대 ONNX Runtime | 최대 오차 약 `3.10e-6` |
| 금지 op 부재 | smoke ONNX 확인 |
| rectangular attention + `P @ x` Aries2 컴파일 | toy graph `.mxq` 생성 확인 |
| production medium 전체 ONNX export | 남음 |
| production medium 전체 `qbcompiler` 컴파일 | 남음 |
| 실제 NPU accuracy/latency/peak memory | 남음 |

NPU 배포 가능성을 확정하려면 네 medium checkpoint를 export한 후 ONNX checker/ORT,
금지 op 검사, `qbcompiler`, calibration, device inference를 순서대로 실행해야 한다.
특히 selection overhead를 포함한 baseline 대비 latency와 peak activation memory를 함께
보고해야 한다.

## 9. 관련 코드와 산출물

- `psvit/architecture.py`: JSON 직렬화와 block별 `keep_ids` 복원
- `psvit/ids.py`: global/local ID 변환과 nested mask 검증
- `psvit/compact.py`: block별 active/keep ID와 compact forward, export bake
- `psvit/slim_block.py`: row 선택과 rectangular attention
- `scripts/export_onnx.py`: static NHWC NPU-safe ONNX export 및 검증
- `scripts/toy_npu_validation.py`: Aries2 핵심 연산 사전 검증
- `artifacts/toy_rect_matmul_in196_out138_d192.onnx`: toy validation graph
- `output/ps_deit_tiny/ps_compact_npu.onnx`: 전체 smoke ONNX
