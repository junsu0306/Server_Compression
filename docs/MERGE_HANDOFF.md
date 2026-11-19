# 통합 저장소 구성 안내 (Server_Compression + npu)

작성일: 2026-11-19

이 문서는 `Server_Compression`과 `npu` 두 저장소를 **하나의 실행 가능한 저장소**로 합치는 작업을 위한 안내다. 통합 작업은 Mobilint 컴파일러가 설치된 x86 컨테이너(RTX A5000)에서 수행한다. 기존 두 저장소는 수정하지 않고 새 저장소를 만든다.

## 1. 목표

- 외부 사용자가 **처음부터** 실행할 수 있어야 한다: 압축 → Patch Slimming → ONNX → MXQ 컴파일 → NPU 벤치마크.
- 코드는 한 저장소, 실행 환경은 두 곳이다.

| 환경 | 수행 단계 | 비고 |
|---|---|---|
| x86 컨테이너 (CUDA GPU + Mobilint 컴파일러) | 메모리 예산 → pruning·KD 학습 → reducing → Patch Slimming → ONNX export → calibration → MXQ 컴파일 | 지금까지 두 서버로 나뉘어 있던 단계를 한 곳에서 수행 |
| NPU 장치 호스트 (aarch64 + Mobilint Aries) | MXQ 추론, 정확도·latency·메모리·전력 측정 | 학습용 패키지를 설치할 수 없는 환경 |

- 새 저장소에는 핵심 코드와 문서만 둔다. 과거 실험 보관물, 배포 감사용 스크립트, 개발 보조 도구의 설정 파일과 자동 생성 서명은 넣지 않는다. 커밋 이력은 새로 시작한다.

## 2. 가져올 원본

| 저장소 | 기준 | 주의 |
|---|---|---|
| `junsu0306/Server_Compression` | `main` 최신 | 최종 실험 `final_20261006`의 코드. `output/release_final_20261006/`에 최종 가중치·ONNX(Git LFS) |
| `junsu0306/npu` | **2026-10-20 제출 정리가 반영된 커밋** | 원격 `main`이 `3e8bb5a`(2026-10-15)이면 정리 전 구조다. 이 경우 `compile/mxq/compile_hwc.py`에 쓰이지 않는 수동 양자화 인자가 남아 있고 `docs/`, `scripts/verify_release.py`, `tests/`가 없다. 정리본이 push됐는지 먼저 확인한다 |

두 저장소는 서로 import하지 않는다. 연결 지점은 파일뿐이다: ONNX 12개와 `.verified.json` → MXQ 12개.

## 3. 환경 제약 (실측 확인)

### 3.1 Mobilint 컴파일러

`qbcompiler 1.1.0+aries2` wheel을 분석한 결과:

| 항목 | 값 |
|---|---|
| Python | 3.10 전용 (`cpython-310`), x86_64 전용 |
| glibc | **2.34 이상** 필요 (Ubuntu 22.04 이상). Ubuntu 18.04/20.04에서는 로드되지 않는다 |
| 네이티브 링크 | `libtorch.so`, `libtorch_cpu.so`, `libtorch_cuda.so`, `libc10_cuda.so` — 설치된 PyTorch의 CUDA 빌드에 직접 링크 |
| 의존 패키지 | `torch>=1.10.1`, `torchvision>=0.11.2`, `tensorflow>=2.9`, `transformers>=4.54.0`, `onnx>=1.16.2`, `opencv-python>=4.5.4.60`, `numpy>=1.23.0`, `pydantic>=2.10.6`, `pydantic_settings>=2.13.1`, `msgpack`, `tqdm`, `PyYAML`, `pycocotools`, `typeguard` |

지켜야 할 것:

- **컨테이너에 설치된 PyTorch를 바꾸지 않는다.** 컴파일러가 그 PyTorch에 링크되어 있어 버전을 바꾸면 컴파일러가 깨질 수 있다.
- 압축 코드용 패키지를 같은 환경에 넣을 때 `timm`은 `pip install --no-deps timm==1.0.25`로 설치한다(PyTorch가 따라 올라가지 않게).
- 충돌이 있으면 **가상환경을 둘로 나눈다**(압축·ONNX용 / 컴파일용). 코드는 하나, 환경은 둘이어도 된다. 이 경우 requirements 파일도 둘로 나눈다.
- 작업을 시작할 때 컨테이너의 `python --version`, `pip list`(torch, torchvision, numpy, onnx, onnxruntime, timm, qbcompiler), `nvidia-smi`, `ldd --version`, Docker 이미지 이름·digest를 기록해 저장소 문서에 남긴다. 기존 기록에는 이 정보가 없다.
- Mobilint SDK(wheel, 매뉴얼 PDF)는 저장소에 넣지 않는다. 설치 방법만 문서에 적는다.

### 3.2 압축 코드의 검증 환경

Python 3.10.18, PyTorch 2.6.0+cu118, torchvision 0.21.0+cu118, timm 1.0.25, NumPy 1.26.4, ONNX 1.21.0, ONNX Runtime 1.23.2, RTX 2080 Ti.

외부 패키지는 `torch`, `torchvision`, `timm`, `numpy`, `PyYAML`, `psutil`, `onnx`, `onnxruntime`, `wandb`뿐이다. 컨테이너의 PyTorch 버전이 다르면 다음을 확인한다.

- `torch.onnx.export`(opset 17) 결과에 금지 연산(Gather, GatherElements, GatherND, TopK, Equal, Where, Expand)이 생기지 않는지. `scripts/export_final_onnx.py`가 검사하고 실패 처리한다. PyTorch 버전에 따라 exporter 동작이 달라질 수 있다.
- timm ViT의 `fused_attn`, `_pos_embed`, `norm_pre`, `patch_drop`, `fc_norm`, `head_drop` 속성이 그대로 있는지. Export wrapper가 이 속성에 의존한다.
- `torch.amp.autocast("cuda")`, `torch.amp.GradScaler("cuda")` 호출 형태(PyTorch 2.4 미만에서는 다르다).

### 3.3 NPU 장치 호스트

기록된 환경: Linux 5.15.185-tegra aarch64, Python 3.10.12, NumPy 2.2.6, OpenCV 4.13.0, qbruntime v1.1.0, mobilint-cli v1.2.0, Aries driver 1.11.0, firmware 1.2.5.

- 벤치마크 코드는 `numpy`, `opencv`, `onnx`, `psutil`, `qbruntime`, `mblt_tracker`만 쓴다. **`torch`, `timm`을 import하지 않는 상태를 유지한다.** 통합하면서 공용 모듈을 만들더라도 장치 쪽 코드가 학습용 패키지를 끌어오지 않게 한다.
- NumPy 버전이 x86 쪽(1.26)과 다르다(2.2). 장치에서 쓰는 코드는 두 버전 모두에서 동작해야 한다.

## 4. 권장 저장소 구조

```text
<new-repo>/
├── README.md                     # 전체 흐름, 빠른 시작, 결과 요약
├── configs/final.yaml            # 실험 설정 (경로·GPU는 사용자 값으로)
├── requirements/
│   ├── compress.txt              # 압축·ONNX
│   ├── compile.txt               # calibration·MXQ (Mobilint SDK 별도 설치)
│   └── device.txt                # NPU 호스트
├── compression/                  # ← Server_Compression/pruning
├── patch_slimming/               # ← Server_Compression/patch_slimming (psvit, scripts)
├── export/                       # ← scripts/export_final_onnx.py
├── compile/                      # ← npu/compile (calibration, mxq)
├── benchmark/                    # ← npu/benchmark (final, memory, runtime, onnx)
├── profiling/, evaluation/       # ← Server_Compression/scripts/profiling, evaluation
├── scripts/run_pipeline.py       # 단계 실행기 (x86 쪽 전체)
├── tests/
└── docs/                         # 방법론, 실행 안내, 결과
```

폴더 이름은 바꿔도 되지만 **단계 순서가 폴더 구조에서 보이게** 한다.

가져오지 않는 것:

- `past/`, `.deps/`, `.cache/`, `.tools/`, `data/`, `output/final_20261006/`(로컬 원본)
- 배포 감사·패키징 전용: `scripts/audit_git_release.py`, `scripts/package_release.py`, `tests/test_release.py`, `docs/GIT_RELEASE.md`, npu의 `RELEASE_MANIFEST.json` 검사 중 Git index 검사 부분
- 사전 학습 가중치와 최종 모델을 저장소에 넣을지는 소유자와 정한다(8절).

## 5. 통합하면서 고쳐야 할 것

현재 코드는 특정 서버에 맞춰져 있다. 외부에서 처음 실행하려면 아래를 정리한다.

| 위치 | 현재 상태 | 조치 |
|---|---|---|
| `configs/final.yaml` `data_path` | 절대 경로 `/workspace/etri_iitp/JS/Server_Compression/data/imagenet` | 인자·환경 변수로 받거나 상대 경로 기본값 |
| `configs/final.yaml` `gpu`, batch | 모델별 GPU 0~3 고정, 2080 Ti(11GB) 기준 batch 후보 | GPU 수가 다른 환경에서도 돌도록 배정을 설정으로 분리. **기본 batch 값은 유지**(6절) |
| `scripts/run_pipeline.py:178`, `patch_slimming/scripts/search.py`, `finetune.py` | 저장소 안 `.deps` 폴더를 `PYTHONPATH`에 추가 | `.deps` 관련 코드 제거, 일반 설치 환경 사용 |
| `scripts/run_pipeline.py:174` | `TMPDIR`을 저장소 `.cache/tmp`로 설정 | 유지해도 되지만 `.gitignore` 처리 확인 |
| W&B | structural·patch 설정에서 `wandb: True` 고정, 프로젝트명 고정 | **기본값을 끄고** 선택 기능으로. 계정 없이 실행 가능해야 한다 |
| `sys.path` 조작 | `scripts/`와 `patch_slimming/scripts/`가 경로를 직접 삽입 | 패키지 구조(`pyproject.toml` 또는 `python -m`)로 정리하면 단순해진다 |
| 재개 보호 | 설정이 바뀌면 재개를 거부, 소스 해시 기록 | 동작은 유지하되 오류 메시지가 새 구조의 경로를 가리키게 |
| ONNX → MXQ 연결 | 사람이 파일을 다른 서버로 복사 | 실행기가 export 출력 폴더를 컴파일 입력으로 바로 넘기게 |
| npu 컴파일 설정 | 환경 변수 `NPU_CALIB_DIR`, `NPU_ONNX_DIR`, `NPU_MXQ_DIR`, `NPU_ONNX_FILTER`, `NPU_INFERENCE_SCHEME` | 같은 설정 파일이나 CLI 인자로 통일 |
| 파일명 규칙 | ONNX `<model>__<stage>__nhwc_b1_npusafe.onnx`, MXQ `<onnx_stem>_vt_global8.mxq` | 벤치마크 코드가 이 이름으로 모델을 찾는다. 바꾸려면 양쪽을 함께 바꾼다 |
| 문서 | 두 저장소에 방법론·실행·결과 문서가 따로 있음 | 하나로 합친다. `docs/END_TO_END_IMPLEMENTATION_AND_RESULTS.md`가 두 저장소를 이미 연결해 설명한다 |

## 6. 바꾸면 안 되는 것 (결과에 직접 영향)

아래는 방법 자체이므로 정리 과정에서 값이 달라지지 않게 한다. 바꾸면 기존 결과와 다른 실험이 된다.

**압축 (`configs/final.yaml`)**

- 모델: timm `vit_tiny_patch16_224`, `vit_small_patch16_224` (`augreg_in21k_ft_in1k`), 전체 파라미터 30%·50% 감소.
- Structural: 30 epoch 종료, **LR schedule 기준 길이 50 epoch**(둘은 다른 설정), terminal checkpoint 사용, AdamW lr 5e-5, weight decay 0.05, warmup 5, min lr 1e-6, grad clip 1.0, label smoothing 0.1, AMP, EMA 0.9998, KD α 0.5 / T 4.0, block 최대 sparsity 0.95, 최소 생존 채널 4, mask 순위 갱신 100 step, seed 2022.
- Batch: structural Tiny 384 / Small 192, patch fine-tuning Tiny30 512 / Tiny50 640 / Small 256. GPU 메모리가 커도 **키우지 않는다**(update 횟수가 달라져 결과가 바뀐다). Batch probe가 후보 중에서 고르므로, 기존 값을 고정하려면 후보를 한 개로 두거나 probe를 끈다.
- 메모리 예산: 목표 구조를 추가 학습 없이 만들어 추론 peak를 측정(FP32, batch 1, warmup 10, 50회). 측정값은 GPU·CUDA에 따라 달라지며, 그래도 방법은 같다.
- Reducing: EMA 가중치에 raw mask를 이식한 뒤 축소.
- Patch Slimming: ε 0.02, 채택 지표 `sample_relative_frobenius_sq_mean`, step 10, subset 2,048 / 4,096 / 2,048 (seed 2022 / 2023 / 2024, 서로 겹치지 않음), 후보별 3 epoch(AdamW 1e-5), 최종 30 epoch CE(lr 2e-5, warmup 3), best checkpoint.

**ONNX 계약**

- 입력 `input` float32 NHWC `[1,224,224,3]`, 출력 `output` `[1,1000]`, batch 1 고정, opset 17, 정규화는 그래프 밖(mean=std=0.5).
- Token 선택은 고정 0/1 행렬의 MatMul, identity 선택은 생략, CLS-only는 static Slice/Squeeze.
- PyTorch ↔ ONNX Runtime 최대 절대오차 1e-4 미만 검증을 통과해야 다음 단계로 넘어간다.

**Calibration·컴파일·런타임**

- 전처리는 calibration과 NPU 평가에서 **완전히 같아야 한다**: OpenCV `imread` → RGB → 짧은 변 249 → `INTER_CUBIC` → center crop 224 → `/255` → `(x−0.5)/0.5` → HWC float32. npu 저장소의 테스트가 두 구현의 일치를 검사한다. 이 테스트를 유지한다.
- Calibration 1,000장.
- 컴파일 인자는 아래 그대로. 다른 양자화 옵션을 넘기지 않는다.

```python
mxq_compile(model=onnx_path, calib_data_path=calib_txt, save_path=mxq_path,
            backend="onnx", inference_scheme="global8",
            cpu_offload=True, config_preset="vision_transformer")
```

- 런타임 core mode도 `global8`로 컴파일과 일치시킨다.
- NPU 평가: ImageNet validation class당 50장(seed 42), warmup 10, `model.infer()` 구간만 latency로 측정, NPU telemetry 500 ms.

## 7. 알려진 차이와 이번에 보완할 기록

- **서버 평가와 NPU 평가의 전처리 구현은 다르며, 그대로 둔다.** 서버 FP32 학습·평가는 timm/PIL(`crop_pct=0.9`), calibration과 NPU 평가는 OpenCV(짧은 변 249)다. NPU 호스트에는 학습용 패키지를 두지 않으므로 장치 쪽은 OpenCV 구현을 쓴다. 설정은 같은 뜻이지만 픽셀 단위로 같지 않다. 통일하지 않는 이유는 기존 서버 FP32 수치를 유지하기 위해서다. 문서에는 두 구현이 다르다는 점과, calibration·NPU 평가끼리는 반드시 같아야 한다는 점을 적는다.
- **컴파일 기록을 남긴다.** 기존 최종 MXQ는 컴파일 로그, calibration 이미지 목록·해시, 컴파일러 버전이 보존되지 않았다. 통합 실행기는 컴파일 stdout 로그, calibration 이미지 경로 목록과 SHA-256, `qbcompiler.__version__`, 입력 ONNX와 출력 MXQ의 SHA-256을 결과 폴더에 저장하게 한다.
- **Patch Slimming 모델은 NPU에서 정확도가 추가로 떨어진다**(서버 FP32 대비 7.9~18.1%p). MXQ 변환 과정의 양자화에서 발생한 것으로 판단하며, 이번 통합의 수정 대상이 아니다. 통합 후에도 이 현상이 재현되는 것이 정상이다.
- MXQ 컴파일은 byte 단위로 재현되지 않는다(같은 ONNX를 두 번 컴파일하면 파일 해시가 다르다). 해시 일치가 아니라 정확도로 비교한다.

## 8. 결정된 사항과 남은 확인

결정된 사항:

| 항목 | 결정 |
|---|---|
| 최종 모델 포함 | 최종 **ONNX 12개와 MXQ 12개**를 참고용으로 새 저장소에 포함한다(Git LFS, 합계 약 618 MiB). 각 ONNX의 `.verified.json`과 최종 결과 파일(서버 `reports/`, NPU `final_20261015/`)도 함께 둔다 |
| 포함 위치 | 새로 실행한 산출물과 섞이지 않게 별도 폴더(예: `reference/final_20261006/`)에 둔다. 실행기의 출력 폴더로 쓰지 않는다 |
| 전처리 | 서버와 NPU의 구현을 통일하지 않는다(7절) |
| 장소 인식 실험 | 포함하지 않는다. `ETRI_Evaluation_Metric`의 코드·문서·결과를 가져오지 않는다 |

참고 모델을 넣을 때 지킬 것:

- `.onnx`, `.mxq`는 Git LFS로 추적한다(`.gitattributes`).
- 원본과 SHA-256이 같은지 확인한다. 서버 `output/release_final_20261006/MANIFEST.json`과 npu `assets/RELEASE_MANIFEST.json`에 해시가 있다.
- ONNX와 MXQ의 대응(같은 이름 stem)과, 그 모델로 측정한 결과 파일을 README에서 연결해 준다.
- 참고 MXQ는 9절 4번 검증에서 새로 컴파일한 MXQ의 비교 대상으로 쓴다.

남은 확인:

1. npu 저장소의 2026-10-20 정리본이 원격에 push됐는가(2절).
2. PyTorch checkpoint(`.pt`, 약 1.58 GiB)도 포함할지. 현재 결정은 ONNX와 MXQ만이다. `.pt`가 없으면 9절 2번(기존 가중치로 ONNX 재export)은 원본 저장소의 가중치를 받아서 수행한다.
3. 새 저장소의 이름과 공개 범위, 라이선스 표기(timm·torchvision 등 사용 패키지의 라이선스 고지 포함).

## 9. 검증 절차

통합이 끝났다고 판단하기 전에 아래를 순서대로 확인한다.

1. **단위 테스트**: 두 저장소의 테스트를 옮겨 모두 통과시킨다(Server_Compression `tests/`와 `patch_slimming/tests/`, npu `tests/`). 배포 감사용 테스트는 제외.
2. **기존 모델로 연결 확인**: `output/release_final_20261006/`의 가중치로 ONNX를 다시 export해 금지 연산 검사와 parity를 통과하는지 본다. PyTorch 버전이 다르면 ONNX 해시는 달라질 수 있다.
3. **짧은 전체 실행**: 한 모델(tiny30)로 epoch과 데이터 수를 줄여 예산 → 학습 → reducing → 탐색 → fine-tuning → ONNX → calibration → MXQ 컴파일까지 한 번에 돈다. 정확도는 보지 않고 단계 연결만 확인한다. 이를 위한 축소 설정 파일을 저장소에 함께 둔다.
4. **기존 MXQ와 새 MXQ 비교**: 2번의 ONNX를 컴파일해 NPU에서 일부 이미지(예: class당 5장)로 평가하고 아래 기준값과 비교한다.
5. **전체 재실행**(선택): 4개 모델 전체 학습은 GPU 4장으로 며칠이 걸린다.

기준값(최종 결과, ImageNet validation 50,000장):

| 실험 | 서버 FP32 Top-1 (baseline / reduced / patch) | NPU Top-1 (baseline / reduced / patch) | NPU latency ms (baseline / reduced / patch) |
|---|---|---|---|
| Tiny30 | 74.660 / 72.212 / 67.902 | 74.226 / 70.880 / 53.528 | 3.192 / 3.065 / 2.408 |
| Tiny50 | 74.660 / 63.230 / 59.252 | 74.226 / 62.442 / 41.164 | 3.170 / 2.976 / 2.388 |
| Small30 | 80.502 / 79.624 / 77.970 | 80.486 / 79.324 / 70.094 | 5.884 / 5.569 / 4.585 |
| Small50 | 80.502 / 74.002 / 70.736 | 80.486 / 73.616 / 61.082 | 5.821 / 5.310 / 4.204 |

다시 학습했을 때 예상되는 차이:

| 항목 | 예상 |
|---|---|
| Baseline 정확도 | 동일(같은 pretrained 가중치) |
| FFN 채널 구성(block별 폭) | 거의 동일. Pretrained 가중치의 L2 norm으로 사실상 정해진다 |
| Reduced 정확도 | 소수점 첫째 자리 수준의 차이 |
| Patch Slimming token 구조 | 달라질 수 있다. 오차 0.02 경계에서 10개 단위로 결정된다 |
| Patch Slimming 정확도 | 1%p 안팎 차이 가능 |
| 서버 메모리 수치(M_max 등) | GPU·CUDA에 따라 달라진다 |

위 범위를 크게 벗어나면(예: reduced 정확도가 2%p 이상 다름) 설정이 달라졌을 가능성이 높다. 6절의 값부터 대조한다.

## 10. 참고 문서

| 내용 | 위치 |
|---|---|
| 전체 방법·설정·결과 | `docs/END_TO_END_IMPLEMENTATION_AND_RESULTS.md` |
| 압축 방법론 | `docs/METHODOLOGY.md` |
| 서버 실행 안내 | `docs/RUNBOOK.md` |
| 최종 결과표 | `output/release_final_20261006/reports/FINAL_RESULTS.md` |
| NPU 방법·계약·실행 | npu `docs/METHODOLOGY.md`, `docs/ONNX_CONTRACT.md`, `docs/RUNBOOK.md` |
| NPU 최종 결과 | npu `results/final_benchmark/final_20261015/summary.md` |
