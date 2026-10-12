# Server Compression

ViT-Tiny / ViT-Small에 **목표 구조 기반 메모리 예산 → global L2 FFN pruning + logit KD → reducing → Statistical Patch Slimming → ONNX**를 적용하는 서버용 프로젝트입니다.

최종 실험 `final_20261006`은 **2026-10-08 UTC에 서버의 학습·변환·평가를 모두 완료**했습니다. 이 저장소에는 최종 방법론 코드와 선별한 배포 산출물을 포함합니다. 과거 실험, 중간 체크포인트, 학습 로그, ImageNet 데이터는 포함하지 않습니다.

- [최종 실험 전체 결과](output/release_final_20261006/reports/FINAL_RESULTS.md)
- [최종 가중치·ONNX 패키지](output/release_final_20261006/README.md)
- [구현 방법론](docs/METHODOLOGY.md) · [실행 안내](docs/RUNBOOK.md) · [결과 해석](docs/RESULTS.md) · [Git/LFS 제출 안내](docs/GIT_RELEASE.md)

**역할 구분:** 이 프로젝트는 서버에서 압축·학습·FP32 평가·메모리 프로파일링·ONNX 변환까지 수행합니다. **MXQ 양자화/컴파일, 실제 Mobilint NPU 정확도·속도·메모리 측정은 별도 [`npu` 프로젝트](https://github.com/junsu0306/npu)에서 수행**합니다. 이번 ONNX의 NPU 결과는 아직 이 저장소의 결과에 포함되지 않습니다. `Orin_Compression`이 후속 작업 경로가 아닙니다.

## 실험 과정

```text
ImageNet pretrained ViT-Tiny / ViT-Small
  → baseline FP32 메모리 프로파일링
  → 사본을 목표 30%/50%로 pruning·reducing (추가 학습 0 step)
  → 사본의 CUDA peak를 고정 예산 M_max로 설정
  → baseline에서 global L2 FFN pruning + KD + 메모리 항, 30epoch
  → 정확히 30epoch 가중치를 물리적 reducing
  → 정적 패치 검색: 중요도 통계 + 후보별 block reconstruction 3epoch
  → 패치가 실제로 줄어든 compact 모델 CE fine-tuning 30epoch
  → baseline / reduced / patch-slimming ONNX 12개 생성·수치 검증
  → ImageNet FP32 정확도·서버 메모리 측정·보고서
```

### 1. FFN 구조적 압축과 메모리 항

- **L2 점수:** FFN 첫 Linear의 행 `||W_fc1[j, :]||₂`를 전체 block에서 비교하여 낮은 hidden channel부터 제거합니다. FC1의 행/bias와 FC2의 대응 열을 함께 자릅니다. Attention head/embedding 차원은 유지합니다.
- **목표 비율:** FFN 채널 수가 아닌 **전체 모델 파라미터 수 대비 30%·50% 감소**입니다. `pruned`는 zero mask 상태라 dense shape가 남고, `reduced`에서 채널을 물리적으로 제거합니다.
- **KD:** 같은 아키텍처의 pretrained teacher를 이용한 logit KL distillation입니다. α=0.5, T=4이며 관계형 KD가 아닙니다.
- **M_max:** 학습 완료된 reduced 모델을 요구하지 않습니다. Baseline의 사본을 목표 비율로 압축한 **추가 학습 없는 target 구조**의 FP32 추론 peak를 사용합니다.
- **메모리 항:** 매 epoch 후보/EMA 사본을 reducing하여 별도 프로세스에서 측정하고 `L_mem = λ max(0, M_candidate − M_max)`를 다음 batch loss에 반영합니다. λ=1, 단위 MiB입니다. 이 항은 **측정 기반 상수로 gradient나 자동 추가 pruning 기능은 없습니다.** 이번 네 실험에서는 후보가 예산을 초과하지 않아 패널티는 모두 0이었습니다.
- **학습/선택:** AdamW, LR=5e-5, warmup 5, EMA=0.9998. 종료 30epoch과 LR horizon 50epoch을 별도 설정했습니다. Structural은 best가 아니라 **정확히 30epoch terminal checkpoint**를 사용합니다.

### 2. Statistical Patch Slimming

ImageNet train의 서로 겹치지 않는 **중요도 통계 2,048장 / 블록 복원 학습 4,096장 / 후보 판정 2,048장**을 사용합니다. 남길 패치 ID를 `architecture.json`에 저장하므로 입력마다 TopK를 계산하는 동적 방식이 아닙니다.

논문 기반 탐색에 프로젝트의 **다음 block 출력, 샘플별 relative squared Frobenius 평균 ε=0.02**를 사용합니다. 후보별 블록 복원은 3epoch이며, 선택한 고정 ID로 토큰 텐서 크기를 실제 줄인 모델을 CE로 30epoch fine-tuning합니다. 이 단계에는 추가 KD가 없습니다. 최종 배포는 30epoch 학습 중 validation Top-1이 가장 높은 checkpoint입니다(Tiny30 28epoch, Tiny50 27epoch, Small30/50 29epoch).

ONNX에서는 고정 0/1 선택행렬의 MatMul로 토큰 수를 축소합니다. Identity 선택은 생략하고 마지막 CLS는 static Slice/Squeeze로 처리합니다. **FP32, batch 1, NHWC `[1,224,224,3]`, opset 17**이며 Gather/TopK 등 제한 연산의 부재와 PyTorch–ORT 출력을 검증합니다. `npusafe`라는 파일명은 실제 NPU 컴파일/양자화 정확도를 보장하는 표현이 아닙니다.

## 최종 정확도

ImageNet validation **50,000장, 1,000 class, PyTorch FP32/AMP off**로 직접 재평가한 결과입니다. 공식 성능표 인용이나 MXQ/NPU 정확도가 아닙니다.

| 원본 모델 | Parameters | FP32 weight MiB | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| ViT-Tiny baseline | 5,717,416 | 21.8102 | 74.660% | 92.494% |
| ViT-Small baseline | 22,050,664 | 84.1166 | 80.502% | 95.862% |

| 실험 | Reduced parameters | FP32 weight MiB | Pruned / Reduced Top-1 | Pruned / Reduced Top-5 | Patch Slimming Top-1 | Patch Slimming Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Tiny30 | 4,003,396 | 15.2717 | 72.212% | 91.168% | 67.902% | 88.130% |
| Tiny50 | 2,862,256 | 10.9186 | 63.230% | 85.832% | 59.252% | 82.270% |
| Small30 | 15,434,188 | 58.8768 | 79.624% | 95.162% | 77.970% | 94.190% |
| Small50 | 11,023,204 | 42.0502 | 74.002% | 92.234% | 70.736% | 90.124% |

Pruned와 reduced의 정확도는 같지만 **pruned의 dense 파라미터 수/저장량은 baseline과 같습니다.** 위 파라미터·weight 열은 reduced 기준이며 Patch Slimming도 동일합니다. Patch Slimming 이후 reduced 대비 Top-1이 각각 **4.310 / 3.978 / 1.654 / 3.266%p 감소**했습니다. 학습 완료가 정확도 회복을 보장하지는 않습니다.

## 서버 메모리 결과

RTX 2080 Ti, FP32, batch 1, 224×224 random input, warmup 10회 후 forward 50회입니다. Peak는 `torch.cuda.max_memory_allocated()`의 **최대값**이고 평균이 아닙니다. 모듈 출력 합은 별도 forward의 leaf-module 출력 텐서 용량 합으로 live activation peak와 다릅니다.

| 실험 | Baseline peak MiB | Reduced peak MiB | Patch peak MiB | Reduced 모듈 출력 합 MiB | Patch 모듈 출력 합 MiB |
|---|---:|---:|---:|---:|---:|
| Tiny30 | 32.8311 | 26.1523 | 27.3125 | 42.8932 | 19.3636 |
| Tiny50 | 32.8311 | 21.8018 | 22.8862 | 33.9834 | 14.5954 |
| Small30 | 97.1577 | 71.5552 | 74.0703 | 86.6843 | 46.8540 |
| Small50 | 97.1577 | 54.3721 | 56.6802 | 69.4420 | 32.6188 |

**Patch Slimming은 모듈 출력 합은 감소했지만, 이번 서버 구현의 CUDA peak와 latency는 reduced보다 증가했습니다.** 상수 선택행렬/buffer, attention 실행 방식과 임시 텐서 등의 비용을 포함한 결과이며, 이 표만으로 원인별 기여를 분리할 수는 없습니다. NPU에서도 메모리·속도가 좋아졌다고 해석하면 안 됩니다. Peak는 이 프로세스의 CUDA allocator 기준으로 다른 프로세스의 사용량을 직접 합산하지 않습니다. Latency에는 공유 서버의 경쟁 영향이 있을 수 있습니다.

## 저장소와 산출물

```text
configs/final.yaml                 # 유일한 본 실험 설정
pruning/                          # global L2, logit KD, 메모리 항, reducing
patch_slimming/psvit/              # 통계·복원·탐색·compact forward
patch_slimming/scripts/            # search.py / finetune.py
scripts/                          # 실행기, ONNX, 보고서, 배포/검증
scripts/profiling/                 # 예산·메모리·activation·batch probe
scripts/evaluation/               # 50,000장 FP32 평가
tests/ + patch_slimming/tests/     # 회귀 테스트
docs/                             # 방법론·실행·결과·Git 전달 안내
output/release_final_20261006/     # Git 포함: 최종 .pt 24개 + ONNX 12개 + 증거
output/final_20261006/             # 로컬 전용: 원본 실험·optimizer·로그
past/                             # 로컬 전용: 과거 자료
```

배포본은 약 **1.58 GiB**입니다. 모델별로 baseline, 예산용 target, structural 30epoch, reduced, 검색 완료, patch fine-tuned best를 보존합니다. Structural/patch 학습 checkpoint에서 optimizer 등 재개용 상태만 제외했으며 **모든 model/model_ema 텐서가 원본과 동일함을 검사**합니다. 나머지 가중치와 ONNX는 원본 바이트 그대로입니다. 학습 재개에는 로컬 원본이 필요합니다.

`.gitignore`는 원본 output을 제외하고, 배포 폴더 내부에서도 `MANIFEST.json`의 파일별 허용 목록만 포함합니다. `.pt`·`.onnx`는 **Git LFS**, 코드와 JSON/CSV/MD는 일반 Git으로 관리합니다. [Git 제출 안내](docs/GIT_RELEASE.md)를 먼저 읽어주세요.

## 내려받기와 검증

Git LFS가 설치된 환경에서:

```bash
git lfs install --local
git lfs pull
python scripts/package_release.py --verify
python scripts/verify_release_models.py
```

첫 검사는 12개 ONNX를 포함한 전체 파일 해시와 누락을 확인합니다. 두 번째는 CPU에서 raw mask+EMA를 reducing한 결과가 배포 reduced와 정확히 같은지, **배포 가중치와 ONNX 12개가 입력 3개씩에서 오차 1e-4 미만인지** 확인합니다. 전체 ImageNet 재평가는 아닙니다.

## 학습 재현

Python 3.10, PyTorch 2.6.0+cu118, torchvision 0.21.0+cu118, timm 1.0.25를 사용했습니다. 주요 의존성은 [requirements.txt](requirements.txt), 실제 환경은 [environment.json](output/release_final_20261006/provenance/environment.json)에 있습니다. 현재 CUDA/드라이버를 바꾸는 스크립트는 없습니다. 다른 환경에서는 호환되는 PyTorch wheel을 별도 설치합니다.

ImageNet은 `train/<class>/...`, `val/<class>/...` 구조가 필요하며 저장소에 배포하지 않습니다. 이번 제공 데이터의 실제 개수는 **train 1,280,297장, val 50,000장**입니다. Validation은 bicubic resize/crop, crop_pct=0.9, mean/std=(0.5,0.5,0.5)를 사용했습니다. 다른 서버의 calibration/runtime도 이 전처리를 맞춰야 합니다.

```bash
export PYTHONPATH="$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python patch_slimming/tests/run_all.py
python scripts/launch_final.py --matrix configs/final.yaml --detach
```

`.deps`는 현재 서버의 로컬 의존성 폴더이며 Git에는 없습니다. 새 환경은 requirements를 설치합니다. 재실험 시 [RUNBOOK](docs/RUNBOOK.md)에 따라 **data_path/GPU, 새 output_root·run_prefix·W&B project**를 설정해야 합니다. 배포 폴더를 학습 output으로 사용하지 마세요. 원본 실행 코드는 `provenance/source_manifest.json`에 hash로 기록돼 있으며 소스 변경 후 기존 실행 재개는 거부될 수 있습니다.

완료된 실험은 GPU 0/1/2/3에 Tiny30/Tiny50/Small30/Small50을 독립 배정했습니다. 선택 batch는 structural 384/384/192/192, patch fine-tuning 512/640/256/256입니다. 학습·검색·최종 평가는 [W&B 프로젝트](https://wandb.ai/junsu000306/vit-final-target-memory-20261006)에 단계별 run으로 기록됩니다. 자격증명은 저장소에 넣지 않습니다.

## 별도 NPU 프로젝트로 전달

전달할 파일은 모델별 `output/release_final_20261006/<model>/04_onnx/`의 `.onnx`와 `.verified.json`입니다. 각 모델의 `06_evaluation/baseline.json`에 입력 전처리 설정도 들어 있습니다. NPU 프로젝트의 기존 모델을 덮어쓰지 않도록 **새 실험 폴더**에 전달하고, 모델명·SHA-256·전처리·calibration·컴파일 옵션·NPU 코어 설정을 함께 기록합니다.

이 저장소는 `qbcompiler`/NPU runtime을 요구하지 않으며 MXQ 컴파일 명령과 실장비 벤치마킹 코드는 `npu`에서 관리합니다. FP32 ONNX parity와 INT8 MXQ 정확도는 별도 검증입니다. 자세한 전달 명령은 [Git/배포 안내](docs/GIT_RELEASE.md#npu-프로젝트로-전달)에 있습니다.
