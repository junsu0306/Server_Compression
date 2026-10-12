# 실행 및 산출물 관리

## 준비

저장소 루트에서 실행합니다. 데이터 경로, output_root, W&B project/run_prefix, GPU 배정은 `configs/final.yaml`에서 관리합니다. 현재 설정은 GPU 0/1/2/3에 Tiny30/Tiny50/Small30/Small50을 각각 배정합니다. 사용 중인 GPU에 임의로 다른 실험을 겹쳐 실행하지 않습니다.

```bash
cd /workspace/etri_iitp/JS/Server_Compression
export PYTHONPATH="$PWD/.deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
nvidia-smi
```

현재 사용 interpreter는 `/opt/conda/envs/efficientvit/bin/python`입니다. 새 환경에서 설치한다면 requirements의 torch/torchvision을 지원되는 CUDA wheel로 구성하고 아래 회귀 테스트를 먼저 수행해야 합니다. 공용 서버의 드라이버/시스템 CUDA는 이 실행 과정에서 설치하거나 업데이트하지 않습니다.

## 테스트

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/efficientvit/bin/python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/efficientvit/bin/python patch_slimming/tests/run_all.py
CUDA_VISIBLE_DEVICES='' /opt/conda/envs/efficientvit/bin/python tests/verify_exports.py
```

ImageNet이나 pretrained 다운로드 없이 CPU에서 수행됩니다. 실제 데이터와 CUDA 프로파일링은 실행기의 `budget` 단계에서 확인합니다.

## 전체 실행

```bash
/opt/conda/envs/efficientvit/bin/python scripts/launch_final.py --matrix configs/final.yaml --detach
```

터미널 연결이 끝나도 detached supervisor가 유지됩니다. GPU별로 budget → structural/reducing → patch search/fine-tuning → export → memory → evaluate를 수행합니다. Stage별 subprocess는 지정한 GPU만 `CUDA_VISIBLE_DEVICES`로 노출합니다. 따라서 worker 내부의 `cuda:0`은 해당 physical GPU를 의미합니다.

Batch probe의 작은 synthetic 학습은 본 checkpoint를 변경하지 않습니다. 예산 JSON을 먼저 만든 다음 probe와 실제 학습에서 같은 예산을 읽습니다. Full experiment는 며칠이 걸릴 수 있으며 epoch 수를 자동으로 줄이지 않습니다.

## 상태와 실패

```bash
cat output/final_20261006/supervisor_status.json
cat output/final_20261006/gpu0_status.json
tail -n 40 output/final_20261006/supervisor.log
tail -n 40 output/final_20261006/tiny30/01_structural/train.log
tail -n 40 output/final_20261006/tiny30/03_patch_slimming_paper/search.log
```

마지막 두 파일은 해당 단계가 시작된 뒤 생성됩니다. `running`은 실행 중, `failed`는 오류, `complete`는 worker의 전체 단계 종료입니다. 보고서에서도 필수 평가/메모리/ONNX 해시를 확인해 `complete` 여부를 판단합니다.

오류 시 마지막 로그를 먼저 확인합니다. GPU reset, 다른 사용자의 프로세스 종료, CUDA 업데이트는 자동으로 수행하지 않습니다. 수정된 코드로 실험할 때는 새 output_root/run_prefix를 사용합니다. Supervisor는 실행 당시 코드 hash와 다르면 같은 실험의 재개를 거부합니다.

## 재개와 단계별 실행

Supervisor가 종료된 상태이고 코드/설정이 동일하면 전체 실행 명령을 다시 호출합니다. 완료된 stage는 건너뛰고 structural/compact 학습은 마지막 epoch, search는 완료된 layer 기준으로 재개합니다. 완전한 bitwise 재현을 보장하지는 않습니다.

Structural 학습은 30epoch에서 종료합니다. 마지막 체크포인트의 완료 epoch가 정확히 30인지 검사한 뒤 `checkpoint_final.pt`로 고정하여 후속 단계에 사용합니다. 이미 30epoch을 넘긴 last 모델을 30epoch 결과로 간주하지 않습니다. LR horizon은 50epoch으로 별도 고정되어 있고, 재개 시 저장된 optimizer/scheduler를 이어받습니다. 중간에 중단된 미완료 epoch는 마지막 완료 checkpoint에서 다시 수행합니다.

특정 GPU의 단계만 실행할 수도 있습니다. 실행 중인 worker와 같은 GPU queue를 동시에 실행할 수 없습니다.

```bash
/opt/conda/envs/efficientvit/bin/python scripts/run_pipeline.py --phase materialize
/opt/conda/envs/efficientvit/bin/python scripts/run_pipeline.py --gpu 0 --phase budget
/opt/conda/envs/efficientvit/bin/python scripts/run_pipeline.py --gpu 0 --phase all
```

지원 phase: `materialize`, `budget`, `structural`, `patch`, `export`, `memory`, `evaluate`, `all`. 개별 phase 실행 시 선행 결과가 필요합니다. 예산 없이 structural을 시작하면 오류로 중단합니다.

## 개별 메모리 측정

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/efficientvit/bin/python scripts/profiling/profile_reduced_memory.py \
  --checkpoint output/final_20261006/tiny30/00_budget/baseline.pt \
  --device cuda:0 --batch-size 1 --warmups 10 --runs 50 --details \
  --output output/final_20261006/tiny30/05_memory/baseline_manual.json
```

Reduced는 checkpoint를 `02_reduced/reduced.pt`로 바꿉니다. Patch는 `--checkpoint` 대신 `--arch .../03_patch_slimming_paper/architecture.json --weights .../compact_finetuned_best.pt`를 지정합니다. `--input-npy`로 FP32 NHWC 공통 입력을 지정할 수도 있습니다. 측정은 batch1 추론 메모리이며 학습 batch와 다릅니다.

## 결과/ONNX 전달

```bash
/opt/conda/envs/efficientvit/bin/python scripts/build_report.py
```

`reports/FINAL_RESULTS.md`와 CSV/JSON을 생성합니다. 완료 후 `tiny30`, `tiny50`, `small30`, `small50` 각각의 `04_onnx/`에서 3종 `.onnx`와 `.verified.json`을 전달합니다. 총 ONNX는 12개입니다. 파일명은 `<model>__<stage>__nhwc_b1_npusafe.onnx`입니다.

다른 서버의 `npu` 프로젝트에서 MXQ로 컴파일하고 실장비 정확도를 측정합니다. 서버 측 `npusafe`는 정적 shape/연산 목록 검사라는 의미이지 해당 compiler 버전과 양자화의 실행 성공 보증이 아닙니다. 이 서버 실험은 NPU 결과를 생성하거나 대신 추정하지 않습니다.

## 제출 범위

코드·설정·문서와 `output/release_final_20261006/`의 최종 checkpoint/ONNX/profile/report를 저장소에 포함합니다. `.pt`/`.onnx`는 Git LFS입니다. 원본 `output/final_20261006/`, `past/`, data, credentials, `.deps`, `.tools`, cache, stdout/W&B는 제외합니다. Release는 추론용이므로 optimizer를 포함한 원본 실행의 재개에는 사용할 수 없습니다. [Git/LFS 배포 안내](GIT_RELEASE.md)를 참고하세요. Commit/push는 사용자가 직접 수행합니다.

```bash
python scripts/package_release.py --verify
python scripts/verify_release_models.py
python scripts/audit_git_release.py
```

새 clone에서 처음부터 학습하려면 `configs/final.yaml`의 data_path/GPU와 새 output_root/run_prefix/project를 지정합니다. Git에 포함된 release를 output_root로 사용하지 않습니다. 최종 가중치를 받아 평가하거나 재내보내기만 할 때는 `git lfs pull` 후 release 경로를 직접 넘깁니다. 예를 들어 최종 Tiny30 patch ONNX를 새 위치에 생성하려면:

```bash
python scripts/export_final_onnx.py --kind patch --npu-safe \
  --arch output/release_final_20261006/tiny30/03_patch_slimming_paper/architecture.json \
  --weights output/release_final_20261006/tiny30/03_patch_slimming_paper/compact_finetuned_best.pt \
  --output output/local_reexport/tiny30_patch.onnx
```
