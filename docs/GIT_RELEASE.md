# Git / 최종 산출물 배포

## 포함 / 제외

| Git 포함 | 로컬에만 보존 |
|---|---|
| 최종 pruning·KD·메모리·Patch Slimming·ONNX·평가 코드 | `past/`의 과거 코드/문서/실험 |
| `configs/final.yaml`, 테스트, 방법론·결과 문서 | ImageNet, `.deps`, `.tools`, cache, 자격증명 |
| `output/release_final_20261006/`의 명시된 파일 | `output/final_20261006/` 원본 전체 |
| 모델별 단계 최종 가중치 6개, 총 `.pt` 24개 | best/last 중복본, optimizer, 중간 search-layer checkpoint |
| ONNX 12개 및 FP32 검증 JSON | stdout `.log`, W&B 로컬 파일, batch probe 임시 checkpoint |
| 정확도/메모리 원본 JSON, 지표 JSONL, 최종 표/CSV, 환경/설정 | 참고 논문 PDF 사본, notebook 자동 백업 |

JSONL 지표와 epoch별 메모리 JSON은 실험 증거로 포함합니다. 콘솔 로그와 달리 최종 표와 패널티 계산을 검증하는 데 필요합니다. 실제 학습 코드, raw+EMA, 고정 patch ID, 사용한 통계 subset ID, 학습/평가 설정을 함께 제공합니다.

모델별 가중치:

```text
<model>/00_budget/baseline.pt
<model>/00_budget/target_untrained.pt
<model>/01_structural/checkpoint_final.pt
<model>/02_reduced/reduced.pt
<model>/03_patch_slimming_paper/searched_student.pt
<model>/03_patch_slimming_paper/compact_finetuned_best.pt
```

Baseline은 Tiny30/50 및 Small30/50 사이에서 중복되지만 동일한 모델별 레이아웃을 유지합니다. LFS는 같은 내용의 파일을 동일한 SHA-256 객체로 관리합니다. 최종 patch는 30epoch **학습 중 best**, structural은 정확히 30epoch **terminal**입니다.

## 학습 상태를 제외한 배포 가중치

`scripts/package_release.py`는 원본을 수정하지 않고 별도의 release를 생성합니다. Structural checkpoint는 `model`, `model_ema`, `epoch`, `best_acc1`, `args`를 보존하고 patch best는 모델·구조·선택 epoch를 보존합니다. Optimizer/scheduler/scaler와 학습 재개용 상태만 제외합니다. 텐서의 key/dtype/값이 원본과 정확히 같은지 재로딩해 검사합니다.

따라서 stripped checkpoint의 **파일 hash는 원본과 다릅니다.** `MANIFEST.json`에 원본 `source_sha256`과 배포 `sha256`, 제거한 key를 기록합니다. 실험 원본 JSON/CSV에 있는 경로·hash는 변경하지 않았으며, 실행 당시 출처라는 의미입니다. 이를 배포본 hash로 오해하지 마세요.

예산용 target, baseline, reduced, searched weights 및 ONNX는 파일 단위로 동일하게 복사합니다. 모델을 재학습·재양자화·재내보내기 하지 않습니다. 배포본으로 추론·reducing·ONNX 재생성은 가능하지만 원래 optimizer를 이용한 학습 재개는 불가능합니다.

```bash
# 이미 존재하는 release를 덮어쓰지 않습니다.
python scripts/package_release.py

# clone/전달/커밋 전에 파일별 hash와 12개 ONNX 검증 정보를 확인합니다.
python scripts/package_release.py --verify

# 4개 raw+EMA → reduced 텐서 동일성, 12개 ONNX ↔ 배포 가중치 CPU parity.
python scripts/verify_release_models.py
```

## Git LFS

`.gitattributes`는 **배포 폴더의 `.pt`·`.onnx` 36개만** LFS로 지정합니다. 다른 파일은 일반 Git입니다. 전체 working-tree 배포본은 약 1.58 GiB이며 일반 Git에는 모델 본문 대신 LFS pointer가 들어갑니다.

[GitHub는 일반 Git 파일의 100 MiB 초과를 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)하므로 대형 checkpoint를 일반 Git blob으로 올리지 않습니다. Push 시 LFS 저장소/전송량 제한은 계정별로 확인해야 하며, 이번 정리에서는 원격 업로드나 계정 결제 설정을 변경하지 않습니다.

현재 서버에는 공식 Git LFS 3.8.0 바이너리를 SHA-256 확인 후 `.tools/git-lfs-3.8.0/git-lfs`에 설치했습니다. Filter·alias·hook은 **이 저장소의 `.git`에만** 설정했습니다. `.tools`와 `.git`은 배포 파일이 아니며 Python/CUDA 환경이나 전역 Git 설정을 변경하지 않았습니다. 이 로컬 저장소에서는 별도 PATH 설정 없이 `git lfs ...`를 사용할 수 있습니다.

다른 서버에서는 Git LFS를 설치한 뒤:

```bash
git lfs install --local
git lfs pull
python scripts/package_release.py --verify
```

LFS가 없는 상태에서 보이는 130바이트 정도의 `.pt`/`.onnx`는 모델이 아니라 pointer입니다. 해당 파일을 MXQ 컴파일 입력으로 사용하면 안 됩니다.

## 커밋 전 확인

```bash
git status --short
git diff --cached --stat
git diff --cached --check
git lfs ls-files
python scripts/audit_git_release.py
python scripts/package_release.py --verify
```

배포 폴더 내부 `.gitignore`는 `MANIFEST.json` 파일만 허용합니다. 향후 결과 파일을 추가할 때 임의의 `git add -f output/...`로 우회하지 말고 새 release와 manifest를 생성하세요. 이번 정리에서는 승인된 코드·문서·release 및 이전 추적 파일의 삭제를 stage하고, **commit/push는 사용자가 직접 수행**합니다.

이미 과거 커밋에 포함됐던 파일은 새 커밋의 최신 트리에서 제거될 뿐 과거 Git 이력에는 남습니다. 이번 작업은 history rewrite, force push, remote 삭제를 수행하지 않습니다.

## NPU 프로젝트로 전달

이 저장소는 서버 FP32 결과까지 담당합니다. 별도 `npu` 프로젝트의 기존 자산을 덮어쓰지 않도록 새 디렉터리에 ONNX와 검증 JSON을 복사하는 예시입니다. 아래 경로는 해당 서버에 맞게 바꾸세요.

```bash
# Server_Compression 저장소 루트에서 실행
NPU_PROJECT=/workspace/etri_iitp/JS/npu
NPU_DELIVERY="$NPU_PROJECT/assets/experiments/server_final_20261006/onnx"
mkdir -p "$NPU_DELIVERY"
for model in tiny30 tiny50 small30 small50; do
  cp -n output/release_final_20261006/"$model"/04_onnx/*.onnx "$NPU_DELIVERY/"
  cp -n output/release_final_20261006/"$model"/04_onnx/*.verified.json "$NPU_DELIVERY/"
done
```

`cp -n`은 기존 파일을 덮어쓰지 않습니다. 이미 있는 파일이 같다는 보장은 없으므로 전달 후 `.verified.json`의 SHA-256과 실제 ONNX를 확인하세요. 이 복사 명령은 문서 예시이며 이번 Git 정리 작업에서는 `npu` 폴더를 변경하지 않습니다.

필수 전달 조건:

- 입력 FP32 NHWC `[1,224,224,3]`, 출력 `[1,1000]`, opset 17.
- 서버 평가 전처리: bicubic, crop_pct 0.9, mean/std 각각 `(0.5,0.5,0.5)`. 정확한 설정은 모델별 `06_evaluation/baseline.json`.
- 4모델 × baseline/reduced/patch_slimming = **12개 ONNX**.
- MXQ calibration/quantization 옵션, compiler/runtime 버전, single/multi-core 설정을 `npu`에서 별도 기록.
- ONNX FP32 parity 통과는 MXQ INT8 또는 NPU Top-1 성능을 보장하지 않음.

MXQ 변환·실제 장치 평가 결과는 `npu` 프로젝트에서 생성·관리하며 서버 보고서와 혼합해 표기하지 않습니다.
