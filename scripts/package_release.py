"""Build or verify the explicit, inference-only Git release of a completed run.

Never modifies the source run. Model tensors are preserved exactly; only the
two training checkpoint types lose optimizer/resume state. No training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("tiny30", "tiny50", "small30", "small50")
STAGES = ("baseline", "reduced", "patch_slimming")
DEFAULT_RELEASE = ROOT / "output/release_final_20261006"
STRUCTURAL_KEYS = ("model", "model_ema", "epoch", "best_acc1", "args")
PATCH_KEYS = ("model", "model_name", "architecture", "epoch", "best_acc1", "mlp_dims")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def slim_checkpoint(source, destination, keep):
    """Keep loader-compatible raw/EMA tensors and validate the serialized copy."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    missing = set(keep) - checkpoint.keys()
    if missing:
        raise ValueError(f"Missing required checkpoint keys: {sorted(missing)}")
    result = {key: checkpoint[key] for key in keep}
    result["release_inference_only"] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    for name in ("model", "model_ema"):
        if name not in result:
            continue
        if result[name].keys() != loaded[name].keys():
            raise ValueError(f"Tensor names changed: {name}")
        for key, tensor in result[name].items():
            if tensor.dtype != loaded[name][key].dtype or not torch.equal(tensor, loaded[name][key]):
                raise ValueError(f"Tensor changed during packaging: {name}/{key}")
    return {"kept_keys": list(result), "removed_keys": sorted(set(checkpoint) - set(keep)),
            "completed_epochs": checkpoint["epoch"] + 1, "tensor_equality_verified": True}


def allowlist_text(paths):
    # Directories stay traversable; only exact file paths are allowed.
    return "# Generated release allowlist. Do not add training logs/checkpoints here.\n*\n!*/\n!.gitignore\n!MANIFEST.json\n" + "".join(
        f"!/{name}\n" for name in sorted(paths))


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe release path: {name}")
    return path


def verify_release(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    entries = manifest["files"]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate manifest paths")
    for entry in entries:
        path = root / safe_relative(entry["path"])
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f"Symlink/escaped release file: {path}")
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Missing/changed artifact (run git lfs pull after cloning): {path}")
    expected = set(paths) | {"MANIFEST.json", ".gitignore"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Unexpected/missing release files: {actual ^ expected}")
    if (root / ".gitignore").read_text() != allowlist_text(paths):
        raise ValueError("Release allowlist differs from manifest")
    onnx_paths = [name for name in paths if name.endswith(".onnx")]
    if len(onnx_paths) != 12:
        raise ValueError(f"Expected 12 ONNX files, got {len(onnx_paths)}")
    for key in KEYS:
        for stage in STAGES:
            path = root / key / "04_onnx" / f"{key}__{stage}__nhwc_b1_npusafe.onnx"
            marker = json.loads(path.with_suffix(".verified.json").read_text())
            if marker["onnx_sha256"] != sha256(path):
                raise ValueError(f"ONNX parity marker hash mismatch: {path}")
            if not marker["npu_safe"] or marker["verify_samples"] < 3:
                raise ValueError(f"Incomplete ONNX parity verification: {path}")
            if not 0 <= marker["pytorch_ort_max_abs"] < 1e-4:
                raise ValueError(f"ONNX parity tolerance failed: {path}")
    return {"files": len(entries), "pt_files": sum(p.endswith(".pt") for p in paths),
            "onnx_files": len(onnx_paths), "total_bytes": sum(e["bytes"] for e in entries),
            "state": "verified"}


def build_release(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and release directories must be separate")
    if destination.exists():
        raise FileExistsError(f"Release already exists; use --verify or a new --destination: {destination}")
    status = json.loads((source / "supervisor_status.json").read_text())
    report = json.loads((source / "reports/FINAL_RESULTS.json").read_text())
    if status["state"] != "complete" or report["state"] != "complete":
        raise ValueError("Only completed experiments can be packaged")
    if any(worker["exit_code"] != 0 for worker in status["workers"].values()):
        raise ValueError("A worker did not exit successfully")
    destination.mkdir(parents=True)
    entries = []

    def record(relative, source_file=None, **extra):
        target = destination / relative
        entry = {"path": relative, "bytes": target.stat().st_size, "sha256": sha256(target), **extra}
        if source_file is not None:
            entry.update(source=str(source_file.relative_to(source)), source_sha256=sha256(source_file))
        entries.append(entry)

    def copy(relative):
        src = source / relative
        if not src.is_file() or src.is_symlink():
            raise ValueError(f"Not a regular source file: {src}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        record(relative, src, action="byte_identical_copy")

    selections = {}
    for key in KEYS:
        print(f"Packaging {key}", flush=True)
        structural = source / key / "01_structural"
        terminal = json.loads((structural / "terminal_checkpoint.json").read_text())
        if terminal["completed_epochs"] != 30 or terminal["sha256"] != sha256(structural / "checkpoint_final.pt"):
            raise ValueError(f"Not the pinned 30-epoch checkpoint: {key}")
        metrics = [json.loads(line) for line in (source / key / "03_patch_slimming_paper/metrics.jsonl").read_text().splitlines()]
        if metrics[-1]["epoch"] != 29:
            raise ValueError(f"Patch fine-tuning did not finish 30 epochs: {key}")
        direct = ["00_budget/baseline.pt", "00_budget/target_untrained.pt", "00_budget/budget.json",
                  "00_budget/baseline_profile.json", "00_budget/target_profile.json",
                  "01_structural/terminal_checkpoint.json", "01_structural/run_metadata.json",
                  "01_structural/metrics.jsonl", "02_reduced/reduced.pt",
                  "03_patch_slimming_paper/architecture.json", "03_patch_slimming_paper/searched_student.pt",
                  "03_patch_slimming_paper/score_ids.txt", "03_patch_slimming_paper/reconstruction_train_ids.txt",
                  "03_patch_slimming_paper/reconstruction_eval_ids.txt", "03_patch_slimming_paper/search_records.jsonl",
                  "03_patch_slimming_paper/metrics.jsonl", "configs/structural.yaml", "configs/patch_slimming_paper.yaml",
                  "batch_probe/structural.json", "batch_probe/patch.json"]
        for relative in direct:
            copy(f"{key}/{relative}")
        for relative, keep in (("01_structural/checkpoint_final.pt", STRUCTURAL_KEYS),
                               ("03_patch_slimming_paper/compact_finetuned_best.pt", PATCH_KEYS)):
            relative = f"{key}/{relative}"
            meta = slim_checkpoint(source / relative, destination / relative, keep)
            record(relative, source / relative, action="strip_training_state", **meta)
            selections[relative] = meta
        # Measurements are evidence, not transient process logs. Preserve all 30.
        profiles = sorted((structural / "memory_budget").glob("epoch_*.json"))
        if len(profiles) != 30:
            raise ValueError(f"Expected 30 epoch memory profiles: {key}")
        for path in profiles:
            copy(str(path.relative_to(source)))
        for stage in STAGES:
            stem = f"{key}/04_onnx/{key}__{stage}__nhwc_b1_npusafe"
            copy(stem + ".onnx")
            copy(stem + ".verified.json")
            copy(f"{key}/05_memory/{key}__{stage}__b1_fp32.json")
        for stage in (*STAGES, "pruned"):
            copy(f"{key}/06_evaluation/{stage}.json")
    for name in ("FINAL_RESULTS.md", "FINAL_RESULTS.json", "accuracy.csv", "memory.csv", "budget.csv", "onnx.csv"):
        copy(f"reports/{name}")
    for name in ("environment.json", "source_manifest.json", "preflight/checks.json"):
        copy(f"provenance/{name}")
    # Preserve the executed matrix and requirements, not a duplicate source tree.
    copy("provenance/source/configs/final.yaml")
    copy("provenance/source/requirements.txt")
    # Bind runtime source to its recorded hash. Documentation/release tooling may evolve.
    hashes = json.loads((source / "provenance/source_manifest.json").read_text())
    for name, expected in hashes.items():
        if name.endswith(".py") and sha256(ROOT / name) != expected:
            raise ValueError(f"Runtime code changed since this experiment: {name}")
    readme = """# 최종 결과 배포 패키지

원본 실행: `output/final_20261006` (서버 단계 완료: 2026-10-08 UTC).
이 폴더는 학습 재개용 디렉터리가 아니라 최종 추론·평가·ONNX 전달용입니다.

- 모델별 `00_budget`: pretrained baseline, 추가 학습 없는 예산용 target, 예산/측정 JSON.
- `01_structural`: 정확히 30epoch의 raw+EMA 가중치, 30회 후보 메모리 측정, 지표.
- `02_reduced`: 실제 FFN 채널을 제거한 최종 모델.
- `03_patch_slimming_paper`: 검색 종료 가중치, 고정 패치 ID, subset ID, 30epoch 학습 중 best 가중치/지표.
- `04_onnx`: baseline/reduced/patch-slimming ONNX와 원본 수치 검증 JSON. 전체 12개.
- `05_memory`, `06_evaluation`, `reports`: 실제 측정/평가 및 최종 표.
- `configs`, `batch_probe`, `provenance`: 실행 설정, 선택 batch, 환경/실행 코드 hash.

`checkpoint_final.pt`와 `compact_finetuned_best.pt`에서 optimizer/scheduler/scaler 등
학습 재개용 상태만 제거했습니다. 원본 model/model_ema 텐서는 모든 원소가 동일함을 검사했습니다.
Structural checkpoint의 raw mask와 EMA를 둘 다 보존하므로 기존 reducing 코드로 읽을 수 있습니다.
다른 `.pt`와 ONNX는 원본 바이트를 그대로 복사했습니다. 학습은 재실행하지 않았습니다.

`terminal_checkpoint.json`, 예산/평가 JSON, 보고서의 경로·hash는 **실험 당시 원본 출처**입니다.
원본과 포장된 파일의 대응/hash는 `MANIFEST.json`의 source_sha256 / sha256을 사용하세요.
절대 경로는 실행 기록이지 다른 서버에 필요한 경로가 아닙니다. 모델 로더에는 이 폴더 경로를 직접 넘깁니다.
Git clone 후 `git lfs pull`을 수행하고 저장소 루트에서 다음 명령으로 전체 파일을 검증합니다.

```bash
python scripts/package_release.py --verify
python scripts/verify_release_models.py
```

최종 정확도는 PyTorch FP32 ImageNet validation 50,000장 평가입니다.
ONNX는 FP32 동등성 검사이며 MXQ INT8/NPU 정확도를 측정한 것이 아닙니다.
MXQ 변환 및 실제 NPU 정확도·속도·메모리 측정은 별도 `npu` 프로젝트에서 수행합니다.

원본 로그·optimizer·중간 search checkpoint·W&B 디렉터리·데이터·past는 Git에 포함하지 않습니다.
학습 재개에는 서버에 보존된 원본 실험 폴더가 필요합니다.
"""
    (destination / "README.md").write_text(readme)
    record("README.md", action="generated_documentation")
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                "source_completed_utc": status["updated_utc"], "inference_only": True,
                "checkpoint_selection": selections, "files": sorted(entries, key=lambda x: x["path"])}
    write_json(destination / "MANIFEST.json", manifest)
    (destination / ".gitignore").write_text(allowlist_text([e["path"] for e in entries]))
    return verify_release(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output/final_20261006")
    parser.add_argument("--destination", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = verify_release(args.destination) if args.verify else build_release(args.source, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
