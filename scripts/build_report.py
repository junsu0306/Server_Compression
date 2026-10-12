"""Generate submission tables from this run's evidence; missing values stay pending."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("baseline", "reduced", "patch_slimming")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default="configs/final.yaml")
    args = ap.parse_args()
    matrix = yaml.safe_load((ROOT / args.matrix).read_text())
    root = ROOT / matrix["output_root"]
    report_dir = root / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    result = {"method": matrix["method"], "project": matrix["project"],
              "structural_epochs": matrix["structural"]["epochs"],
              "lr_schedule_epochs": matrix["structural"]["lr_schedule_epochs"],
              "updated_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    accuracy_rows = []; memory_rows = []; budget_rows = []; onnx_rows = []
    complete = True
    for key, config in matrix["models"].items():
        folder = root / key
        budget = read(folder / "00_budget/budget.json")
        evaluation = {stage: read(folder / f"06_evaluation/{stage}.json")
                      for stage in ("baseline", "pruned", "reduced", "patch_slimming")}
        memory = {stage: read(folder / "05_memory" / f"{key}__{stage}__b1_fp32.json") for stage in STAGES}
        arch = read(folder / "03_patch_slimming_paper/architecture.json")
        terminal = read(folder / "01_structural/terminal_checkpoint.json")
        onnx = {}
        for stage in STAGES:
            path = folder / "04_onnx" / f"{key}__{stage}__nhwc_b1_npusafe.onnx"
            verification = read(path.with_suffix(".verified.json"))
            valid = bool(path.exists() and verification and
                         hashlib.sha256(path.read_bytes()).hexdigest() == verification["onnx_sha256"])
            onnx[stage] = {"path": str(path), "verified": valid, "verification": verification}
            if valid:
                onnx_rows.append({"model": key, "stage": stage, "file": str(path.relative_to(root)),
                                  "size_mib": path.stat().st_size / 2**20,
                                  "pytorch_ort_max_abs": verification["pytorch_ort_max_abs"],
                                  "sha256": verification["onnx_sha256"]})
        training_memory = [read(p) for p in sorted((folder / "01_structural/memory_budget").glob("epoch_*.json"))]
        if budget:
            budget_rows.append({"model": key, "baseline_peak_mib": budget["baseline_profile"]["peak_allocated_bytes"] / 2**20,
                                "target_peak_mib": budget["max_mib"], "target_trained_steps": budget["trained_steps"],
                                "target_parameters": budget["target_profile"]["parameters"],
                                "candidate_measurements": len(training_memory),
                                "candidate_max_mib": max((p["peak_allocated_bytes"] / 2**20 for p in training_memory), default=None)})
        for stage, data in evaluation.items():
            if data:
                accuracy_rows.append({"model": key, "stage": stage, "parameters": data["parameters"],
                                      "weight_mib": data["parameters"] * 4 / 2**20,
                                      "top1": data["metrics"]["acc1"], "top5": data["metrics"]["acc5"],
                                      "samples": data["samples"]})
        for stage, data in memory.items():
            if data:
                memory_rows.append({"model": key, "stage": stage,
                                    "weight_mib": data["parameter_bytes"] / 2**20,
                                    "peak_allocated_mib": data["peak_allocated_bytes"] / 2**20,
                                    "transient_peak_mib": data["transient_peak_bytes"] / 2**20,
                                    "module_output_sum_mib": data["activation"]["module_output_sum_bytes"] / 2**20,
                                    "latency_mean_ms": data["latency_mean_ms"]})
        ready = bool(budget and terminal and terminal["completed_epochs"] == matrix["structural"]["epochs"]
                     and all(evaluation.values()) and all(memory.values()) and all(v["verified"] for v in onnx.values()))
        complete &= ready
        result["models"][key] = {"complete": ready, "budget": budget,
                                "evaluation": evaluation, "memory": memory,
                                "architecture": arch, "onnx": onnx, "structural_terminal": terminal}
    result["state"] = "complete" if complete else "in_progress"
    lines = ["# 최종 실험 결과", "", f"상태: `{result['state']}` · W&B: `{matrix['project']}`", "",
             "이 문서는 현재 실험의 원본 JSON에서 자동 생성됩니다. 미완료 항목은 결과로 제시하지 않습니다.", "",
             f"Structural 학습 종료: {matrix['structural']['epochs']} epoch. 후속 단계는 정확히 해당 epoch의 checkpoint_final.pt를 사용합니다.",
             f"LR schedule horizon: {matrix['structural']['lr_schedule_epochs']} epoch (학습 종료 epoch와 별도).", "",
             "## 메모리 예산", "", "학습 없이 baseline을 global L2 pruning/reducing한 목표 구조의 CUDA peak를 M_max로 고정합니다.", ""]
    tables = [("budget", budget_rows, "사전 예산 및 후보 측정"), ("accuracy", accuracy_rows, "FP32 ImageNet 검증"),
              ("memory", memory_rows, "서버 메모리"), ("onnx", onnx_rows, "ONNX 산출물")]
    for name, rows, title in tables:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["측정/생성 대기 중.", ""]); continue
        fields = list(rows[0])
        with (report_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            values = ["대기" if row[k] is None else f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]) for k in fields]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## 측정 해석", "",
              "- 정확도는 ImageNet validation 50,000장 FP32 재평가이며 학습 중 AMP/EMA best 지표와 구분합니다.",
              "- Pruned는 dense shape에 zero mask를 적용한 모델, reduced는 해당 채널을 물리 제거한 모델입니다.",
              "- 메모리는 batch 1, FP32, random input, warmup 10회 후 forward 50회 중 allocator 최대값입니다. 평균 peak가 아닙니다.",
              "- Module output 합은 별도 1회 forward의 출력 텐서 용량 합이며 동시에 살아 있는 메모리의 peak가 아닙니다.",
              "- 메모리 초과 패널티는 측정 기반 상수 항입니다. gradient나 자동 추가 pruning을 제공하지 않습니다.",
              "- ONNX verification은 FP32 수치 동등성 확인입니다. MXQ/NPU 정확도는 별도 장비 검증 대상입니다.", ""]
    for path, content in ((report_dir / "FINAL_RESULTS.json", json.dumps(result, indent=2) + "\n"),
                          (report_dir / "FINAL_RESULTS.md", "\n".join(lines))):
        temp = path.with_suffix(path.suffix + ".tmp"); temp.write_text(content); temp.replace(path)
    print(f"{result['state']}: {report_dir}")


if __name__ == "__main__":
    main()
