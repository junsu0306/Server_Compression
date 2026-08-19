#!/usr/bin/env python3
"""Build JSON/Markdown summary for the final global-KD L2 experiment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def metric_memory(path):
    data = load_json(path)
    if not data: return None
    blocks = data.get("analytical_block_activation", [])
    return {
        "parameter_mib": data["parameter_bytes"] / 2**20,
        "cuda_peak_allocated_mib": data.get("cuda", {}).get("peak_allocated_bytes", 0) / 2**20,
        "cuda_peak_reserved_mib": data.get("cuda", {}).get("peak_reserved_bytes", 0) / 2**20,
        "hook_output_sum_mib": data["hook_activation"]["sum_module_output_bytes"] / 2**20,
        "block_tensor_volume_mib": sum(x["sum_tensor_bytes_upper_bound"] for x in blocks) / 2**20,
        "attention_matrix_mib": sum(x["attention_matrix_bytes"] for x in blocks) / 2**20,
        "latency_ms": data["latency"]["average_ms"],
        "images_per_second": data["latency"]["images_per_second"],
    }


def parse_eval(path):
    if not path.exists(): return None
    text = path.read_text(errors="replace")
    matches = re.findall(r"top1\s*=\s*([0-9.]+)%", text)
    return float(matches[-1]) if matches else None


def checkpoint_summary(path):
    if not path.exists(): return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {"epoch": data.get("epoch"), "best_top1": data.get("best_acc1")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/final_global_kd_l2_20260814.yaml")
    args = parser.parse_args()
    matrix = yaml.safe_load((ROOT / args.matrix).read_text())
    experiment = ROOT / matrix["output_root"]
    report = {"project": matrix["project"], "models": {}}

    for key, config in matrix["models"].items():
        root = experiment / key
        reduced_path = root / "02_reduced/reduced.pt"
        reduced = torch.load(reduced_path, map_location="cpu", weights_only=False) if reduced_path.exists() else None
        architecture = load_json(root / "03_patch_slimming_paper/architecture.json")
        schedule = None
        if architecture:
            schedule = [architecture["num_global_tokens"]] + [len(x["keep_global_ids"]) for x in architecture["layers"]]
        onnx = {}
        for stage in ("baseline", "reduced", "patch_slimming"):
            path = root / "04_onnx" / f"{key}__{stage}__nhwc_b1.onnx"
            onnx[stage] = path.stat().st_size / 2**20 if path.exists() else None
        report["models"][key] = {
            "model": config["model"], "teacher": config["teacher"],
            "target_compression": config["target_compression"],
            "structural": checkpoint_summary(root / "01_structural/checkpoint_best.pt"),
            "reduced": None if reduced is None else {
                "compression_rate": reduced.get("compression_rate"),
                "params_before": reduced.get("n_params_before"), "params_after": reduced.get("n_params_after"),
                "mlp_dims": reduced.get("mlp_dims"), "top1": parse_eval(root / "02_reduced/eval.log")},
            "patch_slimming": {
                "checkpoint": checkpoint_summary(root / "03_patch_slimming_paper/compact_finetuned_best.pt"),
                "token_schedule": schedule},
            "onnx_size_mib": onnx,
            "memory": {stage: metric_memory(root / "05_memory" / f"{key}__{stage}__b1_fp32.json")
                       for stage in ("baseline", "reduced", "patch_slimming")},
        }

    json_path = experiment / "FINAL_RESULTS.json"
    json_path.write_text(json.dumps(report, indent=2))
    lines = ["# Final Global-KD L2 + Paper Patch Slimming Results", "",
             f"W&B project: `{matrix['project']}`", "",
             "| Run | Reduced comp. | Reduced Top-1 | Patch best Top-1 | Token schedule |", "|---|---:|---:|---:|---|"]
    for key, data in report["models"].items():
        reduced = data["reduced"] or {}; patch = data["patch_slimming"]
        patch_ckpt = patch.get("checkpoint") or {}
        lines.append("| %s | %s | %s | %s | `%s` |" % (
            key, "-" if reduced.get("compression_rate") is None else "%.2f%%" % reduced["compression_rate"],
            "-" if reduced.get("top1") is None else "%.2f%%" % reduced["top1"],
            "-" if patch_ckpt.get("best_top1") is None else "%.2f%%" % patch_ckpt["best_top1"],
            patch.get("token_schedule") or "-"))
    lines += ["", "## Memory and activation (batch 1, FP32)", "",
              "| Run/stage | Params MiB | CUDA peak MiB | Hook act. MiB | Attention MiB | Latency ms |", "|---|---:|---:|---:|---:|---:|"]
    for key, data in report["models"].items():
        for stage, memory in data["memory"].items():
            if not memory: continue
            lines.append("| %s/%s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (
                key, stage, memory["parameter_mib"], memory["cuda_peak_allocated_mib"],
                memory["hook_output_sum_mib"], memory["attention_matrix_mib"], memory["latency_ms"]))
    markdown = experiment / "FINAL_RESULTS.md"; markdown.write_text("\n".join(lines) + "\n")
    print("saved %s and %s" % (json_path, markdown))


if __name__ == "__main__":
    main()
