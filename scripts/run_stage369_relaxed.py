#!/usr/bin/env python3
"""Resumable relaxed 3/6/9 Patch Slimming pipeline for the four final L2 models."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "patch_slimming/scripts/stage_ablation.py"
FINETUNE = ROOT / "patch_slimming/scripts/finetune.py"
EXPORT = ROOT / "patch_slimming/scripts/export_onnx.py"


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def complete(path, epochs):
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt.get("epoch", -1)) + 1 >= epochs


def run(cmd, gpu, log):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    deps = str(ROOT / ".deps")
    env["PYTHONPATH"] = deps + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        stream.flush()
        subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def materialize(matrix):
    root = ROOT / matrix["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    for key, model in matrix["models"].items():
        out = root / key / "stage369_relaxed"
        out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "method": "patch_slimming_static_stage369_relaxed",
            "model": {"name": model["model"], "num_classes": 1000,
                      "reduced_checkpoint": str((ROOT / model["reduced_checkpoint"]).resolve())},
            "data_path": matrix["data_path"], "output_dir": str(out),
            "final_finetune": {**matrix["final_finetune"], "batch_size": model["batch_size"],
                "wandb": True, "wandb_project": matrix["project"],
                "wandb_run_name": f"{key}__stage369_171_151_131__finetune30",
                "wandb_run_id": f"stage369-relaxed-{key}-20260826"},
        }
        with open(root / key / "finetune.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)


def process(matrix, key, phase):
    model = matrix["models"][key]
    gpu = int(model["gpu"])
    root = ROOT / matrix["output_root"] / key
    out = root / "stage369_relaxed"
    counts = list(map(str, matrix["stage369"]["counts"]))
    mask_outputs = [out / name for name in
                    ("architecture.json", "searched_student.pt", "mean_scores.pt", "metrics.json")]
    if phase in ("mask", "all") and not all(path.exists() for path in mask_outputs):
        val_batch = matrix["stage369"]["validation_batch_size_tiny" if "tiny" in key else "validation_batch_size_small"]
        run([sys.executable, STAGE, "--reduced", ROOT / model["reduced_checkpoint"],
             "--output-dir", root / "mask_search", "--data-path", matrix["data_path"],
             "--gpu", 0, "--calibration-samples", matrix["stage369"]["calibration_samples"],
             "--calibration-batch-size", matrix["stage369"]["calibration_batch_size"],
             "--val-batch-size", val_batch, "--workers", matrix["stage369"]["workers"],
             "--counts", *counts, "--policy-name", "stage369_relaxed", "--wandb",
             "--wandb-project", matrix["project"]], gpu, root / "mask_search.log")
        source = root / "mask_search/stage369_relaxed"
        # The fine-tuner expects the selected architecture and starting weights
        # directly in output_dir. Keep the search report in its own directory.
        for name in ("architecture.json", "searched_student.pt", "mean_scores.pt", "metrics.json"):
            shutil.copy2(source / name, out / name)
    epochs = int(matrix["final_finetune"]["epochs"])
    last = out / "compact_finetuned_last.pt"
    if phase in ("finetune", "all") and not complete(last, epochs):
        cmd = [sys.executable, FINETUNE, "--config", root / "finetune.yaml"]
        if last.exists():
            cmd.append("--resume")
        run(cmd, gpu, out / "finetune.log")
    onnx = root / f"{key}__patch_slimming_stage369_relaxed__nhwc_b1_npusafe.onnx"
    if phase in ("export", "all") and not onnx.exists():
        run([sys.executable, EXPORT, "--arch", out / "architecture.json",
             "--weights", out / "compact_finetuned_best.pt", "--output", onnx, "--verify"],
            gpu, out / "export.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final_global_kd_l2_stage369_relaxed_20260826.yaml")
    ap.add_argument("--gpu", type=int, choices=[6, 7])
    ap.add_argument("--phase", choices=["materialize", "mask", "finetune", "export", "all"], required=True)
    args = ap.parse_args()
    matrix = load(ROOT / args.config)
    materialize(matrix)
    if args.phase == "materialize":
        return
    if args.gpu is None:
        ap.error("--gpu is required")
    for key, model in matrix["models"].items():
        if int(model["gpu"]) == args.gpu:
            print(f"[queue] gpu={args.gpu} model={key} phase={args.phase}", flush=True)
            process(matrix, key, args.phase)


if __name__ == "__main__":
    main()
